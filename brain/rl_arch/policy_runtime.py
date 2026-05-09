from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

try:
    from .policy_model import ControllerPolicyNet
    from .state_encoder import encode_state
    # Try to import A2C model for deployment
    try:
        from actor_critic.a2c_model import ActorCriticNet
    except ImportError:
        # Handle standalone or relative path issues
        sys.path.append(str(Path(__file__).resolve().parents[1] / "actor_critic"))
        from a2c_model import ActorCriticNet
except ImportError:
    from policy_model import ControllerPolicyNet
    from state_encoder import encode_state
    try:
        from actor_critic.a2c_model import ActorCriticNet
    except ImportError:
        sys.path.append(str(Path(__file__).resolve().parents[1] / "actor_critic"))
        from a2c_model import ActorCriticNet


ACTION_TO_ID = {
    "retrieve": 0,
    "rewrite_query": 1,
    "answer": 2,
    "verify": 3,
    "stop": 4,
}
ID_TO_ACTION = {v: k for k, v in ACTION_TO_ID.items()}


def _is_numeric_sequence(x: Any) -> bool:
    if not isinstance(x, (list, tuple)):
        return False
    return all(isinstance(v, (int, float, bool)) for v in x)


class PolicyRuntime:
    def __init__(self):
        self.model = None
        self.loaded = False
        self.error = ""
        self.input_dim = 32
        self.num_actions = 5
        self._load_checkpoint()

    def _checkpoint_path(self) -> Path:
        # [NEW] Check for A2C policy first (Phase 4 Goal)
        a2c_path = Path(__file__).resolve().parents[1] / "actor_critic" / "checkpoints" / "a2c_policy.pt"
        if a2c_path.exists():
            return a2c_path
            
        # Fallback to standard policy
        return Path(__file__).resolve().parent / "data" / "checkpoints" / "phase4_policy.pt"

    def _load_checkpoint(self) -> None:
        ckpt_path = self._checkpoint_path()
        if not ckpt_path.exists():
            self.error = f"Checkpoint not found: {ckpt_path}"
            return

        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")

            self.input_dim = int(checkpoint.get("input_dim", 32))
            self.num_actions = int(checkpoint.get("num_actions", 5))

            # Detect if this is an A2C model or a standard policy
            if "a2c_policy.pt" in str(ckpt_path):
                self.model = ActorCriticNet(
                    input_dim=self.input_dim,
                    hidden_dim=int(checkpoint.get("hidden_dim", 128)),
                    output_dim=self.num_actions,
                )
            else:
                self.model = ControllerPolicyNet(
                    input_dim=self.input_dim,
                    hidden_dim=128,
                    output_dim=self.num_actions,
                )
                
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.loaded = True
            self.error = ""
        except Exception as e:
            self.loaded = False
            self.error = str(e)

    def _coerce_feature_vector(self, encoded: Any) -> List[float]:
        # Case 1: dict of feature_name -> numeric_value
        if isinstance(encoded, dict):
            if "feature_vector" in encoded and _is_numeric_sequence(encoded["feature_vector"]):
                return [float(v) for v in encoded["feature_vector"]]

            if "values" in encoded and _is_numeric_sequence(encoded["values"]):
                return [float(v) for v in encoded["values"]]

            numeric_items = []
            for k, v in encoded.items():
                if isinstance(v, (int, float, bool)):
                    numeric_items.append((k, float(v)))

            if numeric_items:
                return [v for _, v in numeric_items]

        # Case 2: direct numeric list/tuple
        if _is_numeric_sequence(encoded):
            return [float(v) for v in encoded]

        # Case 3: tuple/list wrapper
        if isinstance(encoded, (list, tuple)):
            for item in encoded:
                if _is_numeric_sequence(item):
                    return [float(v) for v in item]

        raise ValueError("Could not convert encode_state output into a numeric feature vector")

    def predict(self, state: Dict[str, Any], valid_actions: List[str]) -> Dict[str, Any]:
        result = {
            "loaded": self.loaded,
            "error": self.error,
            "action": "",
            "confidence": 0.0,
            "probabilities": {},
        }

        if not self.loaded or self.model is None:
            return result

        if not valid_actions:
            result["error"] = "No valid actions supplied"
            return result

        try:
            encoded = encode_state(state)
            vector = self._coerce_feature_vector(encoded)

            x = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
            
            # Handle A2C model forward pass (returns logits, value)
            if isinstance(self.model, ActorCriticNet):
                logits, _ = self.model(x)
                logits = logits.squeeze(0)
            else:
                logits = self.model(x).squeeze(0)

            masked_logits = torch.full_like(logits, float("-inf"))
            for action in valid_actions:
                masked_logits[ACTION_TO_ID[action]] = logits[ACTION_TO_ID[action]]

            probs = F.softmax(masked_logits, dim=0)

            top_id = int(torch.argmax(probs).item())
            top_action = ID_TO_ACTION[top_id]
            top_conf = float(probs[top_id].item())

            prob_map = {}
            for action in valid_actions:
                prob_map[action] = float(probs[ACTION_TO_ID[action]].item())

            result.update(
                {
                    "action": top_action,
                    "confidence": top_conf,
                    "probabilities": prob_map,
                }
            )
            return result

        except Exception as e:
            result["loaded"] = False
            result["error"] = str(e)
            return result


POLICY_RUNTIME = PolicyRuntime()