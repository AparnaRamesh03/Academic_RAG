import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------
# Robust imports
# ---------------------------------------------------------
try:
    from . import config as rl_config
except ImportError:
    import config as rl_config  # type: ignore

try:
    from .state_encoder import encode_state
except ImportError:
    from state_encoder import encode_state  # type: ignore

try:
    from . import reward as reward_module
except ImportError:
    import reward as reward_module  # type: ignore


# ---------------------------------------------------------
# Config fallbacks
# ---------------------------------------------------------
ENABLE_TRAJECTORY_LOGGING = getattr(rl_config, "ENABLE_TRAJECTORY_LOGGING", True)

_DEFAULT_TRAJECTORY_DIR = Path(__file__).resolve().parent / "data" / "trajectories"
TRAJECTORY_DIR = Path(getattr(rl_config, "TRAJECTORY_DIR", _DEFAULT_TRAJECTORY_DIR))

TRAJECTORY_FILE_PREFIX = getattr(rl_config, "TRAJECTORY_FILE_PREFIX", "phase4_traj")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_scalar_number(x: Any) -> bool:
    return isinstance(x, (int, float, bool))


def _is_numeric_sequence(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and all(_is_scalar_number(v) for v in x)


def _coerce_encoded_state_to_feature_map(
    encoded: Any,
    existing_feature_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Convert many possible encode_state outputs into:
        {feature_name: float_value}
    """

    # Case 1: dict already looks like a feature map
    if isinstance(encoded, dict):
        numeric_items = [(k, float(v)) for k, v in encoded.items() if _is_scalar_number(v)]
        if numeric_items:
            return {k: v for k, v in numeric_items}

        # Case 2: dict contains explicit names + vector
        names = encoded.get("feature_names")
        values = encoded.get("feature_vector", encoded.get("values"))
        if isinstance(names, list) and _is_numeric_sequence(values) and len(names) == len(values):
            return {str(name): float(val) for name, val in zip(names, values)}

    # Case 3: direct numeric vector
    if _is_numeric_sequence(encoded):
        values = [float(v) for v in encoded]
        if existing_feature_names and len(existing_feature_names) == len(values):
            return {name: val for name, val in zip(existing_feature_names, values)}
        return {f"f_{i}": val for i, val in enumerate(values)}

    # Case 4: tuple/list wrapper around a numeric vector
    if isinstance(encoded, (list, tuple)):
        for item in encoded:
            if _is_numeric_sequence(item):
                values = [float(v) for v in item]
                if existing_feature_names and len(existing_feature_names) == len(values):
                    return {name: val for name, val in zip(existing_feature_names, values)}
                return {f"f_{i}": val for i, val in enumerate(values)}

    raise ValueError("Could not convert encode_state output into a feature map.")


def _extract_reward_fn():
    """
    Support multiple reward function names.
    Expected return forms:
      - (reward_value, reward_details)
      - {"reward": x, "details": {...}}
      - numeric reward only
    """
    candidate_names = [
        "compute_transition_reward",
        "calculate_transition_reward",
        "get_transition_reward",
        "transition_reward",
        "compute_reward",
    ]

    for name in candidate_names:
        fn = getattr(reward_module, name, None)
        if callable(fn):
            return fn

    return None


def _normalize_reward_output(raw: Any) -> Tuple[float, Dict[str, Any]]:
    if isinstance(raw, tuple) and len(raw) == 2:
        reward_value, reward_details = raw
        return float(reward_value), dict(reward_details or {})

    if isinstance(raw, dict):
        reward_value = float(raw.get("reward", 0.0))
        reward_details = dict(raw.get("details", {}))
        return reward_value, reward_details

    if _is_scalar_number(raw):
        return float(raw), {}

    return 0.0, {"warning": "Unrecognized reward output format"}


# ---------------------------------------------------------
# Main logger
# ---------------------------------------------------------
class TrajectoryLogger:
    def __init__(self):
        self.enabled = bool(ENABLE_TRAJECTORY_LOGGING)
        self.trajectory_dir = TRAJECTORY_DIR
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.filepath = self.trajectory_dir / f"{TRAJECTORY_FILE_PREFIX}_{timestamp}.jsonl"

        self.feature_names: List[str] = []
        self.reward_fn = _extract_reward_fn()

    def _append_jsonl(self, record: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        with self.filepath.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def start_episode(
        self,
        query: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        episode_id = str(uuid.uuid4())

        merged_metadata = {}
        if metadata:
            merged_metadata.update(metadata)
        if extra_metadata:
            merged_metadata.update(extra_metadata)

        record = {
            "event": "episode_start",
            "timestamp": _utc_now_iso(),
            "episode_id": episode_id,
            "query": query,
            "extra_metadata": merged_metadata,
        }
        self._append_jsonl(record)
        return episode_id

    def log_transition(
        self,
        episode_id: str,
        step_index: int,
        query: str,
        action: str,
        prev_state: Dict[str, Any],
        next_state: Dict[str, Any],
        extra_metadata: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        merged_metadata = {}
        if metadata:
            merged_metadata.update(metadata)
        if extra_metadata:
            merged_metadata.update(extra_metadata)

        prev_encoded = encode_state(prev_state)
        prev_feature_map = _coerce_encoded_state_to_feature_map(
            prev_encoded,
            existing_feature_names=self.feature_names or None,
        )

        if not self.feature_names:
            self.feature_names = list(prev_feature_map.keys())

        next_encoded = encode_state(next_state)
        next_feature_map = _coerce_encoded_state_to_feature_map(
            next_encoded,
            existing_feature_names=self.feature_names,
        )

        next_feature_map = {
            name: float(next_feature_map.get(name, 0.0))
            for name in self.feature_names
        }
        prev_feature_map = {
            name: float(prev_feature_map.get(name, 0.0))
            for name in self.feature_names
        }

        reward_value = 0.0
        reward_details: Dict[str, Any] = {}

        if self.reward_fn is not None:
            try:
                raw_reward = self.reward_fn(prev_state, action, next_state)
                reward_value, reward_details = _normalize_reward_output(raw_reward)
            except Exception as e:
                reward_value = 0.0
                reward_details = {"reward_error": str(e)}
        else:
            reward_details = {"reward_warning": "No reward function found"}

        controller_snapshot = {
            "valid_actions": next_state.get("valid_actions", prev_state.get("valid_actions", [])),
            "action_mask": next_state.get("action_mask", prev_state.get("action_mask", [])),
            "rule_action": next_state.get("rule_action", prev_state.get("rule_action", "")),
            "policy_action": next_state.get("policy_action", prev_state.get("policy_action", "")),
            "policy_confidence": float(
                next_state.get("policy_confidence", prev_state.get("policy_confidence", 0.0))
            ),
            "chosen_action": next_state.get("chosen_action", prev_state.get("chosen_action", action)),
            "controller_source": next_state.get(
                "controller_source",
                prev_state.get("controller_source", ""),
            ),
            "fallback_used": bool(
                next_state.get("fallback_used", prev_state.get("fallback_used", False))
            ),
        }

        record = {
            "event": "transition",
            "timestamp": _utc_now_iso(),
            "episode_id": episode_id,
            "step_index": int(step_index),
            "query": query,
            "action": action,
            "reward": float(reward_value),
            "reward_details": reward_details,
            "done": bool(next_state.get("done", False)),
            "stop_reason": next_state.get("stop_reason", ""),
            "verification_outcome": next_state.get("verification_outcome", ""),
            "citations_pass": bool(next_state.get("citations_pass", False)),
            "state_feature_names": self.feature_names,
            "state_features": prev_feature_map,
            "next_state_features": next_feature_map,
            "controller": controller_snapshot,
            "valid_actions": controller_snapshot["valid_actions"],
            "action_mask": controller_snapshot["action_mask"],
            "rule_action": controller_snapshot["rule_action"],
            "policy_action": controller_snapshot["policy_action"],
            "policy_confidence": controller_snapshot["policy_confidence"],
            "chosen_action": controller_snapshot["chosen_action"],
            "controller_source": controller_snapshot["controller_source"],
            "fallback_used": controller_snapshot["fallback_used"],
            "extra_metadata": merged_metadata,
        }

        self._append_jsonl(record)

    def end_episode(
        self,
        episode_id: str,
        final_state: Dict[str, Any],
        extra_metadata: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        merged_metadata = {}
        if metadata:
            merged_metadata.update(metadata)
        if extra_metadata:
            merged_metadata.update(extra_metadata)

        record = {
            "event": "episode_end",
            "timestamp": _utc_now_iso(),
            "episode_id": episode_id,
            "done": bool(final_state.get("done", False)),
            "stop_reason": final_state.get("stop_reason", ""),
            "verification_outcome": final_state.get("verification_outcome", ""),
            "citations_pass": bool(final_state.get("citations_pass", False)),
            "confidence": float(final_state.get("confidence", 0.0)),
            "step_count": int(final_state.get("step_count", 0)),
            "latency_so_far": float(final_state.get("latency_so_far", 0.0)),
            "action_history": list(final_state.get("action_history", [])),
            "extra_metadata": merged_metadata,
        }
        self._append_jsonl(record)