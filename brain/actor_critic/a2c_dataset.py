"""
a2c_dataset.py
--------------
Offline Actor-Critic Dataset processing.

Parses trajectory logs, groups by episode, calculates Monte Carlo discounted returns
without critic bootstrapping, and splits train/validation safely by episode_id.
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader

import sys
CURRENT_DIR = Path(__file__).resolve().parent
BRAIN_DIR = CURRENT_DIR.parent
if str(BRAIN_DIR) not in sys.path:
    sys.path.append(str(BRAIN_DIR))

# Import action mappings from existing rl_arch if available
try:
    from rl_arch.action_space import ACTION_TO_ID, ID_TO_ACTION
except ImportError:
    print("Warning: Could not import ACTION_TO_ID from rl_arch. Using fallback mapping.")
    ACTION_TO_ID = {
        "retrieve": 0,
        "rewrite_query": 1,
        "answer": 2,
        "verify": 3,
        "stop": 4,
        "grade_docs": 5
    }
    ID_TO_ACTION = {v: k for k, v in ACTION_TO_ID.items()}

class OfflineA2CDataset(Dataset):
    def __init__(self, transitions: List[Dict]):
        """
        transitions is a list of dictionaries with keys:
            'state_features': List[float]
            'action_id': int
            'return': float
        """
        self.transitions = transitions

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.transitions[idx]
        state_features = torch.tensor(t["state_features"], dtype=torch.float32)
        action_id = torch.tensor(t["action_id"], dtype=torch.long)
        ret = torch.tensor(t["return"], dtype=torch.float32)
        
        return state_features, action_id, ret

def process_trajectories(
    trajectories_dir: Path,
    gamma: float = 0.99,
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[OfflineA2CDataset, OfflineA2CDataset]:
    """
    Parses all .jsonl files in trajectories_dir.
    Groups transitions by episode_id.
    Calculates Monte Carlo discounted returns (no critic bootstrapping).
    Splits into train and validation datasets STRICTLY by episode_id.
    """
    if not trajectories_dir.exists():
        raise FileNotFoundError(f"Trajectories directory not found: {trajectories_dir}")

    # 1. Group all transitions by episode
    # episodes: Dict[episode_id, List[transition_dict]]
    episodes: Dict[str, List[Dict]] = {}

    for file_path in trajectories_dir.glob("*.jsonl"):
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if data.get("event") == "transition":
                        ep_id = data.get("episode_id")
                        if not ep_id:
                            continue
                        
                        if ep_id not in episodes:
                            episodes[ep_id] = []
                        
                        # Extract state feature values. We assume the features are ordered consistently
                        # or we can extract them based on state_feature_names if present.
                        feature_names = data.get("state_feature_names", [])
                        state_dict = data.get("state_features", {})
                        
                        if feature_names and state_dict:
                            state_vec = [float(state_dict[k]) for k in feature_names]
                        else:
                            # Fallback if no names list provided
                            state_vec = [float(v) for v in state_dict.values()]
                        
                        action_str = data.get("action", "")
                        action_id = ACTION_TO_ID.get(action_str, -1)
                        reward = float(data.get("reward", 0.0))
                        
                        if action_id != -1 and len(state_vec) > 0:
                            episodes[ep_id].append({
                                "state_features": state_vec,
                                "action_id": action_id,
                                "reward": reward
                            })
                except json.JSONDecodeError:
                    continue

    # 2. Calculate discounted returns per episode
    processed_episodes: Dict[str, List[Dict]] = {}
    all_returns = []
    
    for ep_id, transitions in episodes.items():
        # Monte Carlo discounted return without bootstrapping: G_t = R_t + gamma * G_{t+1}
        g_t = 0.0
        # Iterate backwards
        ep_processed = []
        for t in reversed(transitions):
            r_t = t["reward"]
            g_t = r_t + gamma * g_t
            
            ep_processed.insert(0, {
                "state_features": t["state_features"],
                "action_id": t["action_id"],
                "return": g_t
            })
            all_returns.append(g_t)
        processed_episodes[ep_id] = ep_processed

    # 2.5 Normalize returns globally
    if all_returns:
        import numpy as np
        returns_mean = np.mean(all_returns)
        returns_std = np.std(all_returns) + 1e-8
        for ep_id in processed_episodes:
            for t in processed_episodes[ep_id]:
                t["return"] = float((t["return"] - returns_mean) / returns_std)

    # 3. Train / Validation split safely by episode_id
    episode_ids = list(processed_episodes.keys())
    random.seed(seed)
    random.shuffle(episode_ids)

    num_val = max(1, int(len(episode_ids) * val_split))
    val_ids = set(episode_ids[:num_val])
    train_ids = set(episode_ids[num_val:])

    train_transitions = []
    val_transitions = []

    for ep_id, transitions in processed_episodes.items():
        if ep_id in val_ids:
            val_transitions.extend(transitions)
        else:
            train_transitions.extend(transitions)

    return OfflineA2CDataset(train_transitions), OfflineA2CDataset(val_transitions)
