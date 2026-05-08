import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset


ACTION_TO_ID = {
    "retrieve": 0,
    "rewrite_query": 1,
    "answer": 2,
    "verify": 3,
    "stop": 4,
}

ID_TO_ACTION = {v: k for k, v in ACTION_TO_ID.items()}


def list_trajectory_files(trajectory_dir: Path) -> List[Path]:
    if not trajectory_dir.exists():
        return []
    return sorted(trajectory_dir.glob("*.jsonl"))


def load_transition_records(trajectory_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    files = list_trajectory_files(trajectory_dir)
    print(f"Trajectory dir: {trajectory_dir}")
    print(f"Found jsonl files: {len(files)}")
    for fp in files:
        print(f" - {fp.name}")

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)

                event_type = obj.get("event", obj.get("record_type"))
                action = obj.get("action")

                print(
                    f"[DEBUG] {file_path.name}:{line_no} "
                    f"event={event_type} action={action}"
                )

                if event_type == "transition" and action in ACTION_TO_ID:
                    records.append(obj)

    print(f"Loaded usable transitions: {len(records)}")
    return records


class TrajectoryTransitionDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

        if not self.records:
            self.features = torch.zeros((0, 32), dtype=torch.float32)
            self.labels = torch.zeros((0,), dtype=torch.long)
            return

        feature_rows: List[List[float]] = []
        labels: List[int] = []

        for record in self.records:
            state_features = record.get("state_features", {})
            # Preserve order from the stored names if present
            feature_names = record.get("state_feature_names", [])
            if feature_names:
                row = [float(state_features.get(name, 0.0)) for name in feature_names]
            else:
                # fallback: deterministic sorted order
                row = [float(v) for _, v in sorted(state_features.items())]

            feature_rows.append(row)
            labels.append(ACTION_TO_ID[record["action"]])

        self.features = torch.tensor(feature_rows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]