import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import (
    ACTION_TO_ID,
    ID_TO_ACTION,
    load_transition_records,
    TrajectoryTransitionDataset,
)
from policy_model import ControllerPolicyNet


ROOT = Path(__file__).resolve().parent
TRAJECTORY_DIR = ROOT / "data" / "trajectories"
CHECKPOINT_DIR = ROOT / "data" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = CHECKPOINT_DIR / "phase4_policy.pt"
META_PATH = CHECKPOINT_DIR / "phase4_policy_meta.json"

# BATCH_SIZE = 32
# EPOCHS = 20
# LR = 1e-3
# VAL_SPLIT = 0.2
# SEED = 42

BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-3
VAL_SPLIT = 0.2
SEED = 42

MIN_TRANSITIONS_TO_TRAIN = 8
MIN_TRANSITIONS_FOR_VAL = 20


# def split_indices(n: int, val_split: float) -> Tuple[List[int], List[int]]:
#     indices = list(range(n))
#     random.Random(SEED).shuffle(indices)

#     val_size = max(1, int(n * val_split)) if n > 1 else 0
#     val_indices = indices[:val_size]
#     train_indices = indices[val_size:] if n > 1 else indices

#     if not train_indices:
#         train_indices = val_indices
#         val_indices = []

#     return train_indices, val_indices

def split_indices(n: int, val_split: float) -> Tuple[List[int], List[int]]:
    indices = list(range(n))
    random.Random(SEED).shuffle(indices)

    if n < MIN_TRANSITIONS_FOR_VAL:
        return indices, []

    val_size = max(1, int(n * val_split))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    if not train_indices:
        train_indices = val_indices
        val_indices = []

    return train_indices, val_indices


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()

    if total == 0:
        return 0.0, 0.0

    return total_loss / total, correct / total


def main():
    records = load_transition_records(TRAJECTORY_DIR)
    print(f"Loaded transitions: {len(records)}")

    # if len(records) < 10:
    #     print("Not enough transitions yet.")
    #     print("Collect more supervisor episodes first. Aim for at least 50-100 transitions.")
    #     return

    if len(records) < MIN_TRANSITIONS_TO_TRAIN:
        print("Not enough transitions yet for even a tiny debug run.")
        print(f"Need at least {MIN_TRANSITIONS_TO_TRAIN} transitions.")
        return

    dataset = TrajectoryTransitionDataset(records)
    print(f"Dataset size: {len(dataset)}")
    print(f"Feature dim: {dataset.features.shape[1]}")
    print(f"Action classes: {len(ACTION_TO_ID)}")

    print("Small-data mode:" if len(dataset) < MIN_TRANSITIONS_FOR_VAL else "Standard mode:")
    print(f" - transitions: {len(dataset)}")
    print(f" - validation enabled: {len(dataset) >= MIN_TRANSITIONS_FOR_VAL}")

    # train_indices, val_indices = split_indices(len(dataset), VAL_SPLIT)
    # train_ds = Subset(dataset, train_indices)
    # val_ds = Subset(dataset, val_indices) if val_indices else None

    # train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False) if val_ds else None

    train_indices, val_indices = split_indices(len(dataset), VAL_SPLIT)
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices) if val_indices else None

    effective_batch_size = min(BATCH_SIZE, max(1, len(train_ds)))

    train_loader = DataLoader(train_ds, batch_size=effective_batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_ds, batch_size=min(BATCH_SIZE, max(1, len(val_ds))), shuffle=False)
        if val_ds and len(val_ds) > 0
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ControllerPolicyNet(
        input_dim=dataset.features.shape[1],
        hidden_dim=128,
        output_dim=len(ACTION_TO_ID),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        total = 0
        correct = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            total += x.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()

        train_loss = running_loss / total if total else 0.0
        train_acc = correct / total if total else 0.0

        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = 0.0, train_acc

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "input_dim": dataset.features.shape[1],
                "num_actions": len(ACTION_TO_ID),
            }

    if best_state is None:
        best_state = {
            "model_state_dict": model.state_dict(),
            "input_dim": dataset.features.shape[1],
            "num_actions": len(ACTION_TO_ID),
        }

    torch.save(best_state, MODEL_PATH)

    meta = {
        "model_path": str(MODEL_PATH),
        "num_transitions": len(dataset),
        "input_dim": dataset.features.shape[1],
        "actions": list(ACTION_TO_ID.keys()),
        "action_to_id": ACTION_TO_ID,
        "id_to_action": ID_TO_ACTION,
        "best_val_acc": best_val_acc,
    }

    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metadata to: {META_PATH}")


if __name__ == "__main__":
    main()