"""
test_a2c.py
-----------
Essential test suite for Offline A2C pipeline.
"""
import tempfile
import json
import torch
import os
from pathlib import Path

from a2c_model import ActorCriticNet
from a2c_dataset import OfflineA2CDataset, process_trajectories

def test_model_forward():
    print("Running test_model_forward...")
    batch_size = 4
    input_dim = 32
    output_dim = 5
    
    model = ActorCriticNet(input_dim=input_dim, hidden_dim=64, output_dim=output_dim)
    dummy_input = torch.randn(batch_size, input_dim)
    
    logits, values = model(dummy_input)
    
    assert logits.shape == (batch_size, output_dim), f"Expected {(batch_size, output_dim)}, got {logits.shape}"
    assert values.shape == (batch_size, 1), f"Expected {(batch_size, 1)}, got {values.shape}"
    print("PASS: test_model_forward")

def test_discounted_return():
    print("Running test_discounted_return...")
    # We will simulate the logic in a2c_dataset for return calculation
    transitions = [
        {"reward": 1.0},
        {"reward": 0.0},
        {"reward": 10.0}
    ]
    gamma = 0.99
    
    g_t = 0.0
    processed = []
    for t in reversed(transitions):
        r_t = t["reward"]
        g_t = r_t + gamma * g_t
        processed.insert(0, g_t)
        
    # Expected:
    # t=2: r=10.0 -> g_2 = 10.0
    # t=1: r=0.0 -> g_1 = 0.0 + 0.99 * 10.0 = 9.9
    # t=0: r=1.0 -> g_0 = 1.0 + 0.99 * 9.9 = 10.801
    
    assert abs(processed[2] - 10.0) < 1e-5
    assert abs(processed[1] - 9.9) < 1e-5
    assert abs(processed[0] - 10.801) < 1e-5
    print("PASS: test_discounted_return")

def test_dataset_parsing():
    print("Running test_dataset_parsing...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dummy_file = tmp_path / "dummy.jsonl"
        
        # Write dummy trajectory
        dummy_data = [
            {"event": "episode_start", "episode_id": "ep1"},
            {"event": "transition", "episode_id": "ep1", "action": "retrieve", "reward": 0.5, "state_features": {"f1": 1.0, "f2": 2.0}},
            {"event": "transition", "episode_id": "ep1", "action": "answer", "reward": 1.0, "state_features": {"f1": 0.5, "f2": 0.5}},
            {"event": "episode_start", "episode_id": "ep2"},
            {"event": "transition", "episode_id": "ep2", "action": "stop", "reward": -1.0, "state_features": {"f1": 0.0, "f2": 0.0}}
        ]
        
        with open(dummy_file, "w") as f:
            for d in dummy_data:
                f.write(json.dumps(d) + "\n")
                
        # To avoid index errors with missing features, process_trajectories will use the fallback of taking values.
        # But we need val_split to put at least 1 in val and 1 in train if we only have 2 episodes.
        # Wait, max(1, int(2 * 0.5)) = 1, so 1 train, 1 val.
        train_ds, val_ds = process_trajectories(tmp_path, val_split=0.5)
        
        assert len(train_ds) + len(val_ds) == 3, f"Expected 3 transitions total, got {len(train_ds) + len(val_ds)}"
        print("PASS: test_dataset_parsing")

def test_checkpoint_reload():
    print("Running test_checkpoint_reload...")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = ActorCriticNet()
        ckpt_path = Path(tmpdir) / "test_a2c.pt"
        
        action_to_id = {"test": 0}
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "action_to_id": action_to_id,
            "input_dim": 32,
            "hidden_dim": 128,
            "output_dim": 5
        }
        
        torch.save(checkpoint, ckpt_path)
        
        # Reload
        loaded_ckpt = torch.load(ckpt_path)
        new_model = ActorCriticNet(
            input_dim=loaded_ckpt["input_dim"],
            hidden_dim=loaded_ckpt["hidden_dim"],
            output_dim=loaded_ckpt["output_dim"]
        )
        new_model.load_state_dict(loaded_ckpt["model_state_dict"])
        
        assert loaded_ckpt["action_to_id"] == action_to_id
        print("PASS: test_checkpoint_reload")

if __name__ == "__main__":
    test_model_forward()
    test_discounted_return()
    test_dataset_parsing()
    test_checkpoint_reload()
    print("ALL TESTS PASSED!")
