"""
train_a2c.py
------------
End-to-end Offline A2C training loop.

- Computes advantages: A_t = G_t - V(s)
- Normalizes advantages per batch
- Calculates Actor, Critic, BC, and Entropy losses
- Saves model weights along with action mappings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from a2c_model import ActorCriticNet
from a2c_dataset import process_trajectories, ACTION_TO_ID, ID_TO_ACTION

def train_offline_a2c(
    trajectories_dir: Path,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    gamma: float = 0.99,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    bc_coef: float = 1.0,
    save_dir: Path = Path("checkpoints"),
):
    print(f"Loading and processing trajectories from {trajectories_dir}...")
    train_dataset, val_dataset = process_trajectories(trajectories_dir, gamma=gamma)
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("No training data found. Exiting.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ActorCriticNet(input_dim=32, hidden_dim=128, output_dim=len(ACTION_TO_ID))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_bc_loss = 0.0

        for state_features, action_id, returns in train_loader:
            # Returns is shape (Batch)
            returns = returns.unsqueeze(-1)  # Shape (Batch, 1)

            logits, values = model(state_features)
            
            # Critic Loss (Huber Loss / Smooth L1)
            critic_loss = F.smooth_l1_loss(values, returns)

            # Advantages
            advantages = returns - values.detach()
            
            # Normalize advantages per batch
            if advantages.size(0) > 1:
                adv_mean = advantages.mean()
                adv_std = advantages.std() + 1e-8
                advantages = (advantages - adv_mean) / adv_std

            # Actor Loss
            # Compute log probabilities of the taken actions
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Gather log prob of the actions actually taken
            action_log_probs = log_probs.gather(1, action_id.unsqueeze(1))
            
            # Actor loss: -log(pi(a|s)) * A
            actor_loss = -(action_log_probs * advantages).mean()

            # Entropy Loss (to encourage exploration)
            entropy_loss = -(probs * log_probs).sum(dim=-1).mean()

            # Behavior Cloning Loss (Cross Entropy)
            bc_loss = F.cross_entropy(logits, action_id)

            # Composite Loss
            loss = (actor_loss 
                    + value_coef * critic_loss 
                    - entropy_coef * entropy_loss 
                    + bc_coef * bc_loss)

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_bc_loss += bc_loss.item()

        num_batches = len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | "
              f"Loss: {total_loss/num_batches:.4f} | "
              f"Actor: {total_actor_loss/num_batches:.4f} | "
              f"Critic: {total_critic_loss/num_batches:.4f} | "
              f"BC: {total_bc_loss/num_batches:.4f}")

        scheduler.step()

    # Save Checkpoint
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "a2c_policy.pt"
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "action_to_id": ACTION_TO_ID,
        "id_to_action": ID_TO_ACTION,
        "input_dim": 32,
        "hidden_dim": 128,
        "output_dim": len(ACTION_TO_ID)
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    # Point to the existing trajectories
    import sys
    CURRENT_DIR = Path(__file__).resolve().parent
    BRAIN_DIR = CURRENT_DIR.parent
    traj_dir = BRAIN_DIR / "rl_arch" / "data" / "trajectories"
    
    train_offline_a2c(
        trajectories_dir=traj_dir,
        epochs=30,
        batch_size=64,
        lr=1e-3
    )
