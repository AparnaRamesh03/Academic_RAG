"""
brain/context_marl_ac/marl/trainer.py
-------------------------------------
A2C Trainer for the Context-Engineered MARL system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict, Any

from context_marl_ac.config import (
    LEARNING_RATE, GAMMA, ENTROPY_COEF, VALUE_COEF, GRAD_CLIP_NORM
)
from context_marl_ac.schemas.trajectory import Episode

class MARLTrainer:
    """
    Handles weight updates for the 5 actors and the centralized critic.
    """
    def __init__(
        self, 
        actors: nn.ModuleDict, 
        critic: nn.Module, 
        lr: float = LEARNING_RATE
    ):
        self.actors = actors
        self.critic = critic
        
        # Combined parameters for optimization
        self.params = list(self.actors.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.AdamW(self.params, lr=lr)

    def train_on_episodes(self, episodes: List[Episode]) -> Dict[str, float]:
        """
        Calculates A2C loss and performs a gradient update step.
        """
        if not episodes:
            return {}

        total_actor_loss = torch.tensor(0.0, requires_grad=True)
        total_critic_loss = torch.tensor(0.0, requires_grad=True)
        total_entropy_loss = torch.tensor(0.0, requires_grad=True)
        total_raw_entropy = torch.tensor(0.0)
        
        num_steps = 0
        
        for ep in episodes:
            # 1. Compute Monte Carlo returns and advantages based on OLD critic values
            ep.compute_advantages(gamma=GAMMA)
            
            for step_data in ep.steps:
                agent_name = step_data.agent
                obs = torch.tensor(step_data.observation, dtype=torch.float32)
                global_feats = torch.tensor(step_data.global_features, dtype=torch.float32)
                mask = torch.tensor(step_data.action_mask, dtype=torch.float32)
                action_id = step_data.action_id
                
                # Advantage from trajectory (Target - Old V)
                # We can also re-compute advantage using current critic for better stability, 
                # but standard A2C often uses the advantage calculated at rollout time.
                advantage = torch.tensor(step_data.advantage, dtype=torch.float32)
                target_value = advantage + step_data.critic_value # G_t (Return)
                
                # 2. Get policy distribution (Actor)
                actor = self.actors[agent_name]
                logits = actor(obs.unsqueeze(0), mask.unsqueeze(0))
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # 3. Actor Loss (Policy Gradient)
                # L_actor = -log_pi(a|s) * A
                selected_log_prob = log_probs[0, action_id]
                actor_loss = -selected_log_prob * advantage
                
                # 4. Entropy Loss (Exploration)
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                
                # 5. Critic Loss (Value Approximation)
                # L_critic = (G_t - V(s))^2
                current_v = self.critic(global_feats.unsqueeze(0))
                critic_loss = F.mse_loss(current_v[0, 0], target_value)
                
                total_actor_loss = total_actor_loss + actor_loss
                total_critic_loss = total_critic_loss + critic_loss
                total_entropy_loss = total_entropy_loss - ENTROPY_COEF * entropy
                total_raw_entropy = total_raw_entropy + entropy.detach()
                num_steps += 1

        if num_steps == 0:
            return {}
            
        # 6. Combined Loss
        # Total Loss = Actor_Loss + Value_Coef * Critic_Loss + Entropy_Loss
        total_loss = (total_actor_loss + VALUE_COEF * total_critic_loss + total_entropy_loss) / num_steps
        
        # 7. Update
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.params, GRAD_CLIP_NORM)
        self.optimizer.step()
        
        return {
            "loss":           total_loss.item(),
            "actor_loss":     total_actor_loss.item() / num_steps,
            "critic_loss":    total_critic_loss.item() / num_steps,
            "entropy_loss":   total_entropy_loss.item() / num_steps,
            "entropy":        total_raw_entropy.item() / num_steps
        }

