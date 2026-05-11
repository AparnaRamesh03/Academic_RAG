"""
brain/context_marl_ac/marl/trainer.py
-------------------------------------
A2C trainer for Supervisor-Guided Fully Free MARL.

Updates:
- Supervisor actor
- Selected agent actors
- Centralized critic
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from context_marl_ac.config import (
    LEARNING_RATE,
    GAMMA,
    ENTROPY_COEF,
    VALUE_COEF,
    GRAD_CLIP_NORM,
)
from context_marl_ac.schemas.trajectory import Episode
from context_marl_ac.schemas.actions import AGENT_NAMES
from context_marl_ac.marl.supervisor_actor import SupervisorActor


class MARLTrainer:
    def __init__(
        self,
        actors: nn.ModuleDict,
        critic: nn.Module,
        supervisor: Optional[SupervisorActor] = None,
        lr: float = LEARNING_RATE,
    ):
        self.actors = actors
        self.critic = critic
        self.supervisor = supervisor

        params = list(self.actors.parameters()) + list(self.critic.parameters())
        if self.supervisor is not None:
            params += list(self.supervisor.parameters())

        self.params = params
        self.optimizer = optim.AdamW(self.params, lr=lr)

    def train_on_episodes(self, episodes: List[Episode]) -> Dict[str, float]:
        if not episodes:
            return {}

        total_supervisor_loss = torch.tensor(0.0)
        total_agent_actor_loss = torch.tensor(0.0)
        total_critic_loss = torch.tensor(0.0)
        total_entropy_loss = torch.tensor(0.0)
        total_raw_entropy = torch.tensor(0.0)

        num_steps = 0

        for ep in episodes:
            ep.compute_advantages(gamma=GAMMA)

            for step_data in ep.steps:
                if step_data.action_id < 0:
                    continue

                global_feats = torch.tensor(step_data.global_features, dtype=torch.float32)
                obs = torch.tensor(step_data.observation, dtype=torch.float32)
                action_mask = torch.tensor(step_data.action_mask, dtype=torch.float32)
                agent_mask = torch.tensor(step_data.agent_mask, dtype=torch.float32)

                advantage = torch.tensor(step_data.advantage, dtype=torch.float32)
                target_value = advantage + torch.tensor(step_data.critic_value, dtype=torch.float32)

                # -------------------------------
                # Supervisor loss
                # -------------------------------
                supervisor_loss = torch.tensor(0.0)
                supervisor_entropy = torch.tensor(0.0)

                if self.supervisor is not None and step_data.supervisor_agent_id >= 0:
                    supervisor_logits = self.supervisor(
                        global_feats.unsqueeze(0),
                        agent_mask.unsqueeze(0),
                    )
                    supervisor_log_probs = torch.log_softmax(supervisor_logits, dim=-1)
                    supervisor_probs = torch.softmax(supervisor_logits, dim=-1)

                    selected_agent_log_prob = supervisor_log_probs[
                        0,
                        step_data.supervisor_agent_id,
                    ]

                    supervisor_loss = -selected_agent_log_prob * advantage
                    supervisor_entropy = -(
                        supervisor_probs * supervisor_log_probs
                    ).sum(dim=-1).mean()

                # -------------------------------
                # Selected agent actor loss
                # -------------------------------
                agent_name = step_data.agent
                actor = self.actors[agent_name]

                logits = actor(obs.unsqueeze(0), action_mask.unsqueeze(0))
                log_probs = torch.log_softmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)

                selected_action_log_prob = log_probs[0, step_data.action_id]
                agent_actor_loss = -selected_action_log_prob * advantage
                agent_entropy = -(probs * log_probs).sum(dim=-1).mean()

                # -------------------------------
                # Critic loss
                # -------------------------------
                current_v = self.critic(global_feats.unsqueeze(0))
                critic_loss = F.mse_loss(current_v[0, 0], target_value)

                entropy = supervisor_entropy + agent_entropy
                entropy_loss = -ENTROPY_COEF * entropy

                total_supervisor_loss = total_supervisor_loss + supervisor_loss
                total_agent_actor_loss = total_agent_actor_loss + agent_actor_loss
                total_critic_loss = total_critic_loss + critic_loss
                total_entropy_loss = total_entropy_loss + entropy_loss
                total_raw_entropy = total_raw_entropy + entropy.detach()

                num_steps += 1

        if num_steps == 0:
            return {}

        total_loss = (
            total_supervisor_loss
            + total_agent_actor_loss
            + VALUE_COEF * total_critic_loss
            + total_entropy_loss
        ) / num_steps

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.params, GRAD_CLIP_NORM)
        self.optimizer.step()

        total_actor_loss = total_supervisor_loss + total_agent_actor_loss

        return {
            "loss": total_loss.item(),
            "actor_loss": total_actor_loss.item() / num_steps,
            "supervisor_loss": total_supervisor_loss.item() / num_steps,
            "agent_actor_loss": total_agent_actor_loss.item() / num_steps,
            "critic_loss": total_critic_loss.item() / num_steps,
            "entropy": total_raw_entropy.item() / num_steps,
            "entropy_loss": total_entropy_loss.item() / num_steps,
        }