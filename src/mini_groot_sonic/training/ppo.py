from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mini_groot_sonic.config import PPOConfig, SonicTinyConfig
from mini_groot_sonic.models.sonic_tiny import TinySonicCritic, TinySonicPolicy


@dataclass
class Rollout:
    prop: torch.Tensor
    ref: torch.Tensor
    action: torch.Tensor
    old_logp: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    ret: torch.Tensor | None = None
    adv: torch.Tensor | None = None


class PPOAuxTrainer:
    def __init__(
        self,
        policy: TinySonicPolicy,
        critic: TinySonicCritic,
        sonic_cfg: SonicTinyConfig,
        ppo_cfg: PPOConfig,
    ):
        self.policy = policy
        self.critic = critic
        self.sonic_cfg = sonic_cfg
        self.cfg = ppo_cfg
        self.optim = torch.optim.Adam(
            [
                {"params": policy.parameters(), "lr": ppo_cfg.actor_lr},
                {"params": critic.parameters(), "lr": ppo_cfg.critic_lr},
            ]
        )

    @torch.no_grad()
    def compute_gae(self, roll: Rollout, last_value: torch.Tensor) -> None:
        t, n = roll.reward.shape
        adv = torch.zeros_like(roll.reward)
        gae = torch.zeros(n, device=roll.reward.device)
        next_value = last_value
        for i in reversed(range(t)):
            nonterminal = (~roll.done[i]).float()
            delta = roll.reward[i] + self.cfg.gamma * next_value * nonterminal - roll.value[i]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * nonterminal * gae
            adv[i] = gae
            next_value = roll.value[i]
        ret = adv + roll.value
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        roll.adv = adv
        roll.ret = ret

    def update(self, roll: Rollout) -> dict[str, float]:
        assert roll.adv is not None and roll.ret is not None
        t, n = roll.reward.shape
        total = t * n
        flat = lambda x: x.reshape(total, *x.shape[2:])
        prop = flat(roll.prop)
        ref = flat(roll.ref)
        action = flat(roll.action)
        old_logp = flat(roll.old_logp)
        old_value = flat(roll.value)
        adv = flat(roll.adv)
        ret = flat(roll.ret)

        batch_size = max(1, total // self.cfg.minibatches)
        stats = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "recon": 0.0, "kl": 0.0, "updates": 0}
        stop = False
        for _ in range(self.cfg.ppo_epochs):
            if stop:
                break
            perm = torch.randperm(total, device=prop.device)
            for start in range(0, total, batch_size):
                ix = perm[start : start + batch_size]
                out = self.policy(prop[ix], ref[ix])
                dist = self.policy.distribution(out.action_mean)
                logp = dist.log_prob(action[ix]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                ratio = (logp - old_logp[ix]).exp()
                pg1 = ratio * adv[ix]
                pg2 = ratio.clamp(1.0 - self.cfg.clip, 1.0 + self.cfg.clip) * adv[ix]
                policy_loss = -torch.minimum(pg1, pg2).mean()

                value = self.critic(prop[ix], ref[ix])
                # Light PPO-style value clipping.
                value_clipped = old_value[ix] + (value - old_value[ix]).clamp(-self.cfg.clip, self.cfg.clip)
                v1 = (value - ret[ix]).square()
                v2 = (value_clipped - ret[ix]).square()
                value_loss = 0.5 * torch.maximum(v1, v2).mean()

                recon_target = ref[ix].flatten(1)
                recon_loss = torch.nn.functional.mse_loss(out.reconstruction, recon_target)
                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy
                    + self.cfg.aux_recon_coef * recon_loss
                )

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.critic.parameters()), self.cfg.max_grad_norm)
                self.optim.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - (logp - old_logp[ix])).mean().abs()
                stats["policy"] += float(policy_loss.detach())
                stats["value"] += float(value_loss.detach())
                stats["entropy"] += float(entropy.detach())
                stats["recon"] += float(recon_loss.detach())
                stats["kl"] += float(approx_kl.detach())
                stats["updates"] += 1
                if approx_kl > 1.5 * self.cfg.target_kl:
                    stop = True
                    break

        u = max(stats.pop("updates"), 1)
        return {k: v / u for k, v in stats.items()}
