from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn
from torch.distributions import Normal

from gear_sonic_mjx.config import PPOConfig


@dataclass
class RolloutBatch:
    encoder_obs: torch.Tensor
    proprio_obs: torch.Tensor
    critic_obs: torch.Tensor
    actions: torch.Tensor
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class RolloutStorage:
    """GPU-resident PPO storage for SONIC-style asymmetric actor/critic training."""

    def __init__(self, nsteps: int, nenv: int, encoder_dim: int, proprio_dim: int, critic_dim: int, action_dim: int, device: torch.device):
        shape = (nsteps, nenv)
        self.nsteps, self.nenv, self.device = nsteps, nenv, device
        self.encoder_obs = torch.zeros(*shape, encoder_dim, device=device)
        self.proprio_obs = torch.zeros(*shape, proprio_dim, device=device)
        self.critic_obs = torch.zeros(*shape, critic_dim, device=device)
        self.actions = torch.zeros(*shape, action_dim, device=device)
        self.log_prob = torch.zeros(*shape, device=device)
        self.values = torch.zeros(*shape, device=device)
        self.rewards = torch.zeros(*shape, device=device)
        self.dones = torch.zeros(*shape, dtype=torch.bool, device=device)
        self.returns = torch.zeros(*shape, device=device)
        self.advantages = torch.zeros(*shape, device=device)
        self.ptr = 0

    def add(self, encoder_obs, proprio_obs, critic_obs, actions, log_prob, values, rewards, dones) -> None:
        i = self.ptr
        if i >= self.nsteps:
            raise RuntimeError("rollout storage full")
        self.encoder_obs[i].copy_(encoder_obs)
        self.proprio_obs[i].copy_(proprio_obs)
        self.critic_obs[i].copy_(critic_obs)
        self.actions[i].copy_(actions)
        self.log_prob[i].copy_(log_prob)
        self.values[i].copy_(values)
        self.rewards[i].copy_(rewards)
        self.dones[i].copy_(dones)
        self.ptr += 1

    @torch.no_grad()
    def compute_returns(self, last_value: torch.Tensor, gamma: float, lam: float) -> None:
        gae = torch.zeros_like(last_value)
        for t in reversed(range(self.nsteps)):
            not_done = (~self.dones[t]).float()
            next_value = last_value if t == self.nsteps - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * not_done - self.values[t]
            gae = delta + gamma * lam * not_done * gae
            self.advantages[t] = gae
        self.returns.copy_(self.advantages + self.values)
        a = self.advantages
        self.advantages.copy_((a - a.mean()) / (a.std(unbiased=False) + 1e-8))

    def minibatches(self, num_minibatches: int, epochs: int) -> Iterator[RolloutBatch]:
        total = self.nsteps * self.nenv
        batch_size = total // num_minibatches
        flat = lambda x: x.reshape(total, *x.shape[2:]) if x.ndim > 2 else x.reshape(total)
        tensors = [flat(x) for x in (
            self.encoder_obs, self.proprio_obs, self.critic_obs, self.actions,
            self.log_prob, self.values, self.returns, self.advantages,
        )]
        for _ in range(epochs):
            perm = torch.randperm(total, device=self.device)
            for start in range(0, total, batch_size):
                idx = perm[start:start + batch_size]
                if idx.numel() == 0:
                    continue
                vals = [x[idx] for x in tensors]
                yield RolloutBatch(*vals)

    def clear(self) -> None:
        self.ptr = 0


class SonicActorCritic(nn.Module):
    """SONIC universal-token actor plus asymmetric critic.

    The critic observation is intentionally supplied by the environment so your MuJoCo port can
    add privileged reference/body/contact state without changing the policy interface.
    """

    def __init__(self, token_module: nn.Module, critic: nn.Module, action_dim: int = 29, init_std: float = 0.05, std_min: float = 0.001, std_max: float = 0.5):
        super().__init__()
        self.token_module = token_module
        self.critic = critic
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_std)).log())
        self.std_min, self.std_max = float(std_min), float(std_max)

    def distribution(self, encoder_obs: torch.Tensor, proprio_obs: torch.Tensor):
        out = self.token_module(encoder_obs, proprio_obs, compute_aux_loss=True)
        std = self.log_std.exp().clamp(self.std_min, self.std_max).expand_as(out.action_mean)
        return Normal(out.action_mean, std), out

    def act(self, encoder_obs: torch.Tensor, proprio_obs: torch.Tensor, deterministic: bool = False):
        dist, out = self.distribution(encoder_obs, proprio_obs)
        action = out.action_mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, out

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs).squeeze(-1)


class PPOTrainer:
    """PPO + SONIC auxiliary kinematic reconstruction and KL-adaptive actor LR."""

    def __init__(self, model: SonicActorCritic, cfg: PPOConfig):
        self.model, self.cfg = model, cfg
        actor_params = list(model.token_module.parameters()) + [model.log_std]
        self.actor_opt = torch.optim.Adam(actor_params, lr=cfg.actor_learning_rate)
        self.critic_opt = torch.optim.Adam(model.critic.parameters(), lr=cfg.critic_learning_rate)

    @torch.no_grad()
    def _adapt_actor_lr(self, approx_kl: float) -> float:
        if self.cfg.schedule != "adaptive":
            return self.actor_opt.param_groups[0]["lr"]
        lr = self.actor_opt.param_groups[0]["lr"]
        if approx_kl > self.cfg.desired_kl * 2.0:
            lr /= 1.5
        elif 0.0 < approx_kl < self.cfg.desired_kl * 0.5:
            lr *= 1.5
        lr = min(max(lr, self.cfg.adaptive_lr_min), self.cfg.adaptive_lr_max)
        for pg in self.actor_opt.param_groups:
            pg["lr"] = lr
        return lr

    def update(self, storage: RolloutStorage) -> dict[str, float]:
        stats = {k: 0.0 for k in ["policy_loss", "value_loss", "entropy", "aux_loss", "kl"]}
        n = 0
        for mb in storage.minibatches(self.cfg.num_mini_batches, self.cfg.num_learning_epochs):
            dist, out = self.model.distribution(mb.encoder_obs, mb.proprio_obs)
            log_prob = dist.log_prob(mb.actions).sum(-1)
            ratio = torch.exp(log_prob - mb.old_log_prob)
            surrogate1 = -mb.advantages * ratio
            surrogate2 = -mb.advantages * torch.clamp(ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param)
            policy_loss = torch.maximum(surrogate1, surrogate2).mean()
            entropy = dist.entropy().sum(-1).mean()
            aux = self.model.token_module.reconstruction_loss(out, mb.encoder_obs)
            actor_loss = policy_loss - self.cfg.entropy_coef * entropy + self.cfg.aux_reconstruction_coef * aux

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(list(self.model.token_module.parameters()) + [self.model.log_std], self.cfg.max_grad_norm)
            self.actor_opt.step()

            value = self.model.value(mb.critic_obs)
            value_clipped = mb.old_value + (value - mb.old_value).clamp(-self.cfg.clip_param, self.cfg.clip_param)
            v1 = (value - mb.returns).square()
            v2 = (value_clipped - mb.returns).square()
            value_loss = 0.5 * torch.maximum(v1, v2).mean()
            self.critic_opt.zero_grad(set_to_none=True)
            (self.cfg.value_loss_coef * value_loss).backward()
            nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.cfg.max_grad_norm)
            self.critic_opt.step()

            with torch.no_grad():
                # Standard first-order approximation used for PPO diagnostics.
                log_ratio = log_prob - mb.old_log_prob
                kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean().clamp_min(0.0)
            stats["policy_loss"] += float(policy_loss.detach())
            stats["value_loss"] += float(value_loss.detach())
            stats["entropy"] += float(entropy.detach())
            stats["aux_loss"] += float(aux.detach())
            stats["kl"] += float(kl)
            n += 1

        if n:
            stats = {k: v / n for k, v in stats.items()}
        stats["actor_lr"] = self._adapt_actor_lr(stats["kl"])
        return stats
