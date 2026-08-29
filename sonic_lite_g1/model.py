from __future__ import annotations

import torch
from torch import nn
from tensordict import TensorDict
from rsl_rl.models.mlp_model import MLPModel

from .quantizer import ScalarQuantizer


class SonicLiteActor(MLPModel):
    """RSL-RL actor with a SONIC-style quantized motor-token bottleneck.

    Raw future motion references never reach the controller MLP. They first go
    through a small encoder and a scalar quantizer. The controller therefore
    has to act from the 64-D motor token plus short robot-state history.
    """

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        *,
        future_group: str = "future",
        proprio_group: str = "proprio",
        token_dim: int = 64,
        quantization_levels: int = 32,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
    ) -> None:
        self.future_group = future_group
        self.proprio_group = proprio_group
        self.token_dim = int(token_dim)

        if future_group not in obs.keys() or proprio_group not in obs.keys():
            raise KeyError(
                f"Expected observation groups '{future_group}' and '{proprio_group}', "
                f"got {list(obs.keys())}"
            )
        self._future_dim = int(obs[future_group].shape[-1])
        self._proprio_dim = int(obs[proprio_group].shape[-1])

        active = obs_groups[obs_set]
        if active != [future_group, proprio_group]:
            raise ValueError(
                "SonicLiteActor expects actor obs_groups to be exactly "
                f"[{future_group!r}, {proprio_group!r}], got {active!r}"
            )

        # MLPModel calls _get_latent_dim() while constructing its control head,
        # so dimensions above must exist before super().__init__().
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        dims = [self._future_dim, *map(int, encoder_hidden_dims), self.token_dim]
        layers: list[nn.Module] = []
        for din, dout in zip(dims[:-1], dims[1:]):
            layers.extend((nn.Linear(din, dout), nn.SiLU()))
        # The final SiLU is harmless but unnecessary before tanh quantization.
        layers.pop()
        self.motion_encoder = nn.Sequential(*layers)
        self.quantizer = ScalarQuantizer(quantization_levels)

    def _get_latent_dim(self) -> int:
        return self.token_dim + self._proprio_dim

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
    ) -> torch.Tensor:
        del masks, hidden_state
        # Keep RSL-RL's running normalization, but enforce that only the token
        # (not the raw motion reference) is passed to the motor controller.
        combined = torch.cat([obs[self.future_group], obs[self.proprio_group]], dim=-1)
        combined = self.obs_normalizer(combined)
        future = combined[..., : self._future_dim]
        proprio = combined[..., self._future_dim :]

        token_pre_q = self.motion_encoder(future)
        token = self.quantizer(token_pre_q)
        return torch.cat([token, proprio], dim=-1)

    @torch.no_grad()
    def encode_motor_token(self, obs: TensorDict) -> torch.Tensor:
        """Return the deployed 64-D motor token for inspection/debugging."""
        combined = torch.cat([obs[self.future_group], obs[self.proprio_group]], dim=-1)
        combined = self.obs_normalizer(combined)
        future = combined[..., : self._future_dim]
        return self.quantizer(self.motion_encoder(future))

    def as_jit(self) -> nn.Module:
        raise NotImplementedError(
            "The stock RSL-RL MLP exporter only copies the final MLP and would "
            "drop the token encoder. Add a dedicated deployment wrapper after V1 training."
        )

    def as_onnx(self, verbose: bool) -> nn.Module:
        del verbose
        raise NotImplementedError(
            "The stock RSL-RL MLP exporter only copies the final MLP and would "
            "drop the token encoder. Add a dedicated deployment wrapper after V1 training."
        )
