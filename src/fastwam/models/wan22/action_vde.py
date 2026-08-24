"""Training-free velocity decomposition estimation for Action DiT."""

from typing import NamedTuple, Optional

import torch


class VelocityAnchor(NamedTuple):
    timestep: float
    alpha: torch.Tensor
    beta: torch.Tensor
    direction: torch.Tensor


def build_action_vde_schedule(
    num_steps: int,
    *,
    warmup_steps: int,
    anchor_interval: int,
) -> list[str]:
    """Build a deterministic schedule of full anchors and VDE estimates."""
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if not 2 <= warmup_steps <= num_steps:
        raise ValueError("warmup_steps must be in [2, num_steps]")
    if anchor_interval < 2:
        raise ValueError("anchor_interval must be at least 2")

    routes = ["full"] * num_steps
    last_warmup = warmup_steps - 1
    for step in range(warmup_steps, num_steps - 1):
        if (step - last_warmup) % anchor_interval != 0:
            routes[step] = "estimate"
    return routes


class ActionVelocityDecompositionEstimator:
    """Estimate an Action DiT velocity from the two latest full anchors."""

    def __init__(self, *, eps: float = 1e-6):
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)
        self.anchors: list[VelocityAnchor] = []

    @property
    def ready(self) -> bool:
        return (
            len(self.anchors) == 2
            and abs(self.anchors[-1].timestep - self.anchors[-2].timestep)
            > self.eps
        )

    def add_anchor(
        self,
        sample: torch.Tensor,
        velocity: torch.Tensor,
        timestep: torch.Tensor | float,
    ) -> None:
        if sample.shape != velocity.shape:
            raise ValueError("sample and velocity must have equal shapes")
        if sample.ndim < 2:
            raise ValueError("sample must include batch and feature dimensions")

        x = sample.detach().float()
        v = velocity.detach().float()
        reduce_dims = tuple(range(1, x.ndim))
        x_norm_sq = x.square().sum(dim=reduce_dims, keepdim=True)
        alpha = (v * x).sum(dim=reduce_dims, keepdim=True) / x_norm_sq.clamp_min(
            self.eps
        )
        residual = v - alpha * x
        residual_norm = residual.square().sum(
            dim=reduce_dims, keepdim=True
        ).sqrt()
        beta = residual_norm / x_norm_sq.sqrt().clamp_min(self.eps)
        direction = residual / residual_norm.clamp_min(self.eps)
        anchor = VelocityAnchor(
            float(torch.as_tensor(timestep).detach().cpu().item()),
            alpha,
            beta,
            direction,
        )
        self.anchors = (self.anchors + [anchor])[-2:]

    def estimate(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float,
    ) -> Optional[torch.Tensor]:
        if not self.ready:
            return None
        previous, latest = self.anchors
        target_t = float(torch.as_tensor(timestep).detach().cpu().item())
        scale = (target_t - latest.timestep) / (
            latest.timestep - previous.timestep
        )
        alpha = latest.alpha + (latest.alpha - previous.alpha) * scale
        beta = latest.beta + (latest.beta - previous.beta) * scale
        x = sample.detach().float()
        reduce_dims = tuple(range(1, x.ndim))
        x_norm = x.square().sum(dim=reduce_dims, keepdim=True).sqrt()
        estimate = alpha * x + beta * x_norm * latest.direction
        return estimate.to(device=sample.device, dtype=sample.dtype)
