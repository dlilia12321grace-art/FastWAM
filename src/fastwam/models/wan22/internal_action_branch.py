import copy

import torch
import torch.nn as nn


def build_action_gap_schedule(num_steps: int, action_gap: int) -> list[str]:
    """Build the LingBot-style periodic correction schedule.

    Step 0, the last step, and every step satisfying
    ``(step + 1) % action_gap == 0`` use the full Action DiT. All remaining
    steps use the internal branch.
    """
    if num_steps <= 0:
        raise ValueError(f"`num_steps` must be positive, got {num_steps}.")
    if action_gap <= 0:
        raise ValueError(f"`action_gap` must be positive, got {action_gap}.")
    last_step = num_steps - 1
    return [
        "full"
        if step == 0 or step == last_step or (step + 1) % action_gap == 0
        else "internal"
        for step in range(num_steps)
    ]


class LoRALinear(nn.Module):
    """Frozen Linear plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        """Fold the low-rank update into a standalone Linear for inference."""
        merged = copy.deepcopy(self.base)
        delta = torch.matmul(
            self.lora_B.weight.float(), self.lora_A.weight.float()
        ).mul_(self.scaling)
        merged.weight.add_(delta.to(device=merged.weight.device, dtype=merged.weight.dtype))
        for parameter in merged.parameters():
            parameter.requires_grad = False
        return merged


def _inject_lora(module: nn.Module, rank: int, alpha: float) -> None:
    """Add LoRA to attention projections and the two FFN projections."""
    target_paths = (
        "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
        "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
        "ffn.0", "ffn.2",
    )
    for path in target_paths:
        parent = module
        parts = path.split(".")
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        leaf = parts[-1]
        base = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Expected Linear at {path}, got {type(base)}.")
        wrapped = LoRALinear(base, rank=rank, alpha=alpha)
        if leaf.isdigit():
            parent[int(leaf)] = wrapped
        else:
            setattr(parent, leaf, wrapped)


class DeepCopyInternalActionBranch(nn.Module):
    """Deep-copied Action DiT suffix with LoRA adapters and a copied output head."""

    def __init__(
        self,
        action_expert: nn.Module,
        fork_layer: int = 18,
        source_start_layer: int = 19,
        source_end_layer: int = 24,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        train_output_head: bool = True,
    ):
        super().__init__()
        if not 1 <= source_start_layer <= source_end_layer <= len(action_expert.blocks):
            raise ValueError("Invalid 1-based source layer range.")
        if not 1 <= fork_layer < len(action_expert.blocks):
            raise ValueError("`fork_layer` must be a valid non-final 1-based layer.")
        self.fork_layer = int(fork_layer)
        self.source_start_layer = int(source_start_layer)
        self.source_end_layer = int(source_end_layer)
        self.blocks = copy.deepcopy(
            action_expert.blocks[source_start_layer - 1:source_end_layer]
        )
        self.head = copy.deepcopy(action_expert.head)

        for parameter in self.parameters():
            parameter.requires_grad = False
        for block in self.blocks:
            _inject_lora(block, rank=lora_rank, alpha=lora_alpha)
        for parameter in self.head.parameters():
            parameter.requires_grad = bool(train_output_head)

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def merge_lora_for_inference(self) -> None:
        """Replace every LoRA wrapper with its merged base Linear in-place."""
        replacements = []
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                replacements.append((name, module.merged_linear()))
        for path, merged in replacements:
            parent = self
            parts = path.split(".")
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            leaf = parts[-1]
            if leaf.isdigit():
                parent[int(leaf)] = merged
            else:
                setattr(parent, leaf, merged)
        self.eval()

    def extra_repr(self) -> str:
        return (
            f"fork_layer={self.fork_layer}, "
            f"source_layers={self.source_start_layer}..{self.source_end_layer}"
        )
