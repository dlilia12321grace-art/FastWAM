import time
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .dynamic_action_gap import (
    DynamicActionGapGate,
    build_compute_matched_action_gap_schedule,
    build_dynamic_action_gap_meta,
    load_dynamic_action_gap_gate,
    select_dynamic_action_gap_route,
)
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .internal_action_branch import (
    DeepCopyInternalActionBranch,
    build_action_gap_schedule,
)
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class InternalActionHead(nn.Module):
    """Lightweight head used to decode an intermediate Action DiT state."""

    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, action_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(hidden))


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.reset_c3cache_analysis()
        self.reset_c3cache_state()
        self.reset_internal_distillation_samples()
        self._internal_action_heads = nn.ModuleDict()
        self._internal_action_heads_checkpoint: Optional[str] = None
        self.internal_lora_branch: Optional[DeepCopyInternalActionBranch] = None
        self._internal_lora_optimizer: Optional[torch.optim.Optimizer] = None
        self._internal_lora_updates = 0
        self._internal_lora_losses: list[float] = []
        self._internal_lora_best_loss: Optional[float] = None
        self._internal_lora_best_update: Optional[int] = None
        self._internal_lora_best_trainable_state: Optional[dict[str, torch.Tensor]] = None
        self._internal_lora_checkpoint: Optional[str] = None
        self.dynamic_action_gap_gate: Optional[DynamicActionGapGate] = None
        self._dynamic_action_gap_checkpoint: Optional[str] = None
        self._dynamic_action_gap_default_threshold: Optional[float] = None
        self.reset_dynamic_action_gap_state()
        self.reset_dynamic_action_gap_samples()

        self.to(self.device)

    def reset_c3cache_analysis(self) -> None:
        """Clear cross-chunk residuals at an episode boundary."""
        self._c3cache_analysis_prev_residuals: dict[int, torch.Tensor] = {}
        self._c3cache_analysis_chunk_index = 0

    def reset_c3cache_state(self) -> None:
        """Clear C3ache residual cache at an episode boundary."""
        self._c3cache_residuals: dict[int, torch.Tensor] = {}
        self._c3cache_chunk_index = 0

    def reset_internal_distillation_samples(self) -> None:
        """Clear the in-memory Internal Head distillation sample buffer."""
        self._internal_distillation_samples: list[dict[str, Any]] = []
        self._internal_distillation_chunk_index = 0

    def reset_dynamic_action_gap_state(self) -> None:
        """Clear episode-level Dynamic ActionGap accounting state."""
        self._dynamic_action_gap_chunk_index = 0

    def reset_dynamic_action_gap_samples(self) -> None:
        """Clear offline training samples for the Dynamic ActionGap gate."""
        self._dynamic_action_gap_samples: list[dict[str, Any]] = []
        self._dynamic_action_gap_collection_chunk_index = 0

    def get_dynamic_action_gap_samples(self) -> list[dict[str, Any]]:
        return self._dynamic_action_gap_samples

    def load_dynamic_action_gap_gate(self, checkpoint_path: str) -> None:
        """Load a trained Dynamic ActionGap gate once per model instance."""
        if self._dynamic_action_gap_checkpoint == checkpoint_path:
            return
        gate, payload = load_dynamic_action_gap_gate(
            checkpoint_path,
            device=self.device,
            dtype=self.torch_dtype,
        )
        if gate.hidden_dim != int(self.action_expert.hidden_dim):
            raise ValueError(
                "Dynamic ActionGap gate hidden size does not match Action DiT: "
                f"{gate.hidden_dim} vs {self.action_expert.hidden_dim}."
            )
        if gate.meta_dim != 3:
            raise ValueError(
                f"This inference path expects three gate metadata features, got {gate.meta_dim}."
            )
        self.dynamic_action_gap_gate = gate
        self._dynamic_action_gap_checkpoint = checkpoint_path
        self._dynamic_action_gap_default_threshold = float(
            payload["default_threshold"]
        )

    def get_internal_distillation_samples(self) -> list[dict[str, Any]]:
        """Return samples collected by ``infer_action`` for offline head training."""
        return self._internal_distillation_samples

    def load_internal_action_heads(self, checkpoint_path: str) -> None:
        """Load offline-distilled intermediate heads once per model instance."""
        if self._internal_action_heads_checkpoint == checkpoint_path:
            return
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        heads = nn.ModuleDict()
        for layer, spec in payload["heads"].items():
            head = InternalActionHead(
                hidden_dim=int(spec["hidden_dim"]),
                action_dim=int(spec["action_dim"]),
            )
            head.load_state_dict(spec["state_dict"], strict=True)
            heads[str(int(layer))] = head.to(device=self.device, dtype=self.torch_dtype).eval()
        self._internal_action_heads = heads
        self._internal_action_heads_checkpoint = checkpoint_path

    def configure_internal_lora_branch(
        self,
        fork_layer: int = 4,
        source_start_layer: int = 30,
        source_end_layer: int = 30,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        checkpoint_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create the teacher-guided deep-copy branch and its optimizer."""
        branch = DeepCopyInternalActionBranch(
            action_expert=self.action_expert,
            fork_layer=fork_layer,
            source_start_layer=source_start_layer,
            source_end_layer=source_end_layer,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            train_output_head=True,
        ).to(device=self.device, dtype=self.torch_dtype)
        if checkpoint_path:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if payload.get("format_version") == 2:
                missing, unexpected = branch.load_state_dict(
                    payload["trainable_state_dict"], strict=False
                )
                allowed_missing = {
                    name for name, parameter in branch.named_parameters()
                    if not parameter.requires_grad
                }
                disallowed_missing = sorted(set(missing) - allowed_missing)
                if disallowed_missing or unexpected:
                    raise ValueError(
                        "Invalid adapter-only Internal LoRA checkpoint: "
                        f"missing={disallowed_missing}, unexpected={unexpected}"
                    )
            else:
                branch.load_state_dict(payload["state_dict"], strict=True)
        self.internal_lora_branch = branch
        self._internal_lora_checkpoint = checkpoint_path
        trainable = [parameter for parameter in branch.parameters() if parameter.requires_grad]
        self._internal_lora_optimizer = torch.optim.AdamW(
            trainable, lr=learning_rate, weight_decay=weight_decay
        )
        self._internal_lora_updates = 0
        self._internal_lora_losses = []
        self._internal_lora_best_loss = None
        self._internal_lora_best_update = None
        self._internal_lora_best_trainable_state = None
        all_params = sum(parameter.numel() for parameter in branch.parameters())
        trainable_params = sum(parameter.numel() for parameter in trainable)
        copied_independently = []
        for offset, copied_block in enumerate(branch.blocks):
            source_block = self.action_expert.blocks[source_start_layer - 1 + offset]
            copied_independently.append({
                "source_layer": source_start_layer + offset,
                "different_module_object": copied_block is not source_block,
                "different_q_weight_storage": (
                    copied_block.self_attn.q.base.weight.data_ptr()
                    != source_block.self_attn.q.weight.data_ptr()
                ),
            })
        trainable_names = branch.trainable_parameter_names()
        invalid_trainable_names = [
            name for name in trainable_names
            if "lora_A" not in name and "lora_B" not in name and not name.startswith("head.")
        ]
        if invalid_trainable_names:
            raise RuntimeError(f"Unexpected trainable Internal Branch parameters: {invalid_trainable_names}")
        if not all(
            item["different_module_object"] and item["different_q_weight_storage"]
            for item in copied_independently
        ):
            raise RuntimeError("Internal Branch is not an independent deep copy of the source blocks.")
        return {
            "source_start_layer": source_start_layer,
            "source_end_layer": source_end_layer,
            "fork_layer": fork_layer,
            "num_blocks": len(branch.blocks),
            "all_parameters": all_params,
            "trainable_parameters": trainable_params,
            "trainable_fraction": trainable_params / max(all_params, 1),
            "trainable_names": trainable_names,
            "invalid_trainable_names": invalid_trainable_names,
            "copied_independently": copied_independently,
        }

    def _internal_lora_trainable_state(self) -> dict[str, torch.Tensor]:
        if self.internal_lora_branch is None:
            raise RuntimeError("Internal LoRA branch has not been configured.")
        trainable_names = set(self.internal_lora_branch.trainable_parameter_names())
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.internal_lora_branch.state_dict().items()
            if name in trainable_names
        }

    def save_internal_lora_branch(
        self, checkpoint_path: str, *, use_best: bool = False
    ) -> None:
        if self.internal_lora_branch is None:
            raise RuntimeError("Internal LoRA branch has not been configured.")
        trainable_names = set(self.internal_lora_branch.trainable_parameter_names())
        if use_best:
            if self._internal_lora_best_trainable_state is None:
                raise RuntimeError("No best Internal LoRA state has been recorded.")
            trainable_state = self._internal_lora_best_trainable_state
        else:
            trainable_state = self._internal_lora_trainable_state()
        payload = {
            "format_version": 2,
            "fork_layer": self.internal_lora_branch.fork_layer,
            "source_start_layer": self.internal_lora_branch.source_start_layer,
            "source_end_layer": self.internal_lora_branch.source_end_layer,
            "trainable_state_dict": trainable_state,
            "trainable_names": sorted(trainable_names),
            "updates": self._internal_lora_updates,
            "losses": self._internal_lora_losses,
            "saved_state": "best" if use_best else "final",
            "best_loss": self._internal_lora_best_loss,
            "best_update": self._internal_lora_best_update,
        }
        torch.save(payload, checkpoint_path)

    def get_internal_lora_training_summary(self) -> dict[str, Any]:
        losses = self._internal_lora_losses
        return {
            "updates": self._internal_lora_updates,
            "loss_mean": float(sum(losses) / len(losses)) if losses else None,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "best_loss": self._internal_lora_best_loss,
            "best_update": self._internal_lora_best_update,
        }

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        return_intermediates: bool = False,
        capture_layers: Optional[set[int]] = None,
        stop_after_layer: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_forward = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            capture_layers=capture_layers,
            stop_after_layer=stop_after_layer,
        )
        if capture_layers:
            action_tokens, captured_tokens = action_forward
        else:
            action_tokens = action_forward
            captured_tokens = {}
        pred_action = self.action_expert.post_dit(action_tokens, action_pre)
        if return_intermediates or capture_layers:
            intermediates = {
                "h0": action_pre["tokens"],
                "hL": action_tokens,
                "action_freqs": action_pre["freqs"],
                "action_t_mod": action_pre["t_mod"],
                "action_context": action_pre["context"],
                "action_context_mask": action_pre["context_mask"],
            }
            if capture_layers:
                intermediates["captured_tokens"] = captured_tokens
                intermediates["layer_predictions"] = {
                    layer: self.action_expert.post_dit(tokens, action_pre)
                    for layer, tokens in captured_tokens.items()
                }
            return pred_action, intermediates
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_from_cached_residual(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        if action_pre["tokens"].shape != residual.shape:
            raise ValueError(
                "C3ache residual shape does not match action tokens: "
                f"{tuple(residual.shape)} vs {tuple(action_pre['tokens'].shape)}"
            )
        action_tokens = action_pre["tokens"] + residual.to(
            device=action_pre["tokens"].device,
            dtype=action_pre["tokens"].dtype,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        profile_timing: bool = False,
        analyze_c3cache_residuals: bool = False,
        enable_c3cache: bool = False,
        c3cache_start_step: int = 0,
        c3cache_end_step: int = 6,
        c3cache_refresh_interval: int = 4,
        analyze_internal_layers: bool = False,
        internal_profile_layers: tuple[int, ...] = (6, 12, 18, 24),
        collect_internal_distillation: bool = False,
        internal_distillation_layers: tuple[int, ...] = (12, 18),
        enable_internal_head: bool = False,
        internal_head_checkpoint: Optional[str] = None,
        internal_head_layer: int = 18,
        internal_head_steps: tuple[int, ...] = (0,),
        train_internal_lora: bool = False,
        internal_lora_train_steps: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6),
        internal_lora_max_updates: Optional[int] = None,
        enable_internal_lora_branch: bool = False,
        internal_lora_checkpoint: Optional[str] = None,
        internal_lora_fork_layer: int = 4,
        internal_lora_source_start_layer: int = 30,
        internal_lora_source_end_layer: int = 30,
        internal_lora_rank: int = 8,
        internal_lora_alpha: float = 16.0,
        internal_lora_inference_steps: tuple[int, ...] = (0,),
        internal_lora_merge_for_inference: bool = True,
        enable_action_gap_schedule: bool = False,
        action_gap: int = 2,
        enable_dynamic_action_gap: bool = False,
        dynamic_action_gap_checkpoint: Optional[str] = None,
        dynamic_action_gap_threshold: Optional[float] = None,
        dynamic_action_gap_max_internal_run: int = 3,
        dynamic_action_gap_anchor_action_gap: Optional[int] = None,
        enable_oracle_action_gap: bool = False,
        oracle_action_gap_threshold: float = 0.36,
        compute_matched_action_gap_mode: Optional[str] = None,
        compute_matched_action_gap_target_internal_ratio: float = 0.65,
        compute_matched_action_gap_seed: int = 0,
        collect_dynamic_action_gap_data: bool = False,
    ) -> dict[str, Any]:
        timing: dict[str, Any] = {}

        def sync_for_timing() -> None:
            if profile_timing and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        sync_for_timing()
        infer_start = time.perf_counter()
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        sync_for_timing()
        image_encode_start = time.perf_counter()
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        sync_for_timing()
        if profile_timing:
            timing["image_encode_ms"] = (time.perf_counter() - image_encode_start) * 1000.0
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            sync_for_timing()
            prompt_encode_start = time.perf_counter()
            context, context_mask = self.encode_prompt(prompt)
            sync_for_timing()
            if profile_timing:
                timing["prompt_encode_ms"] = (time.perf_counter() - prompt_encode_start) * 1000.0
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        sync_for_timing()
        video_pre_start = time.perf_counter()
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        sync_for_timing()
        if profile_timing:
            timing["video_pre_dit_ms"] = (time.perf_counter() - video_pre_start) * 1000.0
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        sync_for_timing()
        video_prefill_start = time.perf_counter()
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        sync_for_timing()
        if profile_timing:
            timing["video_kv_prefill_ms"] = (time.perf_counter() - video_prefill_start) * 1000.0

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        action_denoise_step_ms: list[float] = []
        residual_analysis_steps: list[dict[str, Any]] = []
        analysis_residuals: dict[int, torch.Tensor] = {}
        c3cache_residuals: dict[int, torch.Tensor] = {}
        c3cache_step_modes: list[str] = []
        c3cache_full_steps = 0
        c3cache_cached_steps = 0
        internal_layer_steps: list[dict[str, Any]] = []
        profile_layers = tuple(sorted(set(int(layer) for layer in internal_profile_layers)))
        if analyze_internal_layers:
            invalid_layers = [
                layer for layer in profile_layers
                if layer < 1 or layer >= self.mot.num_layers
            ]
            if invalid_layers:
                raise ValueError(
                    "Internal profiling layers must be 1-based and strictly shallower than "
                    f"the full {self.mot.num_layers}-layer ActionDiT; got {invalid_layers}."
                )
        distillation_layers = tuple(
            sorted(set(int(layer) for layer in internal_distillation_layers))
        )
        if collect_internal_distillation:
            invalid_layers = [
                layer for layer in distillation_layers
                if layer < 1 or layer >= self.mot.num_layers
            ]
            if invalid_layers:
                raise ValueError(
                    "Internal distillation layers must be 1-based and strictly shallower than "
                    f"the full {self.mot.num_layers}-layer ActionDiT; got {invalid_layers}."
                )
        captured_layers = set(profile_layers if analyze_internal_layers else ())
        if collect_internal_distillation:
            captured_layers.update(distillation_layers)
        internal_head_steps_set = {int(step) for step in internal_head_steps}
        if enable_internal_head:
            if enable_c3cache:
                raise ValueError("Internal Head and C3ache cannot be enabled together in this prototype.")
            if analyze_internal_layers or collect_internal_distillation:
                raise ValueError("Disable profiling/collection when deploying the Internal Head.")
            if not internal_head_checkpoint:
                raise ValueError("`internal_head_checkpoint` is required when Internal Head is enabled.")
            if not 1 <= internal_head_layer < self.mot.num_layers:
                raise ValueError(
                    f"`internal_head_layer` must be in [1, {self.mot.num_layers - 1}]."
                )
            self.load_internal_action_heads(internal_head_checkpoint)
            if str(internal_head_layer) not in self._internal_action_heads:
                raise ValueError(
                    f"Checkpoint does not contain a head for layer {internal_head_layer}."
                )
            invalid_steps = sorted(
                step for step in internal_head_steps_set
                if step < 0 or step >= len(infer_timesteps_action)
            )
            if invalid_steps:
                raise ValueError(f"Invalid Internal Head denoise steps: {invalid_steps}.")
        internal_head_used_steps: list[int] = []
        internal_lora_train_steps_set = {int(step) for step in internal_lora_train_steps}
        internal_lora_inference_steps_set = {
            int(step) for step in internal_lora_inference_steps
        }
        action_gap_schedule: Optional[list[str]] = None
        compute_matched_mode = (
            None
            if compute_matched_action_gap_mode is None
            else str(compute_matched_action_gap_mode).strip().lower()
        )
        if collect_dynamic_action_gap_data:
            if (
                enable_dynamic_action_gap
                or enable_oracle_action_gap
                or enable_action_gap_schedule
                or compute_matched_mode
            ):
                raise ValueError(
                    "Dynamic ActionGap data collection must run the full teacher path."
                )
            if not enable_internal_lora_branch:
                raise ValueError(
                    "Dynamic ActionGap data collection requires Internal LoRA inference."
                )
            internal_lora_inference_steps_set = set()
        enabled_routing_modes = sum(
            bool(mode)
            for mode in (
                enable_dynamic_action_gap,
                enable_oracle_action_gap,
                enable_action_gap_schedule,
                compute_matched_mode,
            )
        )
        if enabled_routing_modes > 1:
            raise ValueError(
                "Dynamic, fixed ActionGap, and compute-matched routing are mutually exclusive."
            )
        if enable_oracle_action_gap:
            if not enable_internal_lora_branch:
                raise ValueError(
                    "True-gap oracle routing requires Internal LoRA inference."
                )
            if oracle_action_gap_threshold < 0:
                raise ValueError("`oracle_action_gap_threshold` must be non-negative.")
            if dynamic_action_gap_max_internal_run <= 0:
                raise ValueError("`dynamic_action_gap_max_internal_run` must be positive.")
            # The oracle chooses every denoising route after observing both outputs.
            # Do not let the legacy explicit step list bypass the oracle (the
            # default list contains step 0).
            internal_lora_inference_steps_set = set()
        if enable_dynamic_action_gap:
            if not enable_internal_lora_branch:
                raise ValueError(
                    "`enable_dynamic_action_gap=True` requires Internal LoRA inference."
                )
            if not dynamic_action_gap_checkpoint:
                raise ValueError(
                    "`dynamic_action_gap_checkpoint` is required for dynamic routing."
                )
            if dynamic_action_gap_max_internal_run <= 0:
                raise ValueError(
                    "`dynamic_action_gap_max_internal_run` must be positive."
                )
            if (
                dynamic_action_gap_anchor_action_gap is not None
                and dynamic_action_gap_anchor_action_gap <= 0
            ):
                raise ValueError(
                    "`dynamic_action_gap_anchor_action_gap` must be positive."
                )
            # The learned gate, rather than an explicit step list, chooses the route.
            internal_lora_inference_steps_set = set()
        if enable_action_gap_schedule:
            if not enable_internal_lora_branch:
                raise ValueError(
                    "`enable_action_gap_schedule=True` requires Internal LoRA inference."
                )
            action_gap_schedule = build_action_gap_schedule(
                num_steps=len(infer_timesteps_action), action_gap=action_gap
            )
            internal_lora_inference_steps_set = {
                step for step, mode in enumerate(action_gap_schedule)
                if mode == "internal"
            }
        if compute_matched_mode:
            if compute_matched_mode not in {"fixed", "random"}:
                raise ValueError(
                    "`compute_matched_action_gap_mode` must be 'fixed' or 'random'."
                )
            if not enable_internal_lora_branch:
                raise ValueError(
                    "Compute-matched ActionGap routing requires Internal LoRA inference."
                )
            action_gap_schedule = build_compute_matched_action_gap_schedule(
                num_steps=len(infer_timesteps_action),
                chunk_index=self._dynamic_action_gap_chunk_index,
                target_internal_ratio=compute_matched_action_gap_target_internal_ratio,
                mode=compute_matched_mode,
                seed=compute_matched_action_gap_seed,
                max_internal_run=dynamic_action_gap_max_internal_run,
                anchor_action_gap=dynamic_action_gap_anchor_action_gap,
            )
            internal_lora_inference_steps_set = {
                step for step, route in enumerate(action_gap_schedule)
                if route == "internal"
            }
        if enable_internal_lora_branch:
            if train_internal_lora or enable_internal_head or enable_c3cache:
                raise ValueError(
                    "Internal LoRA inference must run without online training, linear Internal Head, or C3ache."
                )
            if analyze_internal_layers or collect_internal_distillation:
                raise ValueError("Disable profiling/collection for Internal LoRA inference.")
            if not internal_lora_checkpoint:
                raise ValueError("`internal_lora_checkpoint` is required for inference.")
            if (
                self.internal_lora_branch is None
                or self._internal_lora_checkpoint != internal_lora_checkpoint
            ):
                self.configure_internal_lora_branch(
                    fork_layer=internal_lora_fork_layer,
                    source_start_layer=internal_lora_source_start_layer,
                    source_end_layer=internal_lora_source_end_layer,
                    lora_rank=internal_lora_rank,
                    lora_alpha=internal_lora_alpha,
                    checkpoint_path=internal_lora_checkpoint,
                )
                if internal_lora_merge_for_inference:
                    self.internal_lora_branch.merge_lora_for_inference()
            invalid_steps = sorted(
                step for step in internal_lora_inference_steps_set
                if step < 0 or step >= len(infer_timesteps_action)
            )
            if invalid_steps:
                raise ValueError(f"Invalid Internal LoRA inference steps: {invalid_steps}.")
        dynamic_action_gap_effective_threshold: Optional[float] = None
        if enable_dynamic_action_gap:
            self.load_dynamic_action_gap_gate(str(dynamic_action_gap_checkpoint))
            if self.internal_lora_branch.fork_layer != internal_lora_fork_layer:
                raise ValueError(
                    "Dynamic ActionGap gate must route at the configured Internal LoRA fork layer."
                )
            dynamic_action_gap_effective_threshold = (
                self._dynamic_action_gap_default_threshold
                if dynamic_action_gap_threshold is None
                else float(dynamic_action_gap_threshold)
            )
            if dynamic_action_gap_effective_threshold is None:
                raise ValueError("Dynamic ActionGap threshold is unavailable.")
        if collect_dynamic_action_gap_data or enable_oracle_action_gap:
            captured_layers.add(self.internal_lora_branch.fork_layer)
        internal_lora_used_steps: list[int] = []
        internal_lora_step_modes: list[str] = []
        dynamic_action_gap_steps: list[dict[str, Any]] = []
        oracle_action_gap_steps: list[dict[str, Any]] = []
        dynamic_steps_since_full = 0
        if train_internal_lora and (
            self.internal_lora_branch is None or self._internal_lora_optimizer is None
        ):
            raise RuntimeError("Configure the Internal LoRA branch before online distillation.")
        if train_internal_lora:
            captured_layers.add(self.internal_lora_branch.fork_layer)
        if enable_c3cache:
            if c3cache_start_step < 0 or c3cache_end_step < c3cache_start_step:
                raise ValueError(
                    "Invalid C3ache step range: "
                    f"{c3cache_start_step}..{c3cache_end_step}"
                )
            if c3cache_refresh_interval < 0:
                raise ValueError(
                    f"`c3cache_refresh_interval` must be >= 0, got {c3cache_refresh_interval}"
                )
        refresh_cache_this_chunk = (
            enable_c3cache
            and (
                self._c3cache_chunk_index == 0
                or (
                    c3cache_refresh_interval > 0
                    and self._c3cache_chunk_index % c3cache_refresh_interval == 0
                )
            )
        )
        for step_idx, (step_t_action, step_delta_action) in enumerate(
            zip(infer_timesteps_action, infer_deltas_action)
        ):
            sync_for_timing()
            denoise_step_start = time.perf_counter()
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            step_in_cache_range = c3cache_start_step <= step_idx <= c3cache_end_step
            use_cached_residual = (
                enable_c3cache
                and step_in_cache_range
                and not refresh_cache_this_chunk
                and step_idx in self._c3cache_residuals
            )
            use_internal_head = enable_internal_head and step_idx in internal_head_steps_set
            use_internal_lora = (
                enable_internal_lora_branch
                and step_idx in internal_lora_inference_steps_set
            )
            if (
                enable_internal_lora_branch
                and not enable_dynamic_action_gap
                and not enable_oracle_action_gap
            ):
                internal_lora_step_modes.append(
                    "internal" if use_internal_lora else "full"
                )
            if enable_dynamic_action_gap:
                branch = self.internal_lora_branch
                branch_input_layer = branch.fork_layer
                _, intermediates = self._predict_action_noise_with_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                    return_intermediates=True,
                    capture_layers={branch_input_layer},
                    stop_after_layer=branch_input_layer,
                )
                fork_hidden = intermediates["captured_tokens"][branch_input_layer]
                pooled_hidden = fork_hidden.mean(dim=1)
                gate_meta = build_dynamic_action_gap_meta(
                    batch_size=int(pooled_hidden.shape[0]),
                    timestep=float(step_t_action.item()),
                    step_index=step_idx,
                    num_steps=len(infer_timesteps_action),
                    steps_since_full=dynamic_steps_since_full,
                    max_internal_run=dynamic_action_gap_max_internal_run,
                    device=pooled_hidden.device,
                    dtype=pooled_hidden.dtype,
                )
                predicted_gap = float(
                    self.dynamic_action_gap_gate(pooled_hidden, gate_meta)
                    .float()
                    .item()
                )
                route, route_reason = select_dynamic_action_gap_route(
                    predicted_gap=predicted_gap,
                    threshold=dynamic_action_gap_effective_threshold,
                    step_index=step_idx,
                    num_steps=len(infer_timesteps_action),
                    steps_since_full=dynamic_steps_since_full,
                    max_internal_run=dynamic_action_gap_max_internal_run,
                    anchor_action_gap=dynamic_action_gap_anchor_action_gap,
                )
                if route == "internal":
                    branch_tokens = self.mot.forward_internal_action_branch_with_video_cache(
                        action_tokens=fork_hidden,
                        branch_blocks=branch.blocks,
                        source_start_layer=branch.source_start_layer,
                        action_freqs=intermediates["action_freqs"],
                        action_t_mod=intermediates["action_t_mod"],
                        action_context_payload={
                            "context": intermediates["action_context"],
                            "mask": intermediates["action_context_mask"],
                        },
                        video_kv_cache=video_kv_cache,
                        attention_mask=attention_mask,
                        video_seq_len=video_seq_len,
                    )
                    pred_action_posi = branch.head(branch_tokens)
                    internal_lora_used_steps.append(step_idx)
                    dynamic_steps_since_full += 1
                else:
                    full_tokens = self.mot.forward_internal_action_branch_with_video_cache(
                        action_tokens=fork_hidden,
                        branch_blocks=self.action_expert.blocks[branch_input_layer:],
                        source_start_layer=branch_input_layer + 1,
                        action_freqs=intermediates["action_freqs"],
                        action_t_mod=intermediates["action_t_mod"],
                        action_context_payload={
                            "context": intermediates["action_context"],
                            "mask": intermediates["action_context_mask"],
                        },
                        video_kv_cache=video_kv_cache,
                        attention_mask=attention_mask,
                        video_seq_len=video_seq_len,
                    )
                    pred_action_posi = self.action_expert.head(full_tokens)
                    dynamic_steps_since_full = 0
                internal_lora_step_modes.append(route)
                dynamic_action_gap_steps.append({
                    "step": int(step_idx),
                    "timestep": float(step_t_action.item()),
                    "predicted_gap": predicted_gap,
                    "threshold": float(dynamic_action_gap_effective_threshold),
                    "route": route,
                    "reason": route_reason,
                })
            elif use_internal_lora:
                branch = self.internal_lora_branch
                branch_input_layer = branch.fork_layer
                prediction = self._predict_action_noise_with_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                    return_intermediates=True,
                    capture_layers={branch_input_layer},
                    stop_after_layer=branch_input_layer,
                )
                _, intermediates = prediction
                branch_tokens = self.mot.forward_internal_action_branch_with_video_cache(
                    action_tokens=intermediates["captured_tokens"][branch_input_layer],
                    branch_blocks=branch.blocks,
                    source_start_layer=branch.source_start_layer,
                    action_freqs=intermediates["action_freqs"],
                    action_t_mod=intermediates["action_t_mod"],
                    action_context_payload={
                        "context": intermediates["action_context"],
                        "mask": intermediates["action_context_mask"],
                    },
                    video_kv_cache=video_kv_cache,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                )
                pred_action_posi = branch.head(branch_tokens)
                internal_lora_used_steps.append(step_idx)
            elif use_internal_head:
                prediction = self._predict_action_noise_with_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                    return_intermediates=True,
                    capture_layers={internal_head_layer},
                    stop_after_layer=internal_head_layer,
                )
                _, intermediates = prediction
                hidden = intermediates["captured_tokens"][internal_head_layer]
                pred_action_posi = self._internal_action_heads[str(internal_head_layer)](hidden)
                internal_head_used_steps.append(step_idx)
            elif use_cached_residual:
                pred_action_posi = self._predict_action_noise_from_cached_residual(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    residual=self._c3cache_residuals[step_idx],
                )
                c3cache_step_modes.append("cached")
                c3cache_cached_steps += 1
            else:
                prediction = self._predict_action_noise_with_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                    return_intermediates=(
                        analyze_c3cache_residuals
                        or (enable_c3cache and step_in_cache_range)
                        or analyze_internal_layers
                        or collect_internal_distillation
                        or train_internal_lora
                        or collect_dynamic_action_gap_data
                        or enable_oracle_action_gap
                    ),
                    capture_layers=captured_layers or None,
                )
                if (
                    analyze_c3cache_residuals
                    or (enable_c3cache and step_in_cache_range)
                    or analyze_internal_layers
                    or collect_internal_distillation
                    or train_internal_lora
                    or collect_dynamic_action_gap_data
                    or enable_oracle_action_gap
                ):
                    pred_action_posi, intermediates = prediction
                    if collect_internal_distillation:
                        self._internal_distillation_samples.append(
                            {
                                "chunk_index": int(self._internal_distillation_chunk_index),
                                "step_index": int(step_idx),
                                "timestep": timestep_action.detach().to(
                                    device="cpu", dtype=torch.float32
                                ),
                                "teacher_prediction": pred_action_posi.detach().to(
                                    device="cpu", dtype=torch.bfloat16
                                ),
                                "hidden_by_layer": {
                                    int(layer): intermediates["captured_tokens"][layer]
                                    .detach()
                                    .to(device="cpu", dtype=torch.bfloat16)
                                    for layer in distillation_layers
                                },
                            }
                        )
                    if collect_dynamic_action_gap_data or enable_oracle_action_gap:
                        branch = self.internal_lora_branch
                        branch_input_layer = branch.fork_layer
                        fork_hidden = intermediates["captured_tokens"][branch_input_layer]
                        branch_tokens = self.mot.forward_internal_action_branch_with_video_cache(
                            action_tokens=fork_hidden,
                            branch_blocks=branch.blocks,
                            source_start_layer=branch.source_start_layer,
                            action_freqs=intermediates["action_freqs"],
                            action_t_mod=intermediates["action_t_mod"],
                            action_context_payload={
                                "context": intermediates["action_context"],
                                "mask": intermediates["action_context_mask"],
                            },
                            video_kv_cache=video_kv_cache,
                            attention_mask=attention_mask,
                            video_seq_len=video_seq_len,
                        )
                        internal_prediction = branch.head(branch_tokens)
                        teacher_float = pred_action_posi.float()
                        normalized_gap = (
                            (internal_prediction.float() - teacher_float).square().mean()
                            / teacher_float.square().mean().clamp_min(1e-12)
                        )
                        if enable_oracle_action_gap:
                            oracle_score = float(
                                torch.log1p(normalized_gap).detach().cpu()
                            )
                            route, route_reason = select_dynamic_action_gap_route(
                                predicted_gap=oracle_score,
                                threshold=float(oracle_action_gap_threshold),
                                step_index=step_idx,
                                num_steps=len(infer_timesteps_action),
                                steps_since_full=dynamic_steps_since_full,
                                max_internal_run=dynamic_action_gap_max_internal_run,
                                anchor_action_gap=dynamic_action_gap_anchor_action_gap,
                            )
                            if route == "internal":
                                pred_action_posi = internal_prediction
                                internal_lora_used_steps.append(step_idx)
                                dynamic_steps_since_full += 1
                            else:
                                dynamic_steps_since_full = 0
                            internal_lora_step_modes.append(route)
                            oracle_action_gap_steps.append({
                                "step": int(step_idx),
                                "timestep": float(step_t_action.item()),
                                "true_gap": float(normalized_gap.detach().cpu()),
                                "oracle_score": oracle_score,
                                "threshold": float(oracle_action_gap_threshold),
                                "route": route,
                                "reason": route_reason,
                            })
                        if collect_dynamic_action_gap_data:
                            self._dynamic_action_gap_samples.append({
                                "chunk_index": int(
                                    self._dynamic_action_gap_collection_chunk_index
                                ),
                                "step_index": int(step_idx),
                                "num_steps": int(len(infer_timesteps_action)),
                                "timestep": float(step_t_action.item()),
                                "pooled_hidden": fork_hidden.mean(dim=1)
                                .detach()
                                .to(device="cpu", dtype=torch.bfloat16),
                                "gap": float(normalized_gap.detach().cpu()),
                            })
                    update_budget_available = (
                        internal_lora_max_updates is None
                        or self._internal_lora_updates < internal_lora_max_updates
                    )
                    if (
                        train_internal_lora
                        and update_budget_available
                        and step_idx in internal_lora_train_steps_set
                    ):
                        branch = self.internal_lora_branch
                        branch_input_layer = branch.fork_layer
                        branch_input = intermediates["captured_tokens"][branch_input_layer].detach()
                        teacher_target = pred_action_posi.detach()
                        self._internal_lora_optimizer.zero_grad(set_to_none=True)
                        with torch.enable_grad():
                            branch_tokens = self.mot.forward_internal_action_branch_with_video_cache(
                                action_tokens=branch_input,
                                branch_blocks=branch.blocks,
                                source_start_layer=branch.source_start_layer,
                                action_freqs=intermediates["action_freqs"].detach(),
                                action_t_mod=intermediates["action_t_mod"].detach(),
                                action_context_payload={
                                    "context": intermediates["action_context"].detach(),
                                    "mask": intermediates["action_context_mask"],
                                },
                                video_kv_cache=[
                                    {key: value.detach() for key, value in item.items()}
                                    for item in video_kv_cache
                                ],
                                attention_mask=attention_mask,
                                video_seq_len=video_seq_len,
                            )
                            student_prediction = branch.head(branch_tokens)
                            distillation_loss = F.mse_loss(
                                student_prediction.float(), teacher_target.float()
                            )
                            distillation_loss.backward()
                        self._internal_lora_optimizer.step()
                        self._internal_lora_updates += 1
                        loss_value = float(distillation_loss.detach().cpu())
                        self._internal_lora_losses.append(loss_value)
                        if (
                            self._internal_lora_best_loss is None
                            or loss_value < self._internal_lora_best_loss
                        ):
                            self._internal_lora_best_loss = loss_value
                            self._internal_lora_best_update = self._internal_lora_updates
                            self._internal_lora_best_trainable_state = (
                                self._internal_lora_trainable_state()
                            )
                    if analyze_internal_layers:
                        layer_metrics = []
                        full_flat = pred_action_posi.reshape(1, -1).float()
                        for layer, layer_prediction in intermediates["layer_predictions"].items():
                            layer_flat = layer_prediction.reshape(1, -1).float()
                            diff = layer_flat - full_flat
                            layer_metrics.append({
                                "layer": int(layer),
                                "cosine_similarity_to_full": float(
                                    F.cosine_similarity(layer_flat, full_flat, dim=1).item()
                                ),
                                "relative_l2_error": float(
                                    diff.norm().div(full_flat.norm().clamp_min(1e-12)).item()
                                ),
                                "mean_absolute_error": float(diff.abs().mean().item()),
                            })
                        internal_layer_steps.append({
                            "step_index": int(step_idx),
                            "timestep": float(step_t_action.item()),
                            "layers": layer_metrics,
                        })
                    residual = (intermediates["hL"] - intermediates["h0"]).detach()
                    if enable_c3cache and step_in_cache_range:
                        c3cache_residuals[step_idx] = residual.clone()
                    if analyze_c3cache_residuals:
                        analysis_residuals[step_idx] = residual.clone()
                        previous_residual = self._c3cache_analysis_prev_residuals.get(step_idx)
                        cosine_similarity = None
                        if previous_residual is not None:
                            if previous_residual.shape != residual.shape:
                                raise ValueError(
                                    "C3ache analysis residual shape changed across chunks at "
                                    f"step {step_idx}: {tuple(previous_residual.shape)} vs {tuple(residual.shape)}"
                                )
                            cosine_similarity = float(
                                F.cosine_similarity(
                                    previous_residual.reshape(1, -1).float(),
                                    residual.reshape(1, -1).float(),
                                    dim=1,
                                ).item()
                            )
                        residual_analysis_steps.append(
                            {
                                "step_index": step_idx,
                                "timestep": float(step_t_action.item()),
                                "cosine_similarity_to_previous_chunk": cosine_similarity,
                                "residual_l2_norm": float(residual.float().norm().item()),
                            }
                        )
                else:
                    pred_action_posi = prediction
                if enable_c3cache:
                    c3cache_step_modes.append("full")
                    c3cache_full_steps += 1
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            sync_for_timing()
            if profile_timing:
                action_denoise_step_ms.append((time.perf_counter() - denoise_step_start) * 1000.0)

        sync_for_timing()
        result = {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
        if analyze_c3cache_residuals:
            result["c3cache_residual_analysis"] = {
                "chunk_index": self._c3cache_analysis_chunk_index,
                "steps": residual_analysis_steps,
            }
            self._c3cache_analysis_prev_residuals = analysis_residuals
            self._c3cache_analysis_chunk_index += 1
        if enable_c3cache:
            if refresh_cache_this_chunk:
                self._c3cache_residuals = c3cache_residuals
            result["c3cache"] = {
                "chunk_index": self._c3cache_chunk_index,
                "refresh_cache": refresh_cache_this_chunk,
                "step_modes": c3cache_step_modes,
                "full_steps": c3cache_full_steps,
                "cached_steps": c3cache_cached_steps,
                "cache_start_step": c3cache_start_step,
                "cache_end_step": c3cache_end_step,
                "refresh_interval": c3cache_refresh_interval,
            }
            self._c3cache_chunk_index += 1
        if analyze_internal_layers:
            result["internal_layer_profile"] = {
                "profile_layers": list(profile_layers),
                "steps": internal_layer_steps,
            }
        if collect_internal_distillation:
            result["internal_distillation"] = {
                "chunk_index": int(self._internal_distillation_chunk_index),
                "layers": list(distillation_layers),
                "num_samples": len(infer_timesteps_action),
            }
            self._internal_distillation_chunk_index += 1
        if collect_dynamic_action_gap_data:
            result["dynamic_action_gap_collection"] = {
                "chunk_index": int(self._dynamic_action_gap_collection_chunk_index),
                "num_samples": int(len(infer_timesteps_action)),
                "fork_layer": int(self.internal_lora_branch.fork_layer),
            }
            self._dynamic_action_gap_collection_chunk_index += 1
        if enable_internal_head:
            result["internal_head"] = {
                "layer": int(internal_head_layer),
                "configured_steps": sorted(internal_head_steps_set),
                "used_steps": internal_head_used_steps,
            }
        if train_internal_lora:
            result["internal_lora_training"] = self.get_internal_lora_training_summary()
        if enable_internal_lora_branch:
            result["internal_lora_inference"] = {
                "source_start_layer": self.internal_lora_branch.source_start_layer,
                "source_end_layer": self.internal_lora_branch.source_end_layer,
                "fork_layer": self.internal_lora_branch.fork_layer,
                "configured_steps": sorted(internal_lora_inference_steps_set),
                "used_steps": internal_lora_used_steps,
                "schedule_type": (
                    "dynamic_action_gap"
                    if enable_dynamic_action_gap
                    else "oracle_action_gap"
                    if enable_oracle_action_gap
                    else f"compute_matched_{compute_matched_mode}"
                    if compute_matched_mode
                    else "action_gap"
                    if enable_action_gap_schedule
                    else "explicit_steps"
                ),
                "action_gap": int(action_gap) if enable_action_gap_schedule else None,
                "step_modes": internal_lora_step_modes,
                "full_steps": [
                    step for step, mode in enumerate(internal_lora_step_modes)
                    if mode == "full"
                ],
                "internal_steps": [
                    step for step, mode in enumerate(internal_lora_step_modes)
                    if mode == "internal"
                ],
            }
            if enable_dynamic_action_gap:
                result["dynamic_action_gap"] = {
                    "chunk_index": int(self._dynamic_action_gap_chunk_index),
                    "checkpoint": str(dynamic_action_gap_checkpoint),
                    "threshold": float(dynamic_action_gap_effective_threshold),
                    "max_internal_run": int(dynamic_action_gap_max_internal_run),
                    "anchor_action_gap": (
                        None
                        if dynamic_action_gap_anchor_action_gap is None
                        else int(dynamic_action_gap_anchor_action_gap)
                    ),
                    "steps": dynamic_action_gap_steps,
                }
                self._dynamic_action_gap_chunk_index += 1
            elif enable_oracle_action_gap:
                result["oracle_action_gap"] = {
                    "chunk_index": int(self._dynamic_action_gap_chunk_index),
                    "threshold": float(oracle_action_gap_threshold),
                    "score": "log1p(normalized_internal_full_mse)",
                    "max_internal_run": int(dynamic_action_gap_max_internal_run),
                    "anchor_action_gap": (
                        None
                        if dynamic_action_gap_anchor_action_gap is None
                        else int(dynamic_action_gap_anchor_action_gap)
                    ),
                    "steps": oracle_action_gap_steps,
                }
                self._dynamic_action_gap_chunk_index += 1
            elif compute_matched_mode:
                result["compute_matched_action_gap"] = {
                    "chunk_index": int(self._dynamic_action_gap_chunk_index),
                    "mode": compute_matched_mode,
                    "target_internal_ratio": float(
                        compute_matched_action_gap_target_internal_ratio
                    ),
                    "seed": int(compute_matched_action_gap_seed),
                    "max_internal_run": int(dynamic_action_gap_max_internal_run),
                    "anchor_action_gap": (
                        None
                        if dynamic_action_gap_anchor_action_gap is None
                        else int(dynamic_action_gap_anchor_action_gap)
                    ),
                    "schedule": list(action_gap_schedule or []),
                }
                self._dynamic_action_gap_chunk_index += 1
        if profile_timing:
            timing["action_denoise_step_ms"] = action_denoise_step_ms
            timing["action_denoise_total_ms"] = float(sum(action_denoise_step_ms))
            timing["action_denoise_mean_ms"] = (
                float(sum(action_denoise_step_ms) / len(action_denoise_step_ms))
                if action_denoise_step_ms
                else 0.0
            )
            timing["num_inference_steps"] = len(action_denoise_step_ms)
            timing["infer_action_total_ms"] = (time.perf_counter() - infer_start) * 1000.0
            result["timing"] = timing
        return result

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
