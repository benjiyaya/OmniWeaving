# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""
OmniWeaving Training Script

This script provides a complete training pipeline for HunyuanVideo-1.5 model.

Quick Start:
1. Data loading:
   - Implement a `torch.utils.data.Dataset` whose `__getitem__` returns one sample dict.
   - Use `training_collate_fn` as `DataLoader(..., collate_fn=training_collate_fn)` so batches
     can include PIL images, nested lists, and strings (default collate will fail on those).
   - Base fields (t2v / i2v): see docstring of `create_dummy_dataloader1()`.
   - Multi-task training: `--dataloader_probs` is required (e.g. ``t2v:0.5,i2v:0.5``). Each name maps
     to a task via `HunyuanVideoTrainer.LOADER_TASK_MAP`; extra keys per task are documented in
     `create_dummy_dataloader2()` … `create_dummy_dataloader5()`.
   - The `create_dummy_dataloader*()` helpers are minimal placeholders—replace with your own
     dataset and paths.

2. Configure training parameters:
   - Set `--pretrained_model_root` to your pretrained model path
   - Adjust training hyperparameters (learning_rate, batch_size, etc.)
   - Configure distributed training settings (sp_size, enable_fsdp, etc.)

3. Run training:
   - Single GPU: python train.py --pretrained_model_root <path> [other args]
   - Multi-GPU: torchrun --nproc_per_node=N train.py --pretrained_model_root <path> [other args]

4. Monitor training:
   - Checkpoints are saved to `output_dir` at intervals specified by `--save_interval`
   - Validation videos are generated at intervals specified by `--validation_interval`
   - Training logs are printed to console at intervals specified by `--log_interval`

5. Resume training:
   - Use `--resume_from_checkpoint <checkpoint_dir>` to resume from a saved checkpoint

Full per-task sample layouts: `create_dummy_dataloader1()` … `create_dummy_dataloader5()`.
"""

import os
import random
import math
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
)
from diffusers.optimization import get_scheduler
from loguru import logger
import einops
import imageio
import numpy as np
import torchvision.transforms as T
import PIL.Image
from PIL import Image

from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import get_parallel_state, initialize_parallel_state
from hyvideo.optim.muon import get_muon_optimizer
from hyvideo.utils.data_utils import resize_and_center_crop

from torch.distributed._composable.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)


class SNRType(str, Enum):
    UNIFORM = "uniform"
    LOGNORM = "lognorm"
    MIX = "mix"
    MODE = "mode"


def str_to_bool(value):
    """Convert string to boolean, supporting true/false, 1/0, yes/no.
    If value is None (when flag is provided without value), returns True."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ('true', '1', 'yes', 'on'):
            return True
        elif value in ('false', '0', 'no', 'off'):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def save_video(video: torch.Tensor, path: str):
    if video.ndim == 5:
        assert video.shape[0] == 1, f"Expected batch size 1, got {video.shape[0]}"
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    imageio.mimwrite(path, vid.cpu().numpy(), fps=24)


@dataclass
class TrainingConfig:
    # Model paths
    pretrained_model_root: str
    
    # Training parameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_steps: int = 10000
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_muon: bool = True
    
    # Diffusion parameters
    num_train_timesteps: int = 1000
    train_timestep_shift: float = 3.0
    validation_timestep_shift: float = 5.0
    snr_type: SNRType = SNRType.LOGNORM  # Timestep sampling strategy: uniform, lognorm, mix, or mode
    
    # Task configuration
    task_type: str = "t2v"  # "t2v", "i2v", "key_frames_to_v", "multi_imgs_to_v", "v2v", or "tiv2v"
    
    # Text encoder deepstack and setclip settings (must match inference settings)
    deepstack: List[int] = field(default_factory=lambda: [8, 16, 24])  # Deepstack layer indices; use [] to disable
    setclip: bool = True  # Enable CLIP features in text encoder (default: True)
    
    # FSDP configuration
    enable_fsdp: bool = True  # Enable FSDP for distributed training
    enable_gradient_checkpointing: bool = True  # Enable gradient checkpointing
    sp_size: int = 8  # Sequence parallelism size (must divide world_size evenly)
    dp_replicate: int = 1  # Data parallelism replicate size (must divide world_size evenly)
    
    # Data configuration
    batch_size: int = 1
    num_workers: int = 4
    
    # Output configuration
    output_dir: str = "./outputs"
    save_interval: int = 1000
    log_interval: int = 10
    
    # Device configuration
    dtype: str = "bf16"  # "bf16" or "fp32"
    
    # Seed
    seed: int = 42
    
    # Validation configuration
    validation_interval: int = 100  # Run validation every N steps
    validation_prompts: Optional[List[str]] = None  # Prompts for validation (default: single prompt)
    validate_video_length: int = 121  # Video length (number of frames) for validation
    
    # Resume training configuration
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint directory to resume from
    
    # LoRA configuration
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: Optional[List[str]] = None  # Target modules for LoRA (default: all Linear layers)
    pretrained_lora_path: Optional[str] = None
    
    # Multi-dataloader configuration
    dataloader_probs: str = ""  # Dataloader sampling probabilities, e.g. "t2v:0.3,i2v:0.2,key_frames_to_v:0.1,multi_imgs_to_v:0.1,v2v:0.15,tiv2v:0.15"


class LinearInterpolationSchedule:
    """Simple linear interpolation schedule for flow matching"""
    def __init__(self, T: int = 1000):
        self.T = T
    
    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Linear interpolation: x_t = (1 - t/T) * x0 + (t/T) * x1
        Args:
            x0: starting point (clean latents)
            x1: ending point (noise)
            t: timesteps
        """
        t_normalized = t / self.T
        t_normalized = t_normalized.view(-1, *([1] * (x0.ndim - 1)))
        return (1 - t_normalized) * x0 + t_normalized * x1


class TimestepSampler:

    TRAIN_EPS = 1e-5
    SAMPLE_EPS = 1e-3
    
    def __init__(
        self, 
        T: int = 1000, 
        device: torch.device = None,
        snr_type: SNRType = SNRType.LOGNORM,
    ):
        self.T = T
        self.device = device
        self.snr_type = SNRType(snr_type) if isinstance(snr_type, str) else snr_type
    
    def _check_interval(self, eval: bool = False):
        # For ICPlan-like path with velocity model, use [eps, 1-eps]
        eps = self.SAMPLE_EPS if eval else self.TRAIN_EPS
        t0 = eps
        t1 = 1.0 - eps
        return t0, t1
    
    def sample(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = self.device if self.device is not None else torch.device("cuda")
        
        t0, t1 = self._check_interval(eval=False)
        
        if self.snr_type == SNRType.UNIFORM:
            # Uniform sampling: t = rand() * (t1 - t0) + t0
            t = torch.rand((batch_size,), device=device) * (t1 - t0) + t0
            
        elif self.snr_type == SNRType.LOGNORM:
            # Log-normal sampling: t = 1 / (1 + exp(-u)) * (t1 - t0) + t0
            u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)
            t = 1.0 / (1.0 + torch.exp(-u)) * (t1 - t0) + t0
            
        elif self.snr_type == SNRType.MIX:
            # Mix sampling: 70% lognorm + 30% clipped uniform
            u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)
            t_lognorm = 1.0 / (1.0 + torch.exp(-u)) * (t1 - t0) + t0
            
            # Clipped uniform: delta = 0.0 (0.0~0.01 clip)
            delta = 0.0
            t0_clip = t0 + delta
            t1_clip = t1 - delta
            t_clip_uniform = torch.rand((batch_size,), device=device) * (t1_clip - t0_clip) + t0_clip
            
            # Mix with 70% lognorm, 30% uniform
            mask = (torch.rand((batch_size,), device=device) > 0.3).float()
            t = mask * t_lognorm + (1 - mask) * t_clip_uniform
            
        elif self.snr_type == SNRType.MODE:
            # Mode sampling: t = 1 - u - mode_scale * (cos(pi * u / 2)^2 - 1 + u)
            mode_scale = 1.29
            u = torch.rand(size=(batch_size,), device=device)
            t = 1.0 - u - mode_scale * (torch.cos(math.pi * u / 2.0) ** 2 - 1.0 + u)
            # Scale to [t0, t1] range
            t = t * (t1 - t0) + t0
        else:
            raise ValueError(f"Unknown SNR type: {self.snr_type}")
        
        # Scale to [0, T] range
        timesteps = t * self.T
        return timesteps


def timestep_transform(timesteps: torch.Tensor, T: int, shift: float = 1.0) -> torch.Tensor:
    """Transform timesteps with shift"""
    if shift == 1.0:
        return timesteps
    timesteps_normalized = timesteps / T
    timesteps_transformed = shift * timesteps_normalized / (1 + (shift - 1) * timesteps_normalized)
    return timesteps_transformed * T


def is_src(src, group_src, group):
    assert src is not None or group_src is not None
    assert src is None or group_src is None
    if src is not None:
        return dist.get_rank() == src
    if group_src is not None:
        return dist.get_rank() == dist.get_global_rank(group, group_src)
    raise RuntimeError("src and group_src cannot be both None")

def _resolve_src(src, group_src, group):
    """Convert group_src to global src for PyTorch < 2.4 compatibility."""
    if src is not None:
        return src
    if group_src is not None and group is not None:
        return dist.get_global_rank(group, group_src)
    if group_src is not None:
        return group_src
    raise RuntimeError("src and group_src cannot be both None")


def broadcast_object(
        obj,
        src = None,
        group = None,
        device = None,
        group_src = None,
):
    resolved_src = _resolve_src(src, group_src, group)
    buffer = [obj] if is_src(src, group_src, group) else [None]

    dist.broadcast_object_list(buffer, src=resolved_src, group=group, device=device)
    return buffer[0]

def broadcast_tensor(
        tensor,
        src  = None,
        group = None,
        async_op: bool = False,
        group_src = None,
):
    """shape and dtype safe broadcast of tensor"""
    resolved_src = _resolve_src(src, group_src, group)
    if is_src(src, group_src, group):
        tensor = tensor.cuda().contiguous()
    if is_src(src, group_src, group):
        shape, dtype = tensor.shape, tensor.dtype
    else:
        shape, dtype = None, None
    shape = broadcast_object(shape, src=src, group_src=group_src, group=group)
    dtype = broadcast_object(dtype, src=src, group_src=group_src, group=group)

    buffer = tensor if is_src(src, group_src, group) else torch.empty(shape, device='cuda', dtype=dtype)
    dist.broadcast(buffer, src=resolved_src, group=group, async_op=async_op)
    return buffer


def sync_tensor_for_sp(tensor: torch.Tensor, sp_group) -> torch.Tensor:
    """
    Sync tensor within sequence parallel group.
    Ensures all ranks in the SP group have the same tensor values.
    """
    if sp_group is None:
        return tensor
    if not isinstance(tensor, torch.Tensor):
        obj_list = [tensor]
        dist.broadcast_object_list(obj_list, src=dist.get_global_rank(sp_group, 0), group=sp_group)
        return obj_list[0]
    return broadcast_tensor(tensor, group_src=0, group=sp_group)


def training_collate_fn(batch):
    """Collate batches that mix tensors, scalars, strings, lists, and PIL images.

    PyTorch's default collate does not support PIL or arbitrary nested lists. Here,
    tensors are stacked; int/float become 1-D tensors; str / list / other objects
    (including PIL) are kept as Python lists of length ``batch_size``.
    """
    elem = batch[0]
    result = {}
    for key in elem:
        values = [d[key] for d in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values)
        elif isinstance(values[0], str):
            result[key] = values
        elif isinstance(values[0], list):
            result[key] = values
        else:
            result[key] = values
    return result


class MultiDataloader:
    """
    Wraps several ``DataLoader`` instances and draws one batch per training step.

    Each step picks a loader name with probability ``probabilities[name]`` (weights must
    sum to 1.0), then returns ``(name, batch)`` so the trainer can resolve the task via
    ``LOADER_TASK_MAP``. Iterators are restarted when a loader is exhausted; distributed
    SP ranks stay aligned when ``sample(sp_group=...)`` is used.
    
    Args:
        dataloaders: Name -> ``DataLoader``, same keys as ``probabilities``.
        probabilities: Name -> non-negative float, must sum to 1.0.
    
    Example:
        multi_dl = MultiDataloader(
            dataloaders={"t2v": dl1, "i2v": dl2, "v2v": dl3, "tiv2v": dl4},
            probabilities={"t2v": 0.4, "i2v": 0.2, "v2v": 0.2, "tiv2v": 0.2},
        )
        loader_name, batch = multi_dl.sample()
    """
    def __init__(
        self,
        dataloaders: Dict[str, Any],
        probabilities: Dict[str, float],
    ):
        assert set(dataloaders.keys()) == set(probabilities.keys()), (
            f"Dataloader names must match probability names. "
            f"Dataloaders: {set(dataloaders.keys())}, Probabilities: {set(probabilities.keys())}"
        )
        total_prob = sum(probabilities.values())
        assert abs(total_prob - 1.0) < 1e-6, (
            f"Probabilities must sum to 1.0, got {total_prob}"
        )

        self.dataloaders = dataloaders
        self.probabilities = probabilities
        self.names = list(dataloaders.keys())
        self.probs = [probabilities[name] for name in self.names]
        self._iterators: Dict[str, Any] = {}
        self._epochs: Dict[str, int] = {name: 0 for name in self.names}
        self._reset_iterators()

    def _reset_iterators(self):
        self._iterators = {name: iter(dl) for name, dl in self.dataloaders.items()}

    def _get_batch(self, name: str):
        """Get next batch from the named dataloader, resetting its iterator if exhausted."""
        try:
            return next(self._iterators[name])
        except StopIteration:
            self._epochs[name] += 1
            dl = self.dataloaders[name]
            if hasattr(dl, 'sampler') and hasattr(dl.sampler, 'set_epoch'):
                dl.sampler.set_epoch(self._epochs[name])
            self._iterators[name] = iter(dl)
            return next(self._iterators[name])

    def sample(self, sp_group=None):
        """
        Randomly select a dataloader based on probabilities and return a batch.
        
        Args:
            sp_group: Optional SP process group. When provided, only SP rank 0
                      samples the loader name (advancing the Python ``random``
                      state) and broadcasts it.  All SP ranks then call
                      ``_get_batch`` with the *same* name so that every rank's
                      iterators advance in lock-step.  The actual batch data
                      from non-rank-0 processes is later overwritten by
                      ``sync_tensor_for_sp`` in ``prepare_batch``.
        
        Returns:
            (loader_name, batch): A tuple of the selected dataloader name and the batch dict.
        """
        if sp_group is not None:
            sp_rank = dist.get_rank(sp_group)
            if sp_rank == 0:
                name = random.choices(self.names, weights=self.probs, k=1)[0]
            else:
                name = None
            name_list = [name]
            dist.broadcast_object_list(name_list, src=dist.get_global_rank(sp_group, 0), group=sp_group)
            name = name_list[0]
        else:
            name = random.choices(self.names, weights=self.probs, k=1)[0]

        batch = self._get_batch(name)
        return name, batch


class HunyuanVideoTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.is_main_process = self.rank == 0
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.is_main_process = True
        
        if config.sp_size > self.world_size:
            raise ValueError(
                f"sp_size ({config.sp_size}) cannot be greater than world_size ({self.world_size})"
            )
        if self.world_size % config.sp_size != 0:
            raise ValueError(
                f"sp_size ({config.sp_size}) must evenly divide world_size ({self.world_size}). "
                f"world_size % sp_size = {self.world_size % config.sp_size}"
            )

        # Set CUDA device before any distributed/device-mesh initialization to
        # avoid collectives binding to the wrong GPU.
        torch.cuda.set_device(self.local_rank)
        initialize_parallel_state(sp=config.sp_size, dp_replicate=config.dp_replicate)
        self.parallel_state = get_parallel_state()
        self.dp_rank = self.parallel_state.world_mesh['dp'].get_local_rank()
        self.dp_size = self.parallel_state.world_mesh['dp'].size()
        self.sp_enabled = self.parallel_state.sp_enabled
        self.sp_group = self.parallel_state.sp_group if self.sp_enabled else None

        self._set_seed(config.seed + self.dp_rank)
        self._build_models()
        self._build_optimizer()
        
        self.noise_schedule = LinearInterpolationSchedule(T=config.num_train_timesteps)
        self.timestep_sampler = TimestepSampler(
            T=config.num_train_timesteps, 
            device=self.device,
            snr_type=config.snr_type,
        )
        
        self.global_step = 0
        self.current_epoch = 0
        
        if self.is_main_process:
            os.makedirs(config.output_dir, exist_ok=True)
        
        self.validation_output_dir = os.path.join(config.output_dir, "samples")
        if self.is_main_process:
            os.makedirs(self.validation_output_dir, exist_ok=True)
        
        if config.validation_prompts is None:
            config.validation_prompts = ["A beautiful sunset over the ocean with waves gently crashing on the shore"]
    
    def _set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _build_models(self):
        if self.config.dtype == "bf16":
            transformer_dtype = torch.bfloat16
        elif self.config.dtype == "fp32":
            transformer_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {self.config.dtype}")
        
        # Don't create SR pipeline for training (validation uses enable_sr=False)
        self.pipeline = HunyuanVideo_1_5_Pipeline.create_pipeline(
            pretrained_model_name_or_path=self.config.pretrained_model_root,
            transformer_dtype=transformer_dtype,
            enable_offloading=False,
            enable_group_offloading=False,
            overlap_group_offloading=False,
            create_sr_pipeline=False,
            flow_shift=self.config.validation_timestep_shift,
            device=self.device,
        )
        
        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.text_encoder_2 = self.pipeline.text_encoder_2
        self.vision_encoder = self.pipeline.vision_encoder
        self.byt5_kwargs = {
            "byt5_model": self.pipeline.byt5_model,
            "byt5_tokenizer": self.pipeline.byt5_tokenizer,
        }
        
        self.transformer.train()

        # Freeze non-trainable models to prevent gradient computation and save memory
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.eval()
            self.text_encoder_2.requires_grad_(False)
        if self.vision_encoder is not None and isinstance(self.vision_encoder, nn.Module):
            self.vision_encoder.eval()
            self.vision_encoder.requires_grad_(False)
        if self.byt5_kwargs["byt5_model"] is not None:
            self.byt5_kwargs["byt5_model"].eval()
            self.byt5_kwargs["byt5_model"].requires_grad_(False)

        if self.config.use_lora:
            self._apply_lora()
        
        if self.config.enable_gradient_checkpointing:
            self._apply_gradient_checkpointing()
        
        if self.config.enable_fsdp and self.world_size > 1:
            self._apply_fsdp()
        
        if self.is_main_process:
            logger.info(f"Models loaded. Transformer dtype: {transformer_dtype}")
            total_params = sum(p.numel() for p in self.transformer.parameters())
            trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            logger.info(f"Transformer parameters: {total_params:,} (trainable: {trainable_params:,})")
            logger.info(f"LoRA enabled: {self.config.use_lora}")
            logger.info(f"FSDP enabled: {self.config.enable_fsdp and self.world_size > 1}")
            logger.info(f"Gradient checkpointing enabled: {self.config.enable_gradient_checkpointing}")
            logger.info(f"Timestep sampling strategy: {self.config.snr_type.value}")
    
    def _apply_lora(self):
        if self.is_main_process:
            logger.info("Applying LoRA to transformer using PeftAdapterMixin...")
        
        if self.config.pretrained_lora_path is not None:
            if self.is_main_process:
                logger.info(f"Loading pretrained LoRA from {self.config.pretrained_lora_path}")
            self.load_pretrained_lora(self.config.pretrained_lora_path)
        else:
            from peft import LoraConfig
            
            if self.config.lora_target_modules is None:
                target_modules = "all-linear"
            else:
                target_modules = self.config.lora_target_modules
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            
            self.transformer.add_adapter(lora_config, adapter_name="default")

        
        if self.is_main_process:
            trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.transformer.parameters())
            logger.info(f"LoRA applied successfully. Trainable parameters: {trainable_params:,} / {total_params:,} "
                       f"({100 * trainable_params / total_params:.2f}%)")
    
    def _apply_fsdp(self):
        if self.is_main_process:
            logger.info("Applying FSDP2 to transformer...")
        
        param_dtype = torch.bfloat16
        reduce_dtype = torch.float32  # Reduce in float32 for stability

        self.transformer = self.transformer.to(dtype=param_dtype)
        
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        )
        
        fsdp_config = {"mp_policy": mp_policy}
        if self.world_size > 1:
            try:
                fsdp_config["mesh"] = get_parallel_state().fsdp_mesh
            except Exception as e:
                if self.is_main_process:
                    logger.warning(f"Could not create DeviceMesh: {e}. FSDP will use process group instead.")
        
        for block in list(self.transformer.double_blocks) + list(self.transformer.single_blocks):
            if block is not None:
                fully_shard(block, **fsdp_config)
        
        fully_shard(self.transformer, **fsdp_config)
        
        if self.is_main_process:
            logger.info("FSDP2 applied successfully")
    
    def _apply_gradient_checkpointing(self):
        if self.is_main_process:
            logger.info("Applying gradient checkpointing to transformer blocks...")
        
        no_split_module_type = None
        for block in self.transformer.double_blocks:
            if block is not None:
                no_split_module_type = type(block)
                break
        
        if no_split_module_type is None:
            for block in self.transformer.single_blocks:
                if block is not None:
                    no_split_module_type = type(block)
                    break
        
        if no_split_module_type is None:
            logger.warning("Could not find block type for gradient checkpointing. Using fallback.")
            if hasattr(self.transformer, "gradient_checkpointing_enable"):
                self.transformer.gradient_checkpointing_enable()
            return
        
        def non_reentrant_wrapper(module):
            return checkpoint_wrapper(
                module,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
        
        def selective_checkpointing(submodule):
            return isinstance(submodule, no_split_module_type)
        
        apply_activation_checkpointing(
            self.transformer,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=selective_checkpointing,
        )
        
        if self.is_main_process:
            logger.info("Gradient checkpointing applied successfully")
    
    def _build_optimizer(self):
        if self.config.use_muon:
            self.optimizer = get_muon_optimizer(
                model=self.transformer,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.config.weight_decay,
            )
        
        ga = self.config.gradient_accumulation_steps
        self.lr_scheduler = get_scheduler(
            "constant",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps,
        )
        
        if self.is_main_process:
            logger.info(f"Optimizer and scheduler initialized")
    
    def encode_text(self, prompts, data_type: str = "video"):
        text_inputs = self.text_encoder.text2tokens(prompts, data_type=data_type)
        text_outputs = self.text_encoder.encode(text_inputs, data_type=data_type, device=self.device)
        text_emb = text_outputs.hidden_state
        text_mask = text_outputs.attention_mask
        
        text_emb_2 = None
        text_mask_2 = None
        if self.text_encoder_2 is not None:
            text_inputs_2 = self.text_encoder_2.text2tokens(prompts)
            text_outputs_2 = self.text_encoder_2.encode(text_inputs_2, device=self.device)
            text_emb_2 = text_outputs_2.hidden_state
            text_mask_2 = text_outputs_2.attention_mask
        
        return text_emb, text_mask, text_emb_2, text_mask_2
    
    def encode_byt5(self, text_ids: torch.Tensor, attention_mask: torch.Tensor):
        if self.byt5_kwargs["byt5_model"] is None:
            return None, None
        with torch.no_grad():
            byt5_outputs = self.byt5_kwargs["byt5_model"](text_ids, attention_mask=attention_mask.float())
        byt5_emb = byt5_outputs[0]
        return byt5_emb, attention_mask
    
    def encode_images(self, images, target_height=None, target_width=None):
        """Encode images to vision states (for i2v).

        When *target_height* / *target_width* are given, every image is
        ``resize_and_center_crop``-ed to that size before encoding, matching
        the inference pipeline's ``_prepare_vision_states``.
        """
        if self.vision_encoder is None:
            return None
        assert images.max() <= 1.0 and images.min() >= -1.0, f"Images must be in the range [-1, 1], but got {images.min()} {images.max()}"
        images = (images + 1) / 2 # [-1, 1] -> [0, 1]
        images_np = (images.cpu().permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype("uint8")
        if target_height is not None and target_width is not None:
            images_np = np.stack([
                resize_and_center_crop(img, target_width, target_height)
                for img in images_np
            ], axis=0)
        with torch.no_grad():
            vision_states = self.vision_encoder.encode_images(images_np)
        return vision_states.last_hidden_state.to(device=self.device, dtype=self.transformer.dtype)
    
    def encode_pil_to_vision_states(self, pil_images_per_sample,
                                     target_height=None, target_width=None):
        """
        Encode PIL images to vision states using the vision encoder.
        
        For each batch sample the condition images are encoded independently and
        then **averaged** across images (following ``prepare_model_inputs_subject_driven``
        in helpers.py).
        
        When *target_height* / *target_width* are given, every image is
        ``resize_and_center_crop``-ed to that size before encoding, matching
        the inference pipeline's ``_prepare_vision_states``.
        
        Args:
            pil_images_per_sample: List of lists of PIL images, one list per batch
                sample.  A flat list ``[img1, img2, ...]`` is also accepted and
                will be treated as a single-sample batch.
            target_height: Optional pixel-space height for resize_and_center_crop.
            target_width: Optional pixel-space width for resize_and_center_crop.
        
        Returns:
            vision_states: ``[B, tokens, dim]`` or ``None`` if vision_encoder
            is not available.
        """
        if self.vision_encoder is None:
            return None
        
        # Normalise to list-of-lists
        for cur_img_sampel in pil_images_per_sample:
            assert isinstance(cur_img_sampel, list), f"Each sample in pil_images_per_sample must be a list, but got {type(cur_img_sampel)}"
        
        vision_states_list = []
        for imgs in pil_images_per_sample:
            imgs_np = np.stack([np.array(img, dtype=np.uint8) for img in imgs])  # [N, H, W, C]
            if target_height is not None and target_width is not None:
                imgs_np = np.stack([
                    resize_and_center_crop(img, target_width, target_height)
                    for img in imgs_np
                ], axis=0)
            with torch.no_grad():
                encoder_output = self.vision_encoder.encode_images(imgs_np)
                current_vs = encoder_output.last_hidden_state  # [N, tokens, dim]
                current_vs = torch.mean(current_vs, dim=0)  # [tokens, dim]
            vision_states_list.append(current_vs)
        
        vision_states = torch.stack(vision_states_list, dim=0)  # [B, tokens, dim]
        return vision_states.to(device=self.device, dtype=self.transformer.dtype)
    
    @staticmethod
    def _get_semantic_images_np(input_video_paths, nframes=8):
        """
        Read input/condition videos and extract first frames + uniformly sampled
        frames, following ``get_semantic_images_np`` in multitask_utils.py.

        Args:
            input_video_paths: List of video file paths (one per sample in the
                batch).
            nframes: Number of frames to uniformly sample from each video.

        Returns:
            first_images_np: ``np.ndarray`` of shape ``(B, H, W, C)`` uint8
                containing the first frame of each video, or ``None``.
            sampled_frames: ``List[List[PIL.Image.Image]]`` — for each video a
                list of ``nframes`` PIL images uniformly sampled from the clip.
        """
        from decord import VideoReader as _VideoReader

        if input_video_paths is None or len(input_video_paths) == 0:
            return None, None

        first_frames = []
        sampled_frames = []
        for video_path in input_video_paths:
            if isinstance(video_path, str):
                vr = _VideoReader(video_path)
            else:
                vr = video_path
            if len(vr) == 0:
                raise ValueError(f"Video {video_path} has no frames")

            # Uniformly sample nframes (same logic as prepare_custom_video)
            max_frames = min(len(vr), 161)
            sample_indices = np.linspace(0, max_frames - 1, nframes).astype(int)
            frames = vr.get_batch(sample_indices)
            if hasattr(frames, 'asnumpy'):
                frames = frames.asnumpy()
            else:
                frames = frames.numpy()
            sampled_frames.append([Image.fromarray(f) for f in frames])

            # First frame
            first_frame = vr.get_batch([0])
            if hasattr(first_frame, 'asnumpy'):
                first_frame_np = first_frame.asnumpy()
            else:
                first_frame_np = first_frame.numpy()
            first_frames.append(first_frame_np[0])
            del vr

        first_images_np = np.stack(first_frames, axis=0).astype(np.uint8)
        return first_images_np, sampled_frames

    @staticmethod
    def _resize_and_center_crop_pil(
        frame: Image.Image, target_width: int, target_height: int,
    ) -> Image.Image:
        """Resize a PIL image so it covers the target size, then center-crop.

        Matches the ``_resize_and_center_crop`` closure used in
        ``get_task_specific_input`` of the inference pipeline.
        """
        original_width, original_height = frame.size
        scale_factor = max(
            target_width / original_width, target_height / original_height,
        )
        resize_width = int(round(original_width * scale_factor))
        resize_height = int(round(original_height * scale_factor))
        resize_transform = T.Compose([
            T.Resize(
                (resize_height, resize_width),
                interpolation=T.InterpolationMode.LANCZOS,
            ),
            T.CenterCrop((target_height, target_width)),
        ])
        return resize_transform(frame)

    def _pixel_values_to_first_frame_pil(self, pixel_values: torch.Tensor):
        """
        Convert the first frame of ``pixel_values`` tensor to a list of PIL images.
        
        Used to feed the target-video first frame to the Qwen-VL text encoder for
        i2v tasks (prompt_mode=2), following helpers.py ``get_cond_latents → semantic_images_pil``.
        
        Args:
            pixel_values: Tensor of shape ``[B, C, F, H, W]`` or ``[B, C, H, W]``
                          in range [-1, 1].
        
        Returns:
            List of ``B`` PIL images, or ``None`` if pixel_values is None.
        """
        if pixel_values is None:
            return None
        if pixel_values.ndim == 5:
            first_frames = pixel_values[:, :, 0, :, :]  # [B, C, H, W]
        else:
            first_frames = pixel_values  # [B, C, H, W]
        # [-1, 1] → [0, 1] → [0, 255] uint8
        first_frames = (first_frames / 2 + 0.5).clamp(0, 1)
        first_frames_np = (
            first_frames.cpu().permute(0, 2, 3, 1).numpy() * 255
        ).round().astype(np.uint8)
        return [Image.fromarray(frame) for frame in first_frames_np]

    def _decode_first_frame_from_latents(self, latents: torch.Tensor):
        """
        Decode the first frame from latents back to pixel space via VAE,
        following ``get_cond_latents`` in multitask_utils.py.

        Used when ``pixel_values`` is not available (latent-cache-only training)
        and PIL images / numpy arrays are needed for the text encoder and vision
        encoder in i2v tasks.

        Args:
            latents: Tensor ``[B, C, F, H, W]`` already scaled by
                     ``vae.config.scaling_factor``.

        Returns:
            first_images_pil: List of ``B`` PIL images.
            first_images_np:  ``np.ndarray`` of shape ``[B, H, W, C]`` uint8.
        """
        first_image_latents = latents[:, :, 0, ...] if len(latents.shape) == 5 else latents
        if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
            first_image_latents = 1 / self.vae.config.scaling_factor * first_image_latents + self.vae.config.shift_factor
        else:
            first_image_latents = 1 / self.vae.config.scaling_factor * first_image_latents

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16), \
                self.vae.memory_efficient_context():
            first_images = self.vae.decode(
                first_image_latents.unsqueeze(2).to(self.vae.dtype),
                return_dict=False,
            )[0]

        first_images = first_images.squeeze(2)
        first_images = (first_images / 2 + 0.5).clamp(0, 1)
        first_images = first_images.cpu().permute(0, 2, 3, 1).float().numpy()
        first_images_pil = [Image.fromarray((img * 255).round().astype(np.uint8)) for img in first_images]
        first_images_np = (first_images * 255).round().astype("uint8")
        return first_images_pil, first_images_np

    def encode_text_for_task(
        self,
        prompts: List[str],
        task_type: str,
        batch: Dict[str, Any],
        pixel_values: Optional[torch.Tensor] = None,
    ):
        """
        Encode text using task-specific prompt modes via ``text_encoder.prepare_input``.
        
        Follows the patterns in helpers.py where different tasks feed different
        visual context to the Qwen-VL text encoder:
        
        +-----------------------+-------------+--------------------------------------+
        | task_type             | prompt_mode | visual inputs                        |
        +-----------------------+-------------+--------------------------------------+
        | t2v                   | 1           | none (text only)                     |
        | i2v                   | 2           | first frame of target video (PIL)    |
        | key_frames_to_v       | 4           | condition images (list of lists PIL) |
        | multi_imgs_to_v       | 3           | condition images (list of lists PIL) |
        | v2v                   | 5           | videos=sampled_frames (input video)  |
        | tiv2v                 | 6           | all imgs + videos=sampled_frames     |
        +-----------------------+-------------+--------------------------------------+
        
        ``text_encoder_2`` (CLIP) is always text-only.
        
        Returns:
            text_emb, text_mask, text_emb_2, text_mask_2, deepstack_hidden_states
        """
        data_type = "video"
        deepstack = self.config.deepstack
        setclip = self.config.setclip
        
        if task_type == "t2v":
            # Text only — prompt_mode=1
            after_process_inputs, crop_start = self.text_encoder.prepare_input(
                prompts, prompt_mode=1,
            )
        
        elif task_type == "i2v":
            # First frame of target video — prompt_mode=2
            # When pixel_values is available, extract first frame directly;
            # otherwise fall back to VAE-decoded first frame from latents
            # (populated by prepare_batch → _decode_first_frame_from_latents).
            semantic_images_pil = self._pixel_values_to_first_frame_pil(pixel_values)
            if semantic_images_pil is None:
                semantic_images_pil = batch.get("_i2v_first_frame_pil", None)
            if semantic_images_pil is not None:
                newimgs = []
                for p in semantic_images_pil:
                    p2 = p.copy()
                    p2.thumbnail((560, 560))
                    newimgs.append(p2)
                after_process_inputs, crop_start = self.text_encoder.prepare_input(
                    prompts, newimgs, prompt_mode=2,
                )
            else:
                after_process_inputs, crop_start = self.text_encoder.prepare_input(
                    prompts, prompt_mode=1,
                )
        
        elif task_type == "key_frames_to_v":
            # Interpolation — prompt_mode=4 (matches inference interpolation task)
            condition_images = batch.get("condition", None)
            if condition_images is not None:
                newimgs = []
                for ps in condition_images:
                    if isinstance(ps, PIL.Image.Image):
                        ps = [ps]
                    subimgs = []
                    for p in ps:
                        p2 = p.copy()
                        p2.thumbnail((560, 560))
                        subimgs.append(p2)
                    newimgs.append(subimgs)
                after_process_inputs, crop_start = self.text_encoder.prepare_input(
                    prompts, newimgs, prompt_mode=4,
                )
            else:
                after_process_inputs, crop_start = self.text_encoder.prepare_input(
                    prompts, prompt_mode=1,
                )
        
        elif task_type == "multi_imgs_to_v":
            # Reference2v / subject-driven — prompt_mode=3 (matches inference reference2v task)
            condition_images = batch.get("condition", None)
            if condition_images is not None:
                newimgs = []
                for ps in condition_images:
                    if isinstance(ps, PIL.Image.Image):
                        ps = [ps]
                    subimgs = []
                    for p in ps:
                        p2 = p.copy()
                        p2.thumbnail((560, 560))
                        subimgs.append(p2)
                    newimgs.append(subimgs)
                after_process_inputs, crop_start = self.text_encoder.prepare_input(
                    prompts, newimgs, prompt_mode=3,
                )
            else:
                after_process_inputs, crop_start = self.text_encoder.prepare_input(
                    prompts, prompt_mode=1,
                )
        
        elif task_type == "v2v":
            # Video editing: sampled frames from INPUT video — prompt_mode=5
            sampled_frames = batch.get("sampled_frames", None)
            if sampled_frames is not None:
                nframes = len(sampled_frames[0])
                after_process_inputs, crop_start = self.text_encoder.prepare_input(
                    prompts, videos=sampled_frames,
                    num_frames=nframes, prompt_mode=5,
                )
            else:
                after_process_inputs, crop_start = self.text_encoder.prepare_input(
                    prompts, prompt_mode=1,
                )
        
        elif task_type == "tiv2v":
            # Text+image V2V: condition images + sampled frames — prompt_mode=6
            # Inference passes ALL condition images (flat list) to the text
            # encoder, so training must do the same for consistency.
            sampled_frames = batch.get("sampled_frames", None)
            input_imgs = batch.get("input_img", None)
            if sampled_frames is not None and input_imgs is not None:
                newimgs = []
                for ps in input_imgs:
                    if isinstance(ps, PIL.Image.Image):
                        ps = [ps]
                    subimgs = []
                    for p in ps:
                        p2 = p.copy()
                        p2.thumbnail((560, 560))
                        subimgs.append(p2)
                    newimgs.append(subimgs)
                nframes = len(sampled_frames[0])
                after_process_inputs, crop_start = self.text_encoder.prepare_input(
                    prompts, imgs=newimgs, videos=sampled_frames,
                    num_frames=nframes, prompt_mode=6,
                )
            else:
                after_process_inputs, crop_start = self.text_encoder.prepare_input(
                    prompts, prompt_mode=1,
                )
        
        else:
            after_process_inputs, crop_start = self.text_encoder.prepare_input(
                prompts, prompt_mode=1,
            )
        
        after_process_inputs = after_process_inputs.to(self.device)
        with torch.no_grad():
            text_outputs = self.text_encoder.encode(
                after_process_inputs,
                data_type=data_type,
                device=self.device,
                crop_start=crop_start,
                deepstack=deepstack,
                setclip=setclip,
            )
        text_emb = text_outputs.hidden_state
        text_mask = text_outputs.attention_mask
        deepstack_hidden_states = text_outputs.deepstack_hidden_states
        
        # ---- text_encoder_2 (CLIP) — always text-only ----
        text_emb_2 = None
        text_mask_2 = None
        
        return text_emb, text_mask, text_emb_2, text_mask_2, deepstack_hidden_states
    
    def encode_vae(self, images: torch.Tensor) -> torch.Tensor:
        if images.max() > 1.0 or images.min() < -1.0:
            raise ValueError(f"Images must be in the range [-1, 1], but got {images.min()} {images.max()}")
        
        if images.ndim == 4:
            images = images.unsqueeze(2)
        
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16), self.vae.memory_efficient_context():
            latents = self.vae.encode(images).latent_dist.sample()
            if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
                latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            else:
                latents = latents * self.vae.config.scaling_factor
        
        return latents
    
    def encode_pil_images_to_latents(self, pil_images, target_size=None):
        """
        Encode a list of PIL images to VAE latents.
        
        Each image is optionally resized (aspect-ratio-preserving) and center-cropped
        to ``target_size`` before encoding, following the pattern in
        ``video_loader_2.py``.
        
        Args:
            pil_images: List of PIL.Image.Image
            target_size: Optional tuple ``(height, width)`` in **pixel** space.
                         If provided, every image is first resized so that both
                         dimensions are >= the target, then center-cropped to
                         exactly ``(height, width)``.
        
        Returns:
            latents: torch.Tensor of shape [N, C, 1, H, W]
        """
        # ---- Resize + CenterCrop preprocessing (ref: video_loader_2.py) ----
        if target_size is not None:
            target_h, target_w = target_size
            processed = []
            for img in pil_images:
                h, w = img.height, img.width
                # Compute resize dims that guarantee both sides >= target
                if target_h / h > target_w / w:
                    resize_size = (target_h, int(w * target_h / h))
                else:
                    resize_size = (int(h * target_w / w), target_w)
                img = T.Resize(resize_size, antialias=True)(img)
                img = T.CenterCrop((target_h, target_w))(img)
                processed.append(img)
            pil_images = processed

        image_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        pixel_values = torch.stack([image_transform(img) for img in pil_images])  # [N, C, H, W]
        pixel_values = pixel_values.unsqueeze(2).to(self.device)  # [N, C, 1, H, W]
        
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16), self.vae.memory_efficient_context():
            latents = self.vae.encode(pixel_values).latent_dist.mode()
            if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
                latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            else:
                latents = latents * self.vae.config.scaling_factor
        
        return latents  # [N, C, 1, H, W]
    
    def get_condition(self, latents: torch.Tensor, task_type: str, batch: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Build condition latents and mask following the pipeline's ``_prepare_cond_latents``
        conventions.
        
        The returned tensor always has shape [B, C+1, F, H, W], where the last channel
        is the binary mask indicating which frames carry valid condition values.
        For tiv2v, video and image condition latents are **summed** (not concatenated)
        and the two masks are also summed, consistent with the pipeline.
        
        Mask patterns (from multitask_utils.get_mask_from_mask_type):
            t2v              : mask = zeros(F)
            i2v              : mask = zeros(F);  mask[0] = 1
            key_frames_to_v  : mask = zeros(F);  mask[frame_id[i]] = 1 for each i
            multi_imgs_to_v  : mask = zeros(F);  mask[1:num_imgs+1] = 1   (reference2v)
            v2v              : mask = ones(F)                              (editing)
            tiv2v            : mask = ones(F) + mask2;  mask2 = zeros(F), mask2[1] = 1
                               latents = video_latents + img_latents (summed)
        """
        b, c, f, h, w = latents.shape
        
        # Pixel-space target size for PIL image preprocessing (resize + center crop)
        vae_spatial_ratio = getattr(self.pipeline, 'vae_spatial_compression_ratio', 16)
        pixel_target_size = (h * vae_spatial_ratio, w * vae_spatial_ratio)
        
        if task_type == "t2v":
            # 文生视频: mask = zeros(F) — no condition
            cond = torch.zeros([b, c + 1, f, h, w], device=latents.device, dtype=latents.dtype)
            return cond
        
        elif task_type == "i2v":
            # 图生视频: mask[0] = 1, first frame from target latents as condition
            cond = torch.zeros([b, c + 1, f, h, w], device=latents.device, dtype=latents.dtype)
            cond[:, :-1, :1] = latents[:, :, :1]
            cond[:, -1, 0] = 1
            return cond
        
        elif task_type == "key_frames_to_v":
            # Key-frame conditioned generation: each condition image is placed at
            # an explicit frame index given by batch["frame_id"].
            # mask[frame_id[i]] = 1 for each condition image i.
            cond = torch.zeros([b, c + 1, f, h, w], device=latents.device, dtype=latents.dtype)
            assert batch is not None and "condition" in batch and "frame_id" in batch, \
                "`batch` with 'condition' (list of PIL images) and 'frame_id' (list of ints) " \
                "must be provided for key_frames_to_v"
            
            condition_images = batch["condition"]
            frame_ids = batch["frame_id"]
            
            # Ensure every sample in the batch is a list
            for bi in range(b):
                assert isinstance(condition_images[bi], list), (
                    f"key_frames_to_v: condition_images[{bi}] must be a list, "
                    f"got {type(condition_images[bi])}"
                )
                assert isinstance(frame_ids[bi], list), (
                    f"key_frames_to_v: frame_ids[{bi}] must be a list, "
                    f"got {type(frame_ids[bi])}"
                )
                assert len(condition_images[bi]) == len(frame_ids[bi]), (
                    f"key_frames_to_v: number of condition images ({len(condition_images[bi])}) must match "
                    f"number of frame_ids ({len(frame_ids[bi])}) for batch index {bi}"
                )
            
            for bi in range(b):
                imgs = condition_images[bi]
                fids = frame_ids[bi]
                assert len(imgs) == len(fids), (
                    f"key_frames_to_v: number of condition images ({len(imgs)}) must match "
                    f"number of frame_ids ({len(fids)}) for batch index {bi}"
                )
                encoded = self.encode_pil_images_to_latents(imgs, target_size=pixel_target_size)  # [N, C, 1, H, W]
                encoded = encoded.squeeze(2)  # [N, C, H, W]
                for i, fid in enumerate(fids):
                    assert 0 <= fid < f, (
                        f"key_frames_to_v: frame_id {fid} out of range [0, {f}) for batch index {bi}"
                    )
                    cond[bi, :-1, fid] = encoded[i].to(dtype=latents.dtype)
                    cond[bi, -1, fid] = 1
            return cond
        
        elif task_type == "multi_imgs_to_v":
            # reference2v: condition images placed sequentially at frames 1..num_imgs
            # mask[1:num_imgs+1] = 1
            cond = torch.zeros([b, c + 1, f, h, w], device=latents.device, dtype=latents.dtype)
            assert batch is not None and "condition" in batch, \
                "`batch` with 'condition' (list of PIL images) must be provided for multi_imgs_to_v"
            
            condition_images = batch["condition"]
            # Normalize to list-of-lists: each sample should be a list of PIL images.
            # After collation, condition_images is already [sample0_list, sample1_list, ...]
            # but guard against a flat list (single-sample, no collation).
            if len(condition_images) == b:
                for ci in range(b):
                    if not isinstance(condition_images[ci], list):
                        condition_images[ci] = [condition_images[ci]]
            else:
                condition_images = [condition_images]
            
            for bi in range(b):
                imgs = condition_images[bi]
                num_imgs = len(imgs)
                encoded = self.encode_pil_images_to_latents(imgs, target_size=pixel_target_size)  # [num_imgs, C, 1, H, W]
                encoded = encoded.squeeze(2)  # [num_imgs, C, H, W]
                cond[bi, :-1, 1:num_imgs + 1] = encoded.permute(1, 0, 2, 3).to(dtype=latents.dtype)
                cond[bi, -1, 1:num_imgs + 1] = 1
            return cond
        
        elif task_type == "v2v":
            # editing: mask = ones(F), all frames conditioned by input video
            cond = torch.zeros([b, c + 1, f, h, w], device=latents.device, dtype=latents.dtype)
            assert batch is not None and "input_video_latents" in batch, \
                "`batch` with 'input_video_latents' must be provided for v2v"
            
            input_video_latents = batch["input_video_latents"].to(device=latents.device, dtype=latents.dtype)
            assert input_video_latents.shape == latents.shape, (
                f"v2v: input_video_latents shape {input_video_latents.shape} "
                f"must match target latents shape {latents.shape}"
            )
            cond[:, :-1, :] = input_video_latents
            cond[:, -1, :] = 1
            return cond
        
        elif task_type == "tiv2v":
            # Following the pipeline: video and image condition latents are SUMMED
            # into a single [B, C, F, H, W] tensor, and masks are also summed into
            # a single [B, 1, F, H, W] tensor. Final shape: [B, C+1, F, H, W].
            cond = torch.zeros([b, c + 1, f, h, w], device=latents.device, dtype=latents.dtype)
            assert batch is not None and "input_video_latents" in batch and "input_img" in batch, \
                "`batch` with 'input_video_latents' and 'input_img' must be provided for tiv2v"
            
            input_video_latents = batch["input_video_latents"].to(device=latents.device, dtype=latents.dtype)
            assert input_video_latents.shape == latents.shape, (
                f"tiv2v: input_video_latents shape {input_video_latents.shape} "
                f"must match target latents shape {latents.shape}"
            )
            
            # Sum video latents into condition
            cond[:, :-1, :] = input_video_latents
            # mask1 = ones(F)
            cond[:, -1, :] = 1
            
            # Encode condition image and SUM into the same latent channels
            input_imgs = batch["input_img"]
            # Normalize to list-of-lists: each sample should be a list of PIL images.
            if len(input_imgs) == b:
                for ci in range(b):
                    if not isinstance(input_imgs[ci], list):
                        input_imgs[ci] = [input_imgs[ci]]
            else:
                input_imgs = [input_imgs]
            
            for bi in range(b):
                imgs = input_imgs[bi]
                num_imgs = len(imgs)
                encoded = self.encode_pil_images_to_latents(imgs, target_size=pixel_target_size)  # [num_imgs, C, 1, H, W]
                encoded = encoded.squeeze(2)  # [num_imgs, C, H, W]
                cond[bi, :-1, 1:num_imgs + 1] += encoded.permute(1, 0, 2, 3).to(dtype=latents.dtype)
                cond[bi, -1, 1:num_imgs + 1] += 1
            return cond
        
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    # Multi-task only: ``loader_name`` from ``MultiDataloader.sample()`` -> ``task_type``
    # for ``prepare_batch`` / ``encode_text_for_task``. Keep in sync with ``--dataloader_probs``.
    LOADER_TASK_MAP = {
        "t2v": "t2v",                    # text-to-video
        "i2v": "i2v",                    # image-to-video
        "key_frames_to_v": "key_frames_to_v",  # key frames to video
        "multi_imgs_to_v": "multi_imgs_to_v",  # multiple images to video
        "v2v": "v2v",                    # video-to-video editing
        "tiv2v": "tiv2v",               # text+image conditioned video-to-video
    }

    def sample_task(self, loader_name: Optional[str] = None) -> str:
        """
        Determine task type for the current training step.

        When ``loader_name`` is provided (multi-task training), the task type is
        looked up from ``LOADER_TASK_MAP`` — the loader name directly determines
        the task.

        When ``loader_name`` is None (single-dataloader training), uses
        ``config.task_type``.
        """
        if loader_name is not None:
            task = self.LOADER_TASK_MAP.get(loader_name, None)
            if task is None:
                raise ValueError(
                    f"Unknown loader_name '{loader_name}'. "
                    f"Please add it to LOADER_TASK_MAP. Available: {list(self.LOADER_TASK_MAP.keys())}"
                )
            return task
        return self.config.task_type
    
    def prepare_batch(self, batch: Dict[str, Any], loader_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare batch for training. Data is always video.
        
        Args:
            batch: Batch dict from the dataloader.
            loader_name: Name of the dataloader that produced this batch (used in
                         multi-task training to determine task_type).
        
        Expected batch format:
        {
            "pixel_values": torch.Tensor, # [B, C, F, H, W]
                                          # Pixel values must be in range [-1, 1] 
            "text": List[str],
            "byt5_text_ids": Optional[torch.Tensor],
            "byt5_text_mask": Optional[torch.Tensor],
        }
        
        Note: The temporal dimension F must be 4n+1 (e.g., 1, 5, 9, 13, 17, ...)
        to satisfy VAE requirements. The dataset should ensure this before returning data.
        
        """
        pixel_values = batch.get("pixel_values", None)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        if 'latents' in batch:
            latents = batch['latents'].to(self.device)
        else:
            latents = self.encode_vae(pixel_values)
        
        if self.sp_enabled:
            latents = sync_tensor_for_sp(latents, self.sp_group)
            has_pixel_values = sync_tensor_for_sp(pixel_values is not None, self.sp_group)
            if has_pixel_values:
                if pixel_values is None:
                    pixel_values = torch.zeros(1, device=self.device)
                pixel_values = sync_tensor_for_sp(pixel_values, self.sp_group)
            else:
                pixel_values = None
        
        task_type = self.sample_task(loader_name=loader_name)

        if self.sp_enabled:
            task_type = sync_tensor_for_sp(task_type, self.sp_group)
        
        # ---- For v2v / tiv2v: read input video to get sampled_frames & first-
        # frame numpy, following get_semantic_images_np in multitask_utils.py.
        # This replaces the previous approach of passing sampled_frames from the
        # dataloader. ----------------------------------------------------------
        _is_sp_rank0 = (not self.sp_enabled) or (dist.get_rank(self.sp_group) == 0)
        semantic_images_np = None
        sampled_frames = None
        if task_type in ("v2v", "tiv2v") and _is_sp_rank0:
            input_video_paths = batch.get("input_video_path", None)
            if input_video_paths is not None:
                # Determine nframes based on latent temporal length
                cond_latents_tmp = batch.get("input_video_latents", None)
                if cond_latents_tmp is not None:
                    num_video_frames = (cond_latents_tmp.shape[2] - 1) * 4 + 1
                else:
                    num_video_frames = latents.shape[2]
                if num_video_frames >= 24 * 4:
                    nframes = 8
                elif num_video_frames >= 24 * 3:
                    nframes = 6
                else:
                    nframes = 4
                semantic_images_np, sampled_frames = self._get_semantic_images_np(
                    input_video_paths, nframes=nframes,
                )

                # Resize sampled_frames to target resolution, matching
                # inference pipeline (get_task_specific_input).
                vae_spatial_ratio = getattr(
                    self.pipeline, 'vae_spatial_compression_ratio', 16,
                )
                target_height = int(latents.shape[-2] * vae_spatial_ratio)
                target_width = int(latents.shape[-1] * vae_spatial_ratio)

                if sampled_frames is not None:
                    resized_frames = []
                    for frame_list in sampled_frames:
                        if frame_list is None:
                            resized_frames.append(frame_list)
                            continue
                        resized_frames.append([
                            self._resize_and_center_crop_pil(
                                frame, target_width, target_height,
                            )
                            for frame in frame_list
                        ])
                    sampled_frames = resized_frames

                if semantic_images_np is not None:
                    semantic_images_np = np.stack([
                        resize_and_center_crop(
                            image, target_width, target_height,
                        )
                        for image in semantic_images_np
                    ], axis=0)

            # Inject into batch so downstream encode_text_for_task / vision
            # encoder blocks can access them transparently.
            batch["sampled_frames"] = sampled_frames
            batch["semantic_images_np"] = semantic_images_np

        if _is_sp_rank0:
            cond_latents = self.get_condition(latents, task_type, batch=batch)
        else:
            b, c, f, h, w = latents.shape
            cond_latents = torch.zeros([b, c + 1, f, h, w], device=latents.device, dtype=latents.dtype)
        if self.sp_enabled:
            cond_latents = sync_tensor_for_sp(cond_latents, self.sp_group)
        
        # When pixel_values is not available (latent-cache-only) and the task
        # is i2v, we need to decode the first frame from latents to provide PIL
        # images for the text encoder (prompt_mode=2) and numpy arrays for the
        # vision encoder (SigLIP), following get_cond_latents in helpers.py.
        if task_type == "i2v" and pixel_values is None:
            i2v_pil, i2v_np = self._decode_first_frame_from_latents(latents)
            batch["_i2v_first_frame_pil"] = i2v_pil
            batch["_i2v_first_frame_np"] = i2v_np

        prompts = batch["text"]
        if self.sp_enabled:
            prompts = sync_tensor_for_sp(prompts, self.sp_group)
        text_emb, text_mask, text_emb_2, text_mask_2, deepstack_hidden_states = self.encode_text_for_task(
            prompts, task_type, batch, pixel_values=pixel_values,
        )
        if self.sp_enabled:
            text_emb = sync_tensor_for_sp(text_emb, self.sp_group)
            text_mask = sync_tensor_for_sp(text_mask, self.sp_group)
            if text_emb_2 is not None:
                text_emb_2 = sync_tensor_for_sp(text_emb_2, self.sp_group)
            if text_mask_2 is not None:
                text_mask_2 = sync_tensor_for_sp(text_mask_2, self.sp_group)
            if deepstack_hidden_states is not None:
                deepstack_hidden_states = sync_tensor_for_sp(deepstack_hidden_states, self.sp_group)
        
        byt5_text_states = None
        byt5_text_mask = None
        if self.byt5_kwargs["byt5_model"] is not None:
            if "byt5_text_ids" in batch and batch["byt5_text_ids"] is not None:
                byt5_text_ids = batch["byt5_text_ids"].to(self.device)
                byt5_text_mask = batch["byt5_text_mask"].to(self.device)
                if self.sp_enabled:
                    byt5_text_ids = sync_tensor_for_sp(byt5_text_ids, self.sp_group)
                    byt5_text_mask = sync_tensor_for_sp(byt5_text_mask, self.sp_group)
                byt5_text_states, byt5_text_mask = self.encode_byt5(byt5_text_ids, byt5_text_mask)
            else:
                byt5_embeddings_list = []
                byt5_mask_list = []
                with torch.no_grad():
                    for prompt in prompts:
                        emb, mask = self.pipeline._process_single_byt5_prompt(prompt, self.device)
                        byt5_embeddings_list.append(emb)
                        byt5_mask_list.append(mask)
                
                byt5_text_states = torch.cat(byt5_embeddings_list, dim=0)
                byt5_text_mask = torch.cat(byt5_mask_list, dim=0)
                if self.sp_enabled:
                    byt5_text_states = sync_tensor_for_sp(byt5_text_states, self.sp_group)
                    byt5_text_mask = sync_tensor_for_sp(byt5_text_mask, self.sp_group)
        
        # ======================== Vision encoder (SigLIP) ========================
        # Ref: helpers.py — different tasks feed different images to the vision
        # encoder to produce ``vision_states`` for the transformer.
        # For t2v the pipeline passes a **zero tensor** (not None) so that the
        # transformer always sees the same input structure; the attention mask
        # for these tokens is set to 0 inside the transformer when
        # ``mask_type == "t2v"`` and ``torch.all(vision_states == 0)``.
        #
        # The inference pipeline's ``_prepare_vision_states`` applies
        # ``resize_and_center_crop`` to every image before encoding.  We
        # replicate that here so that the vision encoder sees the same input
        # distribution at train and test time.
        vision_num_tokens = getattr(self.pipeline.config, 'vision_num_semantic_tokens', 257)
        vision_dim = getattr(self.pipeline.config, 'vision_states_dim', 1152)
        vision_states = torch.zeros(
            latents.shape[0], vision_num_tokens, vision_dim,
            device=self.device, dtype=self.transformer.dtype,
        )

        vae_spatial_ratio = getattr(self.pipeline, 'vae_spatial_compression_ratio', 16)
        vis_target_h = int(latents.shape[-2] * vae_spatial_ratio)
        vis_target_w = int(latents.shape[-1] * vae_spatial_ratio)

        if task_type == "t2v":
            pass  # keep zero tensor
        
        elif task_type == "i2v":
            # i2v: encode first frame of target video
            # (ref: prepare_model_inputs — get_cond_latents → vision_encoder)
            if pixel_values is not None:
                if pixel_values.ndim == 5:
                    first_frame = pixel_values[:, :, 0, :, :]
                else:
                    first_frame = pixel_values
                vision_states = self.encode_images(
                    first_frame,
                    target_height=vis_target_h,
                    target_width=vis_target_w,
                )
            else:
                i2v_np = batch.get("_i2v_first_frame_np", None)
                if self.vision_encoder is not None and i2v_np is not None:
                    i2v_np = np.stack([
                        resize_and_center_crop(img, vis_target_w, vis_target_h)
                        for img in i2v_np
                    ], axis=0)
                    with torch.no_grad():
                        encoder_output = self.vision_encoder.encode_images(i2v_np)
                    vision_states = encoder_output.last_hidden_state.to(
                        device=self.device, dtype=self.transformer.dtype,
                    )
        
        elif task_type in ("key_frames_to_v", "multi_imgs_to_v"):
            # Subject-driven / reference2v: encode condition images via vision
            # encoder, average across images per sample
            # (ref: prepare_model_inputs_subject_driven — get_cond_latents2 →
            #  vision_encoder.encode_images per batch, mean over images)
            if "condition" in batch and batch["condition"] is not None:
                vision_states = self.encode_pil_to_vision_states(
                    batch["condition"],
                    target_height=vis_target_h,
                    target_width=vis_target_w,
                )
        
        elif task_type == "v2v":
            # Video editing: encode first frame of INPUT video (not target!)
            # (ref: prepare_model_inputs_editing — get_semantic_images_np(cond_video_path)
            #  → vision_encoder.encode_images(semantic_images_np))
            _sem_np = batch.get("semantic_images_np", None)
            if self.vision_encoder is not None and _sem_np is not None:
                with torch.no_grad():
                    encoder_output = self.vision_encoder.encode_images(_sem_np)
                vision_states = encoder_output.last_hidden_state.to(
                    device=self.device, dtype=self.transformer.dtype
                )
        
        elif task_type == "tiv2v":
            # Text+image conditioned V2V: average vision states from INPUT video
            # first frame AND condition image (input_img)
            # (ref: prepare_model_inputs_editing —
            #  get_semantic_images_np2(cond_video_path, cond_img) →
            #  vision_states = (vs_video + vs_img) / 2.0)
            vs_video = None
            _sem_np = batch.get("semantic_images_np", None)
            if self.vision_encoder is not None and _sem_np is not None:
                with torch.no_grad():
                    encoder_output = self.vision_encoder.encode_images(_sem_np)
                vs_video = encoder_output.last_hidden_state.to(
                    device=self.device, dtype=self.transformer.dtype
                )
            
            vs_img = None
            if self.vision_encoder is not None and "input_img" in batch and batch["input_img"] is not None:
                vs_img = self.encode_pil_to_vision_states(
                    batch["input_img"],
                    target_height=vis_target_h,
                    target_width=vis_target_w,
                )
            
            if vs_video is not None and vs_img is not None:
                vision_states = (vs_video + vs_img) / 2.0
            elif vs_video is not None:
                vision_states = vs_video
            elif vs_img is not None:
                vision_states = vs_img
        
        noise = torch.randn_like(latents)
        timesteps = self.timestep_sampler.sample(latents.shape[0], device=self.device)
        timesteps = timestep_transform(timesteps, self.config.num_train_timesteps, self.config.train_timestep_shift)
        
        if self.sp_enabled:
            noise = sync_tensor_for_sp(noise, self.sp_group)
            timesteps = sync_tensor_for_sp(timesteps, self.sp_group)
        
        latents_noised = self.noise_schedule.forward(latents, noise, timesteps)
        target = noise - latents
        
        if self.sp_enabled and vision_states is not None:
            vision_states = sync_tensor_for_sp(vision_states, self.sp_group)
        
        return {
            "latents_noised": latents_noised,
            "cond_latents": cond_latents,
            "timesteps": timesteps,
            "target": target,
            "text_emb": text_emb,
            "text_emb_2": text_emb_2,
            "text_mask": text_mask,
            "text_mask_2": text_mask_2,
            "byt5_text_states": byt5_text_states,
            "byt5_text_mask": byt5_text_mask,
            "vision_states": vision_states,
            "task_type": task_type,
            "deepstack_hidden_states": deepstack_hidden_states,
        }
    
    def train_step(self, batch: Dict[str, Any], loader_name: Optional[str] = None) -> Dict[str, float]:
        inputs = self.prepare_batch(batch, loader_name=loader_name)
        latents_input = torch.cat([inputs["latents_noised"], inputs["cond_latents"]], dim=1)
        model_dtype = torch.bfloat16 if self.config.dtype == "bf16" else torch.float32
        
        extra_kwargs = {}
        if inputs["byt5_text_states"] is not None:
            extra_kwargs["byt5_text_states"] = inputs["byt5_text_states"].to(dtype=model_dtype)
            extra_kwargs["byt5_text_mask"] = inputs["byt5_text_mask"]
        
        deepstack_hidden_states = inputs["deepstack_hidden_states"]
        if deepstack_hidden_states is not None:
            deepstack_hidden_states = deepstack_hidden_states.to(dtype=model_dtype)
        
        with torch.autocast(device_type="cuda", dtype=model_dtype, enabled=(model_dtype == torch.bfloat16)):
            model_pred = self.transformer(
                latents_input.to(dtype=model_dtype),
                inputs["timesteps"],
                text_states=inputs["text_emb"].to(dtype=model_dtype),
                text_states_2=inputs["text_emb_2"].to(dtype=model_dtype) if inputs["text_emb_2"] is not None else None,
                encoder_attention_mask=inputs["text_mask"],
                vision_states=inputs["vision_states"].to(dtype=model_dtype) if inputs["vision_states"] is not None else None,
                mask_type=inputs["task_type"],
                extra_kwargs=extra_kwargs if extra_kwargs else None,
                return_dict=False,
                all_stack_text_states=deepstack_hidden_states,
            )[0]
        
        target = inputs["target"].to(dtype=model_pred.dtype)
        loss = nn.functional.mse_loss(model_pred, target)
        
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()
        
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.transformer.parameters(),
                    self.config.max_grad_norm
                )
            else:
                grad_norm = torch.tensor(0.0)
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        else:
            grad_norm = torch.tensor(0.0)
        
        metrics = {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, "get_last_lr") else self.config.learning_rate,
        }
        
        return metrics
    
    def save_checkpoint(self, step: int):
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        transformer_dir = os.path.join(checkpoint_dir, "transformer")
        
        if self.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
        if self.world_size > 1:
            dist.barrier()
        
        if self.config.use_lora and hasattr(self.transformer, "save_lora_adapter"):
            has_adapters = bool(getattr(self.transformer, "peft_config", None))
            if self.world_size > 1:
                has_adapters_tensor = torch.tensor(
                    int(has_adapters), device=self.device, dtype=torch.int32
                )
                dist.all_reduce(has_adapters_tensor, op=dist.ReduceOp.MIN)
                has_adapters = bool(has_adapters_tensor.item())

            if not has_adapters:
                raise RuntimeError(
                    "No LoRA adapter found in the model on at least one rank; "
                    "aborting checkpoint save to avoid distributed deadlock."
                )

            if self.is_main_process:
                lora_dir = os.path.join(checkpoint_dir, "lora")
                os.makedirs(lora_dir, exist_ok=True)
                
                if hasattr(self.transformer, "peft_config") and self.transformer.peft_config:
                    adapter_names = list(self.transformer.peft_config.keys())
                    logger.info(f"Saving {len(adapter_names)} LoRA adapter(s): {adapter_names}")
                    
                    for adapter_name in adapter_names:
                        adapter_dir = os.path.join(lora_dir, adapter_name)
                        os.makedirs(adapter_dir, exist_ok=True)
                        self.transformer.save_lora_adapter(
                            save_directory=adapter_dir,
                            adapter_name=adapter_name,
                            safe_serialization=True,
                        )
                        logger.info(f"LoRA adapter '{adapter_name}' saved to {adapter_dir}")
            
            if self.world_size > 1:
                dist.barrier()
        
        # Save full model state dict
        model_state_dict = get_model_state_dict(self.transformer)
        dcp.save(
            state_dict={"model": model_state_dict},
            checkpoint_id=transformer_dir,
        )

        optimizer_state_dict = get_optimizer_state_dict(
            self.transformer,
            self.optimizer,
        )
        optimizer_dir = os.path.join(checkpoint_dir, "optimizer")
        dcp.save(
            state_dict={"optimizer": optimizer_state_dict},
            checkpoint_id=optimizer_dir,
        )
        
        if self.is_main_process:
            training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
            torch.save({
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "global_step": step,
            }, training_state_path)
        
        if self.world_size > 1:
            dist.barrier()
        
        if self.is_main_process:
            logger.info(f"Checkpoint saved at step {step} to {checkpoint_dir}")

    def load_pretrained_lora(self, lora_dir: str):
        self.transformer.load_lora_adapter(
            pretrained_model_name_or_path_or_dict=lora_dir,
            prefix=None,
            adapter_name="default",
            use_safetensors=True,
            hotswap=False,
        )
    
    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        if self.is_main_process:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        if self.world_size > 1:
            dist.barrier()
        
        
        transformer_dir = os.path.join(checkpoint_path, "transformer")
        if os.path.exists(transformer_dir):
            model_state_dict = get_model_state_dict(self.transformer)
            dcp.load(
                state_dict={"model": model_state_dict},
                checkpoint_id=transformer_dir,
            )
            if self.is_main_process:
                logger.info("Transformer model state loaded")
        else:
            logger.warning(f"Transformer dcp checkpoint not found from {checkpoint_path}")

        optimizer_dir = os.path.join(checkpoint_path, "optimizer")
        if os.path.exists(optimizer_dir):
            optimizer_state_dict = get_optimizer_state_dict(
                self.transformer,
                self.optimizer,
            )
            dcp.load(
                state_dict={"optimizer": optimizer_state_dict},
                checkpoint_id=optimizer_dir,
            )
            if self.is_main_process:
                logger.info("Optimizer state loaded")
        
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=self.device, weights_only=True)
            self.lr_scheduler.load_state_dict(training_state["lr_scheduler"])
            self.global_step = training_state.get("global_step", 0)
            if self.is_main_process:
                logger.info(f"Training state loaded: global_step={self.global_step}")
        
        if self.world_size > 1:
            global_step_tensor = torch.tensor(self.global_step, device=self.device)
            dist.broadcast(global_step_tensor, src=0)
            self.global_step = global_step_tensor.item()
        
        if self.world_size > 1:
            dist.barrier()
        
        if self.is_main_process:
            logger.info(f"Checkpoint loaded successfully. Resuming from step {self.global_step}")
    
    def _train_step_and_log(self, batch, loader_name: Optional[str] = None):
        """Execute one training step and handle logging/validation/checkpointing."""
        metrics = self.train_step(batch, loader_name=loader_name)
        
        if self.global_step % self.config.log_interval == 0 and self.is_main_process:
            loader_info = f"Loader: {loader_name} | " if loader_name else ""
            logger.info(
                f"Step {self.global_step}/{self.config.max_steps} | "
                f"{loader_info}"
                f"Loss: {metrics['loss']:.6f} | "
                f"Grad Norm: {metrics['grad_norm']:.4f} | "
                f"LR: {metrics['lr']:.2e}"
            )
        
        if self.global_step > 0 and self.global_step % self.config.validation_interval == 0:
            self.validate(self.global_step)
        
        if (self.global_step + 1) % self.config.save_interval == 0:
            self.save_checkpoint(self.global_step + 1)
            if self.world_size > 1:
                dist.barrier()
        
        self.global_step += 1
    
    def train(self, dataloader):
        if self.is_main_process:
            logger.info("Starting training...")
            logger.info(f"Max steps: {self.config.max_steps}")
            logger.info(f"Batch size: {self.config.batch_size}")
            logger.info(f"Learning rate: {self.config.learning_rate}")
        
        if self.config.resume_from_checkpoint is not None:
            self.load_checkpoint(self.config.resume_from_checkpoint)
            self.optimizer.zero_grad()
            ga = self.config.gradient_accumulation_steps
            remainder = self.global_step % ga
            if remainder != 0:
                old_step = self.global_step
                self.global_step = self.global_step - remainder
                if self.is_main_process:
                    logger.warning(
                        f"Aligned global_step from {old_step} to {self.global_step} "
                        f"to match gradient_accumulation_steps={ga}"
                    )
        
        self.transformer.train()
        
        if isinstance(dataloader, MultiDataloader):
            if self.is_main_process:
                logger.info(f"Using MultiDataloader with {len(dataloader.names)} dataloaders:")
                for name, prob in zip(dataloader.names, dataloader.probs):
                    logger.info(f"  - {name}: prob={prob:.2f}")
            
            while self.global_step < self.config.max_steps:
                loader_name, batch = dataloader.sample(sp_group=self.sp_group)
                self._train_step_and_log(batch, loader_name=loader_name)
        else:
            _epoch = 0
            while self.global_step < self.config.max_steps:
                if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
                    dataloader.sampler.set_epoch(_epoch)
                for batch in dataloader:
                    if self.global_step >= self.config.max_steps:
                        break
                    self._train_step_and_log(batch)
                _epoch += 1
        
        if self.global_step % self.config.save_interval != 0:
            self.save_checkpoint(self.global_step)
        
        if self.is_main_process:
            logger.info("Training completed!")
        
        if self.world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
    
    def validate(self, step: int):
        """
        Implement your own validation logic here
        An example:


        logger.info(f"Running validation at step {step}...")
        
        self.transformer.eval()
        
        try:
            for idx, prompt in enumerate(self.config.validation_prompts):
                logger.info(f"Generating validation video {idx+1}/{len(self.config.validation_prompts)}: {prompt[:50]}...")
                
                with torch.no_grad():
                    output = self.pipeline(
                        **input
                    )
                    
                    video_path = os.path.join(
                        self.validation_output_dir,
                        f"step_{step:06d}_prompt_{idx:02d}.mp4"
                    )
                    print(f"Prompt: {prompt}")
                    video_to_save = output.videos
                    if dist.get_rank() == 0:
                        save_video(video_to_save, video_path)
                        logger.info(f"Validation video saved to {video_path}")
        
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        finally:
            self.transformer.train()
        pass
        """


def _make_dataloader(dataset, config: TrainingConfig, dp_rank: int = 0, dp_size: int = 1):
    """Build a DataLoader with DistributedSampler when dp_size > 1."""
    if dp_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=True,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=training_collate_fn,
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=training_collate_fn,
    )


def create_dummy_dataloader1(config: TrainingConfig, dp_rank: int = 0, dp_size: int = 1):
    """
    Placeholder ``DataLoader`` for t2v / i2v-style samples (replace with real data).

    **Per-sample dict from ``Dataset.__getitem__``**

    Required:
    - ``pixel_values`` (``torch.Tensor``, ``float32``): pixels in ``[-1, 1]``.
      Video: ``(C, F, H, W)``; image: ``(C, H, W)``. For video, ``F`` must be ``4n+1``.
    - ``text`` (``str``): prompt.
    - ``data_type`` (``str``): ``"video"`` or ``"image"`` (image path still uses t2v unless
      multi-task maps another loader).

    Optional:
    - ``latents`` (``torch.Tensor``): pre-encoded VAE latents ``(C_lat, F, H_lat, W_lat)`` in
      the same layout as the model's VAE output. If set, training skips encoding ``pixel_values``.
    - ``byt5_text_ids`` / ``byt5_text_mask`` (``torch.LongTensor``, shape ``(L,)``): optional
      pre-tokenized byT5; if both omitted, prompts are tokenized in the trainer.

    **Task routing**
    - Single loader: uses ``TrainingConfig.task_type`` (default t2v).
    - ``MultiDataloader``: ``loader_name`` selects the task via ``LOADER_TASK_MAP`` (e.g. ``t2v``, ``i2v``).

    **Examples (shapes only)**

    Video with pixels::

        {"pixel_values": (3, 121, 480, 848), "text": "...", "data_type": "video"}

    Video with latents only::

        {"latents": (32, 31, 30, 53), "text": "...", "data_type": "video"}
    """
    # Minimal dummy dataset; swap for your own files and decoding.
    class DummyDataset1:
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Video: temporal dimension must be 4n+1, using 121 frames
            # Generate data in range [-1, 1]

            resolution = (121, 480, 848)
            latent_resolution = [(resolution[0] - 1) // 4 + 1, resolution[1] // 16, resolution[2] // 16]

            data = torch.rand(3, *resolution) * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            data_type = "video"

            return {
                "pixel_values": data,
                "text": "A sample prompt",
                "data_type": data_type,
                "latents": torch.randn(32, *latent_resolution),
                "byt5_text_ids": torch.zeros((256), dtype=torch.int64),
                "byt5_text_mask": torch.zeros((256), dtype=torch.int64),
            }
    
    dataset = DummyDataset1()
    return _make_dataloader(dataset, config, dp_rank, dp_size)


def create_dummy_dataloader2(config: TrainingConfig, dp_rank: int = 0, dp_size: int = 1):
    """
    Placeholder for **multi_imgs_to_v** (subject / multi-image conditioning).

    Extra keys beyond ``create_dummy_dataloader1``:
    - ``condition``: ``list[PIL.Image]`` — subject (and optionally background) reference images.

    Layout is analogous to a typical ``video_loader_2``-style dataset; wire your own paths and prompts.
    """

    class DummyDataset2:
        def __init__(self, size=100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            resolution = (121, 480, 848)
            latent_resolution = [(resolution[0] - 1) // 4 + 1, resolution[1] // 16, resolution[2] // 16]

            data = torch.rand(3, *resolution) * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            data_type = "video"

            # Subject injection: condition images (subject + optional background)
            condition_imgs = [
                Image.new('RGB', (resolution[2], resolution[1]), color=(128, 128, 128)),  # subject image
            ]

            return {
                "pixel_values": data,
                "text": "A sample prompt for subject injection",
                "data_type": data_type,
                "latents": torch.randn(32, *latent_resolution),
                "condition": condition_imgs,
                "byt5_text_ids": torch.zeros((256), dtype=torch.int64),
                "byt5_text_mask": torch.zeros((256), dtype=torch.int64),
            }

    dataset = DummyDataset2()
    return _make_dataloader(dataset, config, dp_rank, dp_size)

def create_dummy_dataloader3(config: TrainingConfig, dp_rank: int = 0, dp_size: int = 1):
    """
    Placeholder for **key_frames_to_v** (sparse key-frame conditioning).

    Extra keys:
    - ``condition``: ``list[PIL.Image]`` — one image per key frame.
    - ``frame_id``: ``list[int]`` — same length as ``condition``; frame index where each image applies
      (e.g. ``[0, 15, 30]`` places three images at those temporal indices).
    """

    class DummyDataset3:
        def __init__(self, size=100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            resolution = (121, 480, 848)
            latent_resolution = [(resolution[0] - 1) // 4 + 1, resolution[1] // 16, resolution[2] // 16]

            data = torch.rand(3, *resolution) * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            data_type = "video"

            # Key-frame condition: images placed at explicit frame indices
            condition_imgs = [
                Image.new('RGB', (resolution[2], resolution[1]), color=(100, 100, 100)),  # key frame 1
                Image.new('RGB', (resolution[2], resolution[1]), color=(200, 200, 200)),  # key frame 2
            ]
            frame_ids = [0, 15]  # place first image at frame 0, second at frame 15

            return {
                "pixel_values": data,
                "text": "A sample prompt for key frames to video",
                "data_type": data_type,
                "latents": torch.randn(32, *latent_resolution),
                "condition": condition_imgs,
                "frame_id": frame_ids,
                "byt5_text_ids": torch.zeros((256), dtype=torch.int64),
                "byt5_text_mask": torch.zeros((256), dtype=torch.int64),
            }

    dataset = DummyDataset3()
    return _make_dataloader(dataset, config, dp_rank, dp_size)


def create_dummy_dataloader4(config: TrainingConfig, dp_rank: int = 0, dp_size: int = 1):
    """
    Placeholder for **v2v** (video-to-video / editing).

    Extra keys:
    - ``input_video_path``: path to the conditioning video (read inside ``prepare_batch``).
    - ``input_video_latents``: conditioning latents, same spatiotemporal shape as target latents.

    ``sampled_frames`` / first-frame numpy for the condition stream are derived in ``prepare_batch``
    from ``input_video_path`` (see multitask_utils-style helpers), not from the collated batch alone.
    """
    class DummyDataset4:
        def __init__(self, size=100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            resolution = (121, 480, 848)
            latent_resolution = [(resolution[0] - 1) // 4 + 1, resolution[1] // 16, resolution[2] // 16]

            data = torch.rand(3, *resolution) * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            data_type = "video"

            # Video editing: condition latents of same shape as target latents
            condition_latents = torch.randn(32, *latent_resolution)

            return {
                "pixel_values": data,
                "text": "A sample prompt for video editing",
                "data_type": data_type,
                "latents": torch.randn(32, *latent_resolution),
                "input_video_path": "/dummy/condition_video.mp4",
                "input_video_latents": condition_latents,
                "byt5_text_ids": torch.zeros((256), dtype=torch.int64),
                "byt5_text_mask": torch.zeros((256), dtype=torch.int64),
            }

    dataset = DummyDataset4()
    return _make_dataloader(dataset, config, dp_rank, dp_size)



def create_dummy_dataloader5(config: TrainingConfig, dp_rank: int = 0, dp_size: int = 1):
    """
    Placeholder for **tiv2v** (text + image + video conditioning for editing).

    Extra keys:
    - ``input_img``: ``list[PIL.Image]`` — reference frame(s) for image conditioning.
    - ``input_video_path`` / ``input_video_latents``: same role as in ``create_dummy_dataloader4``.

    Condition streams from the path are finalized in ``prepare_batch`` together with ``input_video_path``.
    """

    class DummyDataset5:
        def __init__(self, size=100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            resolution = (121, 480, 848)
            latent_resolution = [(resolution[0] - 1) // 4 + 1, resolution[1] // 16, resolution[2] // 16]

            data = torch.rand(3, *resolution) * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            data_type = "video"

            # IV2V editing: condition images + condition latents
            condition_latents = torch.randn(32, *latent_resolution)
            input_imgs = [
                Image.new('RGB', (resolution[2], resolution[1]), color=(150, 150, 150)),  # condition image
            ]

            return {
                "pixel_values": data,
                "text": "A sample prompt for IV2V editing",
                "data_type": data_type,
                "latents": torch.randn(32, *latent_resolution),
                "input_img": input_imgs,
                "input_video_path": "/dummy/condition_video.mp4",
                "input_video_latents": condition_latents,
                "byt5_text_ids": torch.zeros((256), dtype=torch.int64),
                "byt5_text_mask": torch.zeros((256), dtype=torch.int64),
            }

    dataset = DummyDataset5()
    return _make_dataloader(dataset, config, dp_rank, dp_size)


def main():
    parser = argparse.ArgumentParser(description="Train HunyuanVideo-1.5 on video data")
    
    # Model paths
    parser.add_argument("--pretrained_model_root", type=str, default='ckpts', help="Path to pretrained model")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--train_timestep_shift", type=float, default=3.0, help="Train Timestep shift")
    parser.add_argument("--flow_snr_type", type=str, default="lognorm", 
                        choices=["uniform", "lognorm", "mix", "mode"],
                        help="SNR type for flow matching: uniform, lognorm, mix, or mode (default: lognorm)")
    
    # Data parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint save interval")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    
    # Other parameters
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"], help="Data type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deepstack", type=int, nargs='*', default=[8, 16, 24],
                        help="Deepstack layer indices for text encoder (default: 8 16 24). "
                             "Must match the value used during inference. Use --deepstack (empty) to disable.")
    parser.add_argument("--setclip", type=str_to_bool, nargs='?', const=True, default=True,
                        help="Enable CLIP features in text encoder (default: true). "
                             "Must match the value used during inference.")
    parser.add_argument("--use_muon", type=str_to_bool, nargs='?', const=True, default=True,
        help="Use Muon optimizer for training (default: true). "
             "Use --use_muon or --use_muon true/1 to enable, --use_muon false/0 to disable"
    )
    # FSDP and gradient checkpointing
    parser.add_argument(
        "--enable_fsdp", type=str_to_bool, nargs='?', const=True, default=True,
        help="Enable FSDP for distributed training (default: true). "
             "Use --enable_fsdp or --enable_fsdp true/1 to enable, --enable_fsdp false/0 to disable"
    )
    parser.add_argument(
        "--enable_gradient_checkpointing", type=str_to_bool, nargs='?', const=True, default=True,
        help="Enable gradient checkpointing (default: true). "
             "Use --enable_gradient_checkpointing or --enable_gradient_checkpointing true/1 to enable, "
             "--enable_gradient_checkpointing false/0 to disable"
    )
    parser.add_argument(
        "--sp_size", type=int, default=8,
        help="Sequence parallelism size (default: 8). Must evenly divide world_size. "
             "For example, if world_size=8, valid sp_size values are 1, 2, 4, 8."
    )
    parser.add_argument(
        "--dp_replicate", type=int, default=1,
        help="Data parallelism replicate size (default: 1). "
    )
    
    # Validation parameters
    parser.add_argument("--validation_interval", type=int, default=100, help="Run validation every N steps (default: 100)")
    parser.add_argument("--validation_prompts", type=str, nargs="+", default=None, 
                        help="Prompts for validation (default: single default prompt). Can specify multiple prompts.")
    parser.add_argument("--validation_timestep_shift", type=float, default=5.0, help="Validation Timestep shift")
    parser.add_argument("--validate_video_length", type=int, default=241, help="Video length (number of frames) for validation (default: 241)")
    
    # Resume training parameters
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume training from (e.g., ./outputs/checkpoint-1000)")
    
    # LoRA parameters
    parser.add_argument("--use_lora", type=str_to_bool, nargs='?', const=True, default=False,
                        help="Enable LoRA training (default: false). "
                             "Use --use_lora or --use_lora true/1 to enable, --use_lora false/0 to disable")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha scaling parameter (default: 16)")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout rate (default: 0.0)")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                        help="Target modules for LoRA (default: all Linear layers). "
                             "Example: --lora_target_modules img_attn_q img_attn_v img_mlp.fc1")
    parser.add_argument("--pretrained_lora_path", type=str, default=None,
                        help="Path to pretrained LoRA adapter to load. If provided, will load this adapter instead of creating a new one.")
    
    # Multi-dataloader parameters
    parser.add_argument("--dataloader_probs", type=str, required=True,
                        help="Multi-task mixing: comma-separated name:prob pairs (probabilities sum to 1). "
                             "Example: t2v:0.3,i2v:0.2,key_frames_to_v:0.1,multi_imgs_to_v:0.1,v2v:0.15,tiv2v:0.15. "
                             "Names must match LOADER_TASK_MAP and the built-in dummy loaders.")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        pretrained_model_root=args.pretrained_model_root,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        dtype=args.dtype,
        seed=args.seed,
        deepstack=args.deepstack if args.deepstack else [],
        setclip=args.setclip,
        enable_fsdp=args.enable_fsdp,
        enable_gradient_checkpointing=args.enable_gradient_checkpointing,
        sp_size=args.sp_size,
        use_muon=args.use_muon,
        dp_replicate=args.dp_replicate,
        validation_interval=args.validation_interval,
        validation_prompts=args.validation_prompts,
        train_timestep_shift=args.train_timestep_shift,
        validation_timestep_shift=args.validation_timestep_shift,
        snr_type=SNRType(args.flow_snr_type),
        validate_video_length=args.validate_video_length,
        resume_from_checkpoint=args.resume_from_checkpoint,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        pretrained_lora_path=args.pretrained_lora_path,
        dataloader_probs=args.dataloader_probs,
    )
    
    trainer = HunyuanVideoTrainer(config)
    
    prob_dict = {}
    for item in config.dataloader_probs.split(","):
        name, prob = item.strip().split(":")
        p = float(prob.strip())
        if p > 0:
            prob_dict[name.strip()] = p
    
    if not prob_dict:
        raise ValueError("All dataloader probabilities are 0. At least one must be > 0.")
    
    available_dataloaders = {
        "t2v": create_dummy_dataloader1,
        "i2v": create_dummy_dataloader1,
        "key_frames_to_v": create_dummy_dataloader3,
        "multi_imgs_to_v": create_dummy_dataloader2,
        "v2v": create_dummy_dataloader4,
        "tiv2v": create_dummy_dataloader5,
    }
    
    dataloaders = {}
    for name in prob_dict:
        if name not in available_dataloaders:
            raise ValueError(
                f"Unknown dataloader name '{name}'. "
                f"Available: {list(available_dataloaders.keys())}"
            )
        dataloaders[name] = available_dataloaders[name](config, dp_rank=trainer.dp_rank, dp_size=trainer.dp_size)
    
    dataloader = MultiDataloader(
        dataloaders=dataloaders,
        probabilities=prob_dict,
    )
    if trainer.is_main_process:
        logger.info(f"Created MultiDataloader with probabilities: {prob_dict}")
    
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
