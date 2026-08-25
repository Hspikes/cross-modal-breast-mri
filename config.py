from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class UNetConfig:
    """
    U-Net 结构配置。

    关键字段:
        latent_channels:    VAE latent 的通道数 C_z，必须与 VAEConfig.latent_channels 一致
        block_out_channels: 每一层 U-Net 的 feature 通道数 (encoder/decoder 对称)
        down_block_types:   下采样阶段每层使用的 Block 类型
        up_block_types:     上采样阶段每层使用的 Block 类型
        num_res_blocks:     每个尺度下 residual block 的数量 (layers_per_block)

        sample_size:        latent 特征图的空间尺寸 H' (H' = W')，可以为 None，
                           None 时 diffusers 会在第一次前向时根据输入自动推断。
    """

    latent_channels: int = 8
    max_timesteps: int = 1000

    block_out_channels: Tuple[int, ...] = (64, 128, 256, 256)
    down_block_types: Tuple[str, ...] = (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
    )
    up_block_types: Tuple[str, ...] = (
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )
    num_res_blocks: int = 2

    sample_size: Optional[int] = None


@dataclass
class VAEConfig:
    img_channels: int = 1
    img_size: int = 256
    latent_channels: int = 8
    lr: float = 1e-4
    epochs: int = 100
    batch_size: int = 1
    num_workers: int = 4
    recon_loss_type: str = "l1"
    kl_weight: float = 1e-4
    device: str = "cuda:0"
    data_root: str = "./data/raw_png"
    train_list: str = "train_list.txt"
    eval_list: str = "eval_list.txt"
    ckpt_dir: str = "./ckpt/ckpt_vae"
    recon_vis_dir: str = "./recon_vis/VAE"


@dataclass
class UnetTrainConfig:
    # 数据与设备
    device: str = "cuda:0"
    data_root: str = "./data/raw_png"
    train_list: str = "train_list.txt"
    eval_list: str = "eval_list.txt"
    num_workers: int = 4

    # 训练超参
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-4
    grad_clip: float = 1.0

    # 扩散噪声调度
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # 可视化：固定验证样本 + 固定加噪强度
    vis_interval_epochs: int = 10
    vis_num_images: int = 4
    vis_timestep: int = 100
    vis_noise_seed: int = 1234

    # 路径
    vae_ckpt_path: str = "./ckpt/ckpt_vae/vae_best.pth"
    ckpt_dir: str = "./ckpt/ckpt_unet"
    log_interval: int = 50
    save_interval: int = 5
    recon_vis_dir: str = "./recon_vis/Unet"


@dataclass
class UnetInferConfig:
    # 数据与设备
    device: str = "cuda:0"
    data_root: str = "./data/raw_png"
    eval_list: str = "eval_list.txt"
    num_workers: int = 0

    # 随机抽样（验证集切片）
    sample_seed: int = 42
    num_images: int = 8

    # 扩散噪声调度（需与训练保持一致）
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # 可视化：对同一张原图在多个时间步分别加噪并去噪
    timesteps: Tuple[int, ...] = (50, 200, 500, 800)
    noise_seed: int = 1234
    use_same_noise_across_timesteps: bool = True

    # 路径
    vae_ckpt_path: str = "./ckpt/ckpt_vae/vae_best.pth"
    unet_ckpt_path: str = "./ckpt/ckpt_unet/latest.pt"
    out_dir: str = "./recon_vis/UnetInfer"


@dataclass
class CondLDMTrainConfig:
    device: str = "cuda:0"
    data_root: str = "./data/raw_png"
    train_list: str = "train_list.txt"
    eval_list: str = "eval_list.txt"
    num_workers: int = 4

    epochs: int = 100
    batch_size: int = 2
    lr: float = 1e-4
    grad_clip: float = 1.0

    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # 可视化：固定验证样本 + 固定加噪强度（按 epoch 触发）
    vis_interval_epochs: int = 10
    vis_num_images: int = 4
    vis_timestep: int = 500
    vis_noise_seed: int = 1234

    vae_ckpt_path: str = "./ckpt/ckpt_vae/vae_best.pth"
    teacher_unet_ckpt: str = "./ckpt/ckpt_unet/latest.pt"
    ckpt_dir: str = "./ckpt/ckpt_cond"
    log_interval: int = 50
    save_interval: int = 5
    recon_vis_dir: str = "./recon_vis/Cond"


@dataclass
class CondLDMTrainCFGConfig:
    """
    条件 LDM 训练（改进版）：
    - per-sample cond dropout（用于 CFG）
    - 对 gate/bias/norm 等参数禁用 weight decay
    - 可配置 gate_init 与多尺度条件映射
    """

    device: str = "cuda:0"
    data_root: str = "./data/raw_png"
    train_list: str = "train_list.txt"
    eval_list: str = "eval_list.txt"
    num_workers: int = 4

    epochs: int = 100
    batch_size: int = 2
    lr: float = 1e-4
    grad_clip: float = 1.0

    # Optimizer
    weight_decay: float = 0.01
    gate_lr_mult: float = 1.0
    teacher_lr_mult: float = 0.05
    teacher_weight_decay: float = 0.0

    # Partially unfreeze teacher UNet (module1)
    unfreeze_teacher: bool = True
    unfreeze_teacher_mid: bool = True
    unfreeze_teacher_up_blocks: int = 1  # unfreeze last N up_blocks
    unfreeze_teacher_out: bool = True

    # CFG training: randomly drop condition (per sample)
    cond_drop_prob: float = 0.1
    uncond_fill: str = "zeros"  # "zeros" | "noise"

    # Diffusion schedule
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Model: cross-attn + condition mapping
    cross_attn_heads: int = 4
    cross_attn_dim_head: int = 64
    cond_proj_channels: Optional[int] = None
    cross_attn_feat_indices: Tuple[int, ...] = (-1, -2, -3)
    gate_init: float = 1e-2

    # 多尺度条件映射：键为 "stem"/"down"/"mid"/"up"/"default" 或 "down_0".../"up_0"...
    # 值为 cross_attn_feat_indices 中的一个 index（如 -1/-2/-3）
    stage_cond_feat_indices: Tuple[Tuple[str, int], ...] = (
        ("default", -2),
        ("stem", -3),
        ("down", -2),
        ("mid", -1),
        ("up", -2),
    )

    # Visualization
    vis_interval_epochs: int = 10
    vis_num_images: int = 4
    vis_timestep: int = 500
    vis_noise_seed: int = 1234
    vis_guidance_scale: float = 3.0

    # Paths
    vae_ckpt_path: str = "./ckpt/ckpt_vae/vae_best.pth"
    teacher_unet_ckpt: str = "./ckpt/ckpt_unet/latest.pt"
    ckpt_dir: str = "./ckpt/ckpt_cond_cfg"
    log_interval: int = 50
    save_interval: int = 5
    recon_vis_dir: str = "./recon_vis/CondCFG"

    # SwanLab
    swanlab_project: str = "Conditional-injected-LDM-training-CFG"
    swanlab_enabled: bool = True


@dataclass
class CondLDMInferConfig:
    # 数据与设备
    device: str = "cuda:0"
    data_root: str = "./data/raw_png"
    eval_list: str = "eval_list.txt"
    num_workers: int = 0

    # 随机抽样（验证集切片）
    sample_seed: int = 42
    num_images: int = 8

    # 扩散噪声调度（需与训练保持一致）
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # 可视化：对同一张原图在多个时间步分别加噪并去噪（条件为 DWI）
    timesteps: Tuple[int, ...] = (50, 200, 500, 800)
    noise_seed: int = 1234
    use_same_noise_across_timesteps: bool = True

    # 路径
    vae_ckpt_path: str = "./ckpt/ckpt_vae/vae_best.pth"
    teacher_unet_ckpt: str = "./ckpt/ckpt_unet/latest.pt"
    cond_ckpt_path: str = "./ckpt/ckpt_cond/latest.pt"
    out_dir: str = "./recon_vis/CondInfer"


@dataclass
class DistillConfig:
    device: str = "cuda:0"
    data_root: str = "./data/raw_png"
    train_list: str = "train_list.txt"
    eval_list: str = "eval_list.txt"
    num_workers: int = 4

    epochs: int = 10
    batch_size: int = 2
    lr: float = 1e-4
    grad_clip: float = 1.0

    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    vae_ckpt_path: str = "./ckpt/ckpt_vae/vae_best.pth"
    teacher_unet_ckpt: str = "./ckpt/ckpt_unet/latest.pt"
    ckpt_dir: str = "./ckpt/ckpt_distill"
    log_interval: int = 50
    save_interval: int = 5
    recon_vis_dir: str = "./recon_vis/Distill"

    module2_ckpt_path: str = "./ckpt/ckpt_cond/latest.pt"

    distill_weight: float = 1.0

    use_vae_adapter_branch: bool = True

    vis_max: int = 4

    # 可视化：固定验证样本 + 固定加噪强度（按 epoch 触发）
    vis_interval_epochs: int = 10
    vis_timestep: int = 500
    vis_noise_seed: int = 1234


@dataclass
class CondLDMDiagConfig:
    """
    cond_ldm 诊断/对比脚本配置：
    - 固定时间步 t 下比较 MSE(pred_noise, noise)
    - 从 t 开始做多步反向扩散到 0 再 decode 可视化
    """

    t: int = 900
    num_ids: int = 1
    steps: int = 50
    seed: int = 42
    out_dir: str = "./recon_vis/CondDiag"


@dataclass
class CondLDMDiagCFGConfig:
    """
    CFG 去噪诊断/对比脚本配置：
    - 固定时间步 t 下比较 MSE(pred_eps, eps)
    - 从 t 开始做多步反向扩散到 0 再 decode 可视化
    - 增加 CFG：eps_guided = eps_uncond + s*(eps_cond - eps_uncond)
    """

    t: int = 900
    num_ids: int = 1
    steps: int = 50
    seed: int = 42

    guidance_scale: float = 3.0
    uncond_fill: str = "zeros"  # "zeros" | "noise"
    uncond_noise_seed: int = 12345

    out_dir: str = "./recon_vis/CondDiagCFG"


@dataclass
class CondLDMGenConfig:
    """
    cond_ldm 条件生成（给定 b800/DWI -> 生成 T1CE）配置。
    """

    num_ids: int = 1
    seed: int = 42

    num_inference_steps: int = 50

    sampler: str = "ddim"  # "dpmpp" | "ddim"
    eta: float = 0.0

    start_t: int = 800

    # 初始噪声尺度：
    # - "unit": z_start ~ N(0, I)
    # - "match_t": z_start ~ sqrt(1 - alpha_bar(start_t)) * N(0, I)
    init_noise_scale_mode: str = "match_t"

    out_dir: str = "./recon_vis/CondGenB800"


@dataclass
class CondLDMGenCFGConfig:
    """
    CFG 条件生成（给定 b800/DWI -> 生成 T1CE）配置。
    - 采样时每步做两次 forward（cond/uncond）并用 guidance_scale 合成 eps。
    """

    num_ids: int = 1
    seed: int = 42

    num_inference_steps: int = 50
    sampler: str = "ddim"  # "dpmpp" | "ddim"
    eta: float = 0.0

    start_t: int = 800
    init_noise_scale_mode: str = "match_t"  # "unit" | "match_t"

    guidance_scale: float = 3.0
    uncond_fill: str = "zeros"  # "zeros" | "noise"
    uncond_noise_seed: int = 23456

    out_dir: str = "./recon_vis/CondGenB800CFG"


__all__ = [
    "UNetConfig",
    "VAEConfig",
    "UnetTrainConfig",
    "UnetInferConfig",
    "CondLDMTrainConfig",
    "CondLDMTrainCFGConfig",
    "CondLDMInferConfig",
    "CondLDMDiagConfig",
    "CondLDMDiagCFGConfig",
    "CondLDMGenConfig",
    "CondLDMGenCFGConfig",
    "DistillConfig",
]
