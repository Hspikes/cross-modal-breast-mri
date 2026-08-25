from __future__ import annotations

import os
from typing import Iterable, Tuple

import torch
from torchvision.utils import save_image

from config import CondLDMInferConfig as InferConfig
from config import UNetConfig, VAEConfig

from ..data_pre_processing.dataset import MRIDataset, MRItransform
from ..VAE.model import VAE
from .model import Dwi_Cond_LatentUNet


def setup_seed(seed: int = 42):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_noise_schedule(cfg: InferConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_train_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas_cumprod


def load_vae(config: VAEConfig, ckpt_path: str, device: torch.device) -> VAE:
    vae = VAE(config)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        vae.load_state_dict(state["model"])
    else:
        vae.load_state_dict(state)
    vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def load_teacher_unet_config_and_state(
    default_config: UNetConfig,
    ckpt_path: str,
    device: torch.device,
) -> Tuple[UNetConfig, dict]:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        return state.get("config", default_config), state["model"]
    return default_config, state


def _load_model_state(model: torch.nn.Module, state: dict) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"checkpoint 存在多余参数，示例：{unexpected[:5]}")

    missing_non_teacher = [k for k in missing if not k.startswith("latent_unet.")]
    if missing_non_teacher:
        raise RuntimeError(f"checkpoint 缺少必要参数，示例：{missing_non_teacher[:5]}")


def load_cond_model(
    *,
    cfg: InferConfig,
    vae_cfg: VAEConfig,
    device: torch.device,
) -> Dwi_Cond_LatentUNet:
    teacher_default_cfg = UNetConfig(latent_channels=vae_cfg.latent_channels)
    unet_cfg, unet_state = load_teacher_unet_config_and_state(
        teacher_default_cfg,
        cfg.teacher_unet_ckpt,
        device,
    )
    model = Dwi_Cond_LatentUNet(
        unet_config=unet_cfg,
        unet_state_dict=unet_state,
        dwi_in_channels=vae_cfg.img_channels,
    ).to(device)

    ckpt = torch.load(cfg.cond_ckpt_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    _load_model_state(model, state_dict)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) * 0.5


@torch.no_grad()
def infer_one(
    *,
    x_t1ce: torch.Tensor,  # [1, C, H, W]
    x_dwi: torch.Tensor,  # [1, C, H, W]
    vae: VAE,
    model: Dwi_Cond_LatentUNet,
    alphas_cumprod: torch.Tensor,  # [T]
    timesteps: Iterable[int],
    noise_seed: int,
    use_same_noise_across_timesteps: bool,
    sample_index: int,
) -> torch.Tensor:
    mean, _logvar = vae.encode(x_t1ce)
    z0 = mean

    ts = list(int(t) for t in timesteps)
    if not ts:
        raise ValueError("timesteps 不能为空。")
    max_t = int(alphas_cumprod.numel()) - 1
    if max(ts) > max_t:
        raise ValueError(f"timesteps 存在超过最大时间步 {max_t} 的值：{ts}")

    def _make_noise(seed: int) -> torch.Tensor:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        return torch.randn(z0.shape, device="cpu", generator=gen, dtype=z0.dtype).to(
            z0.device
        )

    base_noise = (
        _make_noise(noise_seed + sample_index)
        if use_same_noise_across_timesteps
        else None
    )

    noisy_imgs = []
    denoised_imgs = []
    for t_int in ts:
        t = torch.full(
            (x_t1ce.size(0),), int(t_int), device=x_t1ce.device, dtype=torch.long
        )
        a_bar = alphas_cumprod[t].view(-1, 1, 1, 1)

        noise = (
            base_noise
            if base_noise is not None
            else _make_noise(noise_seed + sample_index * 10000 + t_int)
        )
        zt = torch.sqrt(a_bar) * z0 + torch.sqrt(1.0 - a_bar) * noise
        x_noisy = vae.decode(zt)

        pred_noise = model(latents=zt, timesteps=t, dwi_image=x_dwi)
        z0_hat = (zt - torch.sqrt(1.0 - a_bar) * pred_noise) / torch.sqrt(a_bar + 1e-8)
        x_denoised = vae.decode(z0_hat)

        noisy_imgs.append(x_noisy)
        denoised_imgs.append(x_denoised)

    n_t = len(ts)
    dwi_row = _denorm(x_dwi).repeat(n_t, 1, 1, 1)
    orig_row = _denorm(x_t1ce).repeat(n_t, 1, 1, 1)
    noisy_row = torch.cat([_denorm(v) for v in noisy_imgs], dim=0)
    denoised_row = torch.cat([_denorm(v) for v in denoised_imgs], dim=0)
    return torch.cat([dwi_row, orig_row, noisy_row, denoised_row], dim=0)


@torch.no_grad()
def main():
    cfg = InferConfig()
    setup_seed(cfg.sample_seed)

    device = torch.device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    _betas, alphas_cumprod = get_noise_schedule(cfg)
    alphas_cumprod = alphas_cumprod.to(device)

    tf = MRItransform()
    eval_dataset = MRIDataset(
        data_folder=cfg.data_root,
        sequence_list_txt=cfg.eval_list,
        transforms=tf,
    )

    if cfg.num_images <= 0:
        raise ValueError(f"num_images 需为正数，当前为 {cfg.num_images}")
    if cfg.num_images > len(eval_dataset):
        raise ValueError(
            f"num_images={cfg.num_images} 超过验证集大小 {len(eval_dataset)}"
        )

    perm_gen = torch.Generator()
    perm_gen.manual_seed(int(cfg.sample_seed))
    chosen = torch.randperm(len(eval_dataset), generator=perm_gen)[
        : int(cfg.num_images)
    ].tolist()

    vae_cfg = VAEConfig(device=cfg.device)
    vae = load_vae(vae_cfg, cfg.vae_ckpt_path, device)

    model = load_cond_model(cfg=cfg, vae_cfg=vae_cfg, device=device)

    for i, dataset_index in enumerate(chosen):
        item = eval_dataset[int(dataset_index)]
        x_t1ce = item["gt"].to(device).unsqueeze(0)  # [1,1,H,W]
        x_dwi = item["image"].to(device).unsqueeze(0)  # [1,1,H,W]

        grid = infer_one(
            x_t1ce=x_t1ce,
            x_dwi=x_dwi,
            vae=vae,
            model=model,
            alphas_cumprod=alphas_cumprod,
            timesteps=cfg.timesteps,
            noise_seed=cfg.noise_seed,
            use_same_noise_across_timesteps=cfg.use_same_noise_across_timesteps,
            sample_index=i,
        )

        out_path = os.path.join(
            cfg.out_dir, f"sample_{i:03d}_idx{int(dataset_index):06d}.png"
        )
        save_image(grid, out_path, nrow=len(cfg.timesteps))


if __name__ == "__main__":
    main()
