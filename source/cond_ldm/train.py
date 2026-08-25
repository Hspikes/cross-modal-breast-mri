import glob
import math
import os
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import swanlab
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from config import CondLDMTrainConfig as TrainConfig
from config import UNetConfig, VAEConfig

from ..data_pre_processing.dataset import MRIDataset, MRItransform
from ..VAE.model import VAE, reparameterize
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


def _find_resume_ckpt(ckpt_dir: str) -> Optional[str]:
    latest = os.path.join(ckpt_dir, "latest.pt")
    if os.path.isfile(latest):
        return latest
    candidates = sorted(glob.glob(os.path.join(ckpt_dir, "cond_epoch_*.pt")))
    return candidates[-1] if candidates else None


def _load_ckpt_state(ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        return state
    raise ValueError(f"Unexpected checkpoint format at {ckpt_path}")


def get_noise_schedule(cfg: TrainConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_train_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas_cumprod


def add_noise(latents: torch.Tensor, t: torch.Tensor, alphas_cumprod: torch.Tensor):
    a_bar = alphas_cumprod[t].view(-1, 1, 1, 1)
    noise = torch.randn_like(latents)
    noisy = torch.sqrt(a_bar) * latents + torch.sqrt(1.0 - a_bar) * noise
    return noisy, noise


@torch.no_grad()
def save_recon_vis(
    x_t1ce: torch.Tensor,
    zt: torch.Tensor,
    pred_noise: torch.Tensor,
    t: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    vae: VAE,
    cfg: TrainConfig,
    epoch: int,
):
    """
    将固定验证样本的预测去噪结果可视化：原始 T1CE vs 估计 z0 解码后的重建。
    """
    os.makedirs(cfg.recon_vis_dir, exist_ok=True)
    a_bar = alphas_cumprod[t].view(-1, 1, 1, 1)
    z0_hat = (zt - torch.sqrt(1.0 - a_bar) * pred_noise) / torch.sqrt(a_bar + 1e-8)
    x_hat = vae.decode(z0_hat)

    def _denorm(v):
        return (v.clamp(-1, 1) + 1) * 0.5

    n = (
        min(int(cfg.vis_num_images), x_t1ce.size(0))
        if cfg.vis_num_images > 0
        else min(4, x_t1ce.size(0))
    )
    grid = torch.cat([_denorm(x_t1ce[:n]), _denorm(x_hat[:n])], dim=0)
    t0 = int(t[0].item()) if t.numel() > 0 else -1
    save_image(
        grid, os.path.join(cfg.recon_vis_dir, f"e{epoch:03d}_t{t0:04d}.png"), nrow=n
    )


@torch.no_grad()
def get_fixed_eval_images(
    eval_dataset: torch.utils.data.Dataset, device: torch.device, cfg: TrainConfig
):
    """
    固定取验证集开头的若干张图片（T1CE + DWI），用于整个训练过程中的可视化对比。
    """
    loader = DataLoader(
        eval_dataset, batch_size=cfg.vis_num_images, shuffle=False, num_workers=0
    )
    batch = next(iter(loader))
    return batch["gt"].to(device), batch["image"].to(device)


@torch.no_grad()
def visualize_on_fixed_eval(
    model: Dwi_Cond_LatentUNet,
    vae: VAE,
    x_t1ce: torch.Tensor,
    x_dwi: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    cfg: TrainConfig,
    epoch: int,
):
    """
    固定使用验证集样例、固定扩散时间步与固定噪声，便于跨 epoch 对比可视化效果。
    """
    model.eval()

    mean, _logvar = vae.encode(x_t1ce)
    z0 = mean

    t_fixed = int(max(0, min(cfg.num_train_timesteps - 1, cfg.vis_timestep)))
    t = torch.full((x_t1ce.size(0),), t_fixed, device=x_t1ce.device, dtype=torch.long)

    gen = torch.Generator()
    gen.manual_seed(int(cfg.vis_noise_seed))
    noise = torch.randn(z0.shape, device="cpu", generator=gen, dtype=z0.dtype).to(
        z0.device
    )

    a_bar = alphas_cumprod[t].view(-1, 1, 1, 1)
    zt = torch.sqrt(a_bar) * z0 + torch.sqrt(1.0 - a_bar) * noise

    pred_noise = model(latents=zt, timesteps=t, dwi_image=x_dwi)
    save_recon_vis(x_t1ce, zt, pred_noise, t, alphas_cumprod, vae, cfg, epoch=epoch)
    model.train()


def load_vae(cfg: VAEConfig, ckpt_path: str, device: torch.device) -> VAE:
    vae = VAE(cfg)
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


def load_teacher_unet(
    ckpt_path: str, device: torch.device, latent_channels: int
) -> tuple:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        unet_state = state["model"]
        unet_cfg = state.get("config", UNetConfig(latent_channels=latent_channels))
    else:
        unet_state = state
        unet_cfg = UNetConfig(latent_channels=latent_channels)
    return unet_cfg, unet_state


def create_dataloaders(cfg: TrainConfig):
    tf = MRItransform()
    train_dataset = MRIDataset(
        cfg.data_root, cfg.train_list, transforms=tf, expand_train_fixed3=True
    )
    eval_dataset = MRIDataset(cfg.data_root, cfg.eval_list, transforms=tf)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, eval_loader


def train_one_epoch(
    model,
    vae,
    loader,
    optimizer,
    device,
    alphas_cumprod,
    cfg: TrainConfig,
    epoch_idx: int,
):
    model.train()
    total_loss, total_samples = 0.0, 0

    for step, batch in enumerate(loader):
        x_t1ce = batch["gt"].to(device)  # T1CE
        x_dwi = batch["image"].to(device)  # DWI

        with torch.no_grad():
            mean, logvar = vae.encode(x_t1ce)
            z0 = reparameterize(mean, logvar)

        t = torch.randint(
            0,
            cfg.num_train_timesteps,
            (x_t1ce.size(0),),
            device=device,
            dtype=torch.long,
        )
        zt, noise = add_noise(z0, t, alphas_cumprod)

        pred_noise = model(latents=zt, timesteps=t, dwi_image=x_dwi)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        bs = x_t1ce.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        if (step + 1) % cfg.log_interval == 0:
            print(f"step {step + 1}: loss={loss.item():.4f}")

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, vae, loader, device, alphas_cumprod, cfg: TrainConfig):
    model.eval()
    total_loss, total_samples = 0.0, 0

    for batch in loader:
        x_t1ce = batch["gt"].to(device)
        x_dwi = batch["image"].to(device)
        mean, logvar = vae.encode(x_t1ce)
        z0 = reparameterize(mean, logvar)

        t = torch.randint(
            0,
            cfg.num_train_timesteps,
            (x_t1ce.size(0),),
            device=device,
            dtype=torch.long,
        )
        zt, noise = add_noise(z0, t, alphas_cumprod)

        pred_noise = model(latents=zt, timesteps=t, dwi_image=x_dwi)
        loss = F.mse_loss(pred_noise, noise)

        bs = x_t1ce.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / max(total_samples, 1)


def main():
    cfg = TrainConfig()
    swanlab.init(
        project="Conditional-injected-LDM-training",
        config=asdict(cfg),  # Convert dataclass to plain dict for SwanLab config
    )
    setup_seed()
    device = torch.device(cfg.device)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    betas, alphas_cumprod = get_noise_schedule(cfg)
    alphas_cumprod = alphas_cumprod.to(device)

    vae_cfg = VAEConfig(device=cfg.device)
    vae = load_vae(vae_cfg, cfg.vae_ckpt_path, device)

    resume_path = _find_resume_ckpt(cfg.ckpt_dir)
    start_epoch = 1
    best_eval = math.inf

    # Always load teacher UNet for construction; resume will overwrite weights via load_state_dict.
    teacher_unet_cfg, teacher_unet_state = load_teacher_unet(
        cfg.teacher_unet_ckpt, device, latent_channels=vae_cfg.latent_channels
    )

    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt_state = _load_ckpt_state(resume_path, device)

        unet_cfg = ckpt_state.get("unet_config") or teacher_unet_cfg
        model = Dwi_Cond_LatentUNet(
            unet_config=unet_cfg, unet_state_dict=teacher_unet_state, dwi_in_channels=1
        ).to(device)
        model.load_state_dict(ckpt_state.get("model", ckpt_state), strict=True)
        # 确保 teacher UNet 始终来自当前 cfg.teacher_unet_ckpt（避免旧 ckpt_cond 内部携带的 teacher 权重过时/较差）
        try:
            model.latent_unet.load_state_dict(teacher_unet_state, strict=True)
        except Exception as e:
            print(
                f"Warning: failed to refresh teacher UNet weights from {cfg.teacher_unet_ckpt}: {e}"
            )

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr
        )
        opt_state = ckpt_state.get("optimizer")
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr

        best_eval = float(ckpt_state.get("best_eval", math.inf))
        start_epoch = int(ckpt_state.get("epoch", 0)) + 1
    else:
        unet_cfg = teacher_unet_cfg
        model = Dwi_Cond_LatentUNet(
            unet_config=unet_cfg, unet_state_dict=teacher_unet_state, dwi_in_channels=1
        ).to(device)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr
        )

    train_loader, eval_loader = create_dataloaders(cfg)
    fixed_eval_t1ce, fixed_eval_dwi = get_fixed_eval_images(
        eval_loader.dataset, device, cfg
    )

    if start_epoch > cfg.epochs:
        print(f"start_epoch={start_epoch} > cfg.epochs={cfg.epochs}; nothing to do.")
        swanlab.finish()
        return

    for epoch in range(start_epoch, cfg.epochs + 1):
        print(f"[Epoch {epoch:03d}]")
        train_loss = train_one_epoch(
            model,
            vae,
            train_loader,
            optimizer,
            device,
            alphas_cumprod,
            cfg,
            epoch_idx=epoch - 1,
        )
        eval_loss = evaluate(model, vae, eval_loader, device, alphas_cumprod, cfg)
        print(f"train_loss={train_loss:.4f} eval_loss={eval_loss:.4f}")
        swanlab.log({"train_loss": train_loss, "eval_loss": eval_loss})

        if cfg.vis_interval_epochs > 0 and (epoch % cfg.vis_interval_epochs == 0):
            visualize_on_fixed_eval(
                model,
                vae,
                fixed_eval_t1ce,
                fixed_eval_dwi,
                alphas_cumprod,
                cfg,
                epoch=epoch,
            )

        is_best = eval_loss < best_eval
        if is_best:
            best_eval = eval_loss

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "unet_config": unet_cfg,
            "best_eval": best_eval,
        }
        torch.save(ckpt, os.path.join(cfg.ckpt_dir, "latest.pt"))

        if epoch % cfg.save_interval == 0 or is_best:
            torch.save(ckpt, os.path.join(cfg.ckpt_dir, f"cond_epoch_{epoch:03d}.pt"))
    swanlab.finish()


if __name__ == "__main__":
    main()
