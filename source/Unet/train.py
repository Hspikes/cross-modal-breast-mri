import glob
import math
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from config import UNetConfig, VAEConfig
from config import UnetTrainConfig as TrainConfig

from ..data_pre_processing.dataset import MRIDataset, MRItransform
from ..VAE.model import VAE, reparameterize
from .model import LatentUNet


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
    candidates = sorted(glob.glob(os.path.join(ckpt_dir, "unet_epoch_*.pt")))
    return candidates[-1] if candidates else None


def _load_unet_ckpt_state(ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        return state
    raise ValueError(f"Unexpected checkpoint format at {ckpt_path}")


def _parse_unet_ckpt_state(
    state: Dict[str, Any],
    fallback_cfg: UNetConfig,
) -> Tuple[Dict[str, Any], int, float, UNetConfig, Optional[Dict[str, Any]]]:
    model_state = state.get("model", state)
    epoch = int(state.get("epoch", 0))
    best_eval = float(state.get("best_eval", math.inf))
    unet_cfg = state.get("config") or fallback_cfg
    opt_state = state.get("optimizer")
    return model_state, epoch, best_eval, unet_cfg, opt_state


def get_noise_schedule(config: TrainConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    betas = torch.linspace(
        config.beta_start, config.beta_end, config.num_train_timesteps
    )
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas_cumprod


def add_noise(latents: torch.Tensor, t: torch.Tensor, alphas_cumprod: torch.Tensor):
    # t shape [B], latents [B,C,H,W]
    a_bar = alphas_cumprod[t].view(-1, 1, 1, 1)
    noise = torch.randn_like(latents)
    noisy = torch.sqrt(a_bar) * latents + torch.sqrt(1.0 - a_bar) * noise
    return noisy, noise


@torch.no_grad()
def save_recon_vis(
    x: torch.Tensor,
    zt: torch.Tensor,
    pred_noise: torch.Tensor,
    t: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    vae: VAE,
    cfg: TrainConfig,
    epoch: int,
):
    """
    将当前 batch 的预测去噪结果可视化：原图 vs 解码后的 z0_hat。
    """
    os.makedirs(cfg.recon_vis_dir, exist_ok=True)
    # 估计 z0： z0_hat = (zt - sqrt(1-a_bar)*eps) / sqrt(a_bar)
    a_bar = alphas_cumprod[t].view(-1, 1, 1, 1)
    z0_hat = (zt - torch.sqrt(1.0 - a_bar) * pred_noise) / torch.sqrt(a_bar + 1e-8)
    x_hat = vae.decode(z0_hat)

    def _denorm(v):
        return (v.clamp(-1, 1) + 1) * 0.5

    n = (
        min(int(cfg.vis_num_images), x.size(0))
        if cfg.vis_num_images > 0
        else min(4, x.size(0))
    )
    grid = torch.cat([_denorm(x[:n]), _denorm(x_hat[:n])], dim=0)
    t0 = int(t[0].item()) if t.numel() > 0 else -1
    save_image(
        grid,
        os.path.join(cfg.recon_vis_dir, f"e{epoch:03d}_t{t0:04d}.png"),
        nrow=n,
    )


@torch.no_grad()
def get_fixed_eval_images(
    eval_dataset: torch.utils.data.Dataset, device: torch.device, cfg: TrainConfig
) -> torch.Tensor:
    """
    固定取验证集开头的若干张图片，用于整个训练过程中的可视化对比。
    """
    loader = DataLoader(
        eval_dataset, batch_size=cfg.vis_num_images, shuffle=False, num_workers=0
    )
    batch = next(iter(loader))
    return batch["gt"].to(device)


@torch.no_grad()
def visualize_on_fixed_eval(
    unet: LatentUNet,
    vae: VAE,
    x: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    cfg: TrainConfig,
    epoch: int,
):
    """
    固定使用验证集样例、固定扩散时间步与固定噪声，便于跨训练阶段对比可视化效果。
    """
    unet.eval()

    mean, _logvar = vae.encode(x)
    z0 = mean

    t_fixed = int(max(0, min(cfg.num_train_timesteps - 1, cfg.vis_timestep)))
    t = torch.full((x.size(0),), t_fixed, device=x.device, dtype=torch.long)

    gen = torch.Generator()
    gen.manual_seed(int(cfg.vis_noise_seed))
    noise = torch.randn(z0.shape, device="cpu", generator=gen, dtype=z0.dtype).to(
        z0.device
    )

    a_bar = alphas_cumprod[t].view(-1, 1, 1, 1)
    zt = torch.sqrt(a_bar) * z0 + torch.sqrt(1.0 - a_bar) * noise

    pred_noise = unet(zt, t)
    save_recon_vis(x, zt, pred_noise, t, alphas_cumprod, vae, cfg, epoch=epoch)
    unet.train()


def load_vae(config: VAEConfig, ckpt_path: str, device: torch.device) -> VAE:
    vae = VAE(config)
    state = torch.load(ckpt_path, map_location=device)
    # 兼容两种保存方式：直接 state_dict 或包含 model 键
    if isinstance(state, dict) and "model" in state:
        vae.load_state_dict(state["model"])
    else:
        vae.load_state_dict(state)
    vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def create_dataloaders(cfg: TrainConfig):
    tf = MRItransform()
    train_dataset = MRIDataset(
        data_folder=cfg.data_root,
        sequence_list_txt=cfg.train_list,
        transforms=tf,
        expand_train_fixed3=True,
    )
    eval_dataset = MRIDataset(
        data_folder=cfg.data_root,
        sequence_list_txt=cfg.eval_list,
        transforms=tf,
    )
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
    unet: LatentUNet,
    vae: VAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alphas_cumprod: torch.Tensor,
    cfg: TrainConfig,
    epoch: int,
):
    unet.train()
    total_loss = 0.0
    total_samples = 0

    for step, batch in enumerate(dataloader):
        x = batch["gt"].to(device)
        with torch.no_grad():
            mean, logvar = vae.encode(x)
            z0 = reparameterize(mean, logvar)

        t = torch.randint(
            low=0,
            high=cfg.num_train_timesteps,
            size=(x.size(0),),
            device=device,
            dtype=torch.long,
        )
        zt, noise = add_noise(z0, t, alphas_cumprod)

        pred_noise = unet(zt, t)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), cfg.grad_clip)
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        if (step + 1) % cfg.log_interval == 0:
            print(f"step {step + 1}: loss={loss.item():.4f}")

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    unet: LatentUNet,
    vae: VAE,
    dataloader: DataLoader,
    device: torch.device,
    alphas_cumprod: torch.Tensor,
    cfg: TrainConfig,
):
    unet.eval()
    total_loss = 0.0
    steps = 0
    total_samples = 0

    for batch in dataloader:
        x = batch["gt"].to(device)
        mean, logvar = vae.encode(x)
        z0 = reparameterize(mean, logvar)

        t = torch.randint(
            low=0,
            high=cfg.num_train_timesteps,
            size=(x.size(0),),
            device=device,
            dtype=torch.long,
        )
        zt, noise = add_noise(z0, t, alphas_cumprod)
        pred_noise = unet(zt, t)
        loss = F.mse_loss(pred_noise, noise)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        steps += 1

    return total_loss / max(total_samples, 1)


def main():
    cfg = TrainConfig()
    device = torch.device(cfg.device)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    betas, alphas_cumprod = get_noise_schedule(cfg)
    betas = betas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)

    # 初始化模型
    vae_cfg = VAEConfig(device=cfg.device)
    vae = load_vae(vae_cfg, cfg.vae_ckpt_path, device)

    resume_path = _find_resume_ckpt(cfg.ckpt_dir)
    start_epoch = 1
    best_eval = math.inf

    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt_state = _load_unet_ckpt_state(resume_path, device)
        fallback_cfg = UNetConfig(latent_channels=vae_cfg.latent_channels)
        model_state, last_epoch, best_eval, unet_cfg, opt_state = (
            _parse_unet_ckpt_state(ckpt_state, fallback_cfg=fallback_cfg)
        )
        if unet_cfg.latent_channels != vae_cfg.latent_channels:
            print(
                "Warning: checkpoint UNetConfig.latent_channels "
                f"({unet_cfg.latent_channels}) != VAE latent_channels ({vae_cfg.latent_channels})."
            )
        unet = LatentUNet(unet_cfg).to(device)
        unet.load_state_dict(model_state, strict=True)
        optimizer = torch.optim.AdamW(unet.parameters(), lr=cfg.lr)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr
        start_epoch = int(last_epoch) + 1
    else:
        unet_cfg = UNetConfig(latent_channels=vae_cfg.latent_channels)
        unet = LatentUNet(unet_cfg).to(device)
        optimizer = torch.optim.AdamW(unet.parameters(), lr=cfg.lr)

    train_loader, eval_loader = create_dataloaders(cfg)
    fixed_eval_x = get_fixed_eval_images(eval_loader.dataset, device, cfg)

    if start_epoch > cfg.epochs:
        print(f"start_epoch={start_epoch} > cfg.epochs={cfg.epochs}; nothing to do.")
        return

    for epoch in range(start_epoch, cfg.epochs + 1):
        print(f"[Epoch {epoch:03d}]")
        train_loss = train_one_epoch(
            unet, vae, train_loader, optimizer, device, alphas_cumprod, cfg, epoch
        )
        eval_loss = evaluate(unet, vae, eval_loader, device, alphas_cumprod, cfg)

        if cfg.vis_interval_epochs > 0 and (epoch % cfg.vis_interval_epochs == 0):
            visualize_on_fixed_eval(
                unet, vae, fixed_eval_x, alphas_cumprod, cfg, epoch=epoch
            )

        print(f"train_loss={train_loss:.4f} eval_loss={eval_loss:.4f}")

        is_best = eval_loss < best_eval
        if is_best:
            best_eval = eval_loss

        ckpt = {
            "epoch": epoch,
            "model": unet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": unet_cfg,
            "best_eval": best_eval,
        }
        torch.save(ckpt, os.path.join(cfg.ckpt_dir, "latest.pt"))

        if epoch % cfg.save_interval == 0 or is_best:
            torch.save(ckpt, os.path.join(cfg.ckpt_dir, f"unet_epoch_{epoch:03d}.pt"))


if __name__ == "__main__":
    setup_seed()
    main()
