# train_distill.py
import math
import os
from dataclasses import asdict
from typing import List, Tuple

import swanlab
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from config import DistillConfig, UNetConfig, VAEConfig

from ..cond_ldm.model import Dwi_Cond_LatentUNet
from ..data_pre_processing.dataset import MRIDataset, MRItransform
from ..VAE.model import VAE, reparameterize


def setup_seed(seed: int = 42):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_noise_schedule(cfg: DistillConfig) -> Tuple[torch.Tensor, torch.Tensor]:
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
    cfg: DistillConfig,
    epoch: int,
):
    os.makedirs(cfg.recon_vis_dir, exist_ok=True)
    a_bar = alphas_cumprod[t].view(-1, 1, 1, 1)
    z0_hat = (zt - torch.sqrt(1.0 - a_bar) * pred_noise) / torch.sqrt(a_bar + 1e-8)
    x_hat = vae.decode(z0_hat)

    def _denorm(v):
        return (v.clamp(-1, 1) + 1) * 0.5

    n = min(cfg.vis_max, x_t1ce.size(0))
    grid = torch.cat([_denorm(x_t1ce[:n]), _denorm(x_hat[:n])], dim=0)
    t0 = int(t[0].item()) if t.numel() > 0 else -1
    save_image(
        grid, os.path.join(cfg.recon_vis_dir, f"e{epoch:03d}_t{t0:04d}.png"), nrow=n
    )


@torch.no_grad()
def get_fixed_eval_images(
    eval_dataset: torch.utils.data.Dataset, device: torch.device, cfg: DistillConfig
):
    """
    固定取验证集开头的若干张图片（T1CE + DWI），用于整个训练过程中的可视化对比。
    """
    loader = DataLoader(
        eval_dataset, batch_size=cfg.vis_max, shuffle=False, num_workers=0
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
    cfg: DistillConfig,
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


def create_dataloaders(cfg: DistillConfig):
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


@torch.no_grad()
def extract_vae_encoder_feats(vae: VAE, x: torch.Tensor) -> List[torch.Tensor]:
    feats: List[torch.Tensor] = []
    hooks = []

    for blk in vae.encoder.down_blocks:

        def _make_hook():
            def _hook(module, inp, out):
                h = out[0] if isinstance(out, (tuple, list)) else out
                feats.append(h)

            return _hook

        hooks.append(blk.register_forward_hook(_make_hook()))

    _ = vae.encoder(x)

    for h in hooks:
        h.remove()

    return feats


def freeze_all(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_module(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def train_one_epoch_distill(
    model: Dwi_Cond_LatentUNet,
    vae: VAE,
    loader,
    optimizer,
    device,
    alphas_cumprod,
    cfg: DistillConfig,
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
        loss_cond = F.mse_loss(pred_noise, noise)

        if cfg.use_vae_adapter_branch:
            unet_feats, vae_like_feats = model.dwi_encoder(
                x_dwi, t, return_vae_feats=True
            )

            with torch.no_grad():
                teacher_feats = extract_vae_encoder_feats(vae, x_t1ce)

            L = min(len(vae_like_feats), len(teacher_feats))
            loss_distill = 0.0
            for i in range(L):
                loss_distill = loss_distill + F.l1_loss(
                    vae_like_feats[i], teacher_feats[i]
                )
        else:
            loss_distill = torch.tensor(0.0, device=device)

        loss = loss_cond + cfg.distill_weight * loss_distill

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        bs = x_t1ce.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        if (step + 1) % cfg.log_interval == 0:
            print(
                f"step {step + 1}: "
                f"loss={loss.item():.4f} "
                f"cond={loss_cond.item():.4f} "
                f"distill={float(loss_distill):.4f}"
            )

        # 可视化频率沿用模块二风格

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_distill(
    model: Dwi_Cond_LatentUNet,
    vae: VAE,
    loader,
    device,
    alphas_cumprod,
    cfg: DistillConfig,
):
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
        loss_cond = F.mse_loss(pred_noise, noise)

        # eval 时也可以顺便监控 distill
        if cfg.use_vae_adapter_branch:
            _, vae_like_feats = model.dwi_encoder(x_dwi, t, return_vae_feats=True)
            teacher_feats = extract_vae_encoder_feats(vae, x_t1ce)
            L = min(len(vae_like_feats), len(teacher_feats))
            loss_distill = 0.0
            for i in range(L):
                loss_distill = loss_distill + F.l1_loss(
                    vae_like_feats[i], teacher_feats[i]
                )
        else:
            loss_distill = torch.tensor(0.0, device=device)

        loss = loss_cond + cfg.distill_weight * loss_distill

        bs = x_t1ce.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / max(total_samples, 1)


def main():
    cfg = DistillConfig()
    swanlab.init(project="Module3-Distillation", config=asdict(cfg))
    setup_seed()
    device = torch.device(cfg.device)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    _, alphas_cumprod = get_noise_schedule(cfg)
    alphas_cumprod = alphas_cumprod.to(device)

    vae_cfg = VAEConfig(device=cfg.device)
    vae = load_vae(vae_cfg, cfg.vae_ckpt_path, device)

    unet_cfg, unet_state = load_teacher_unet(
        cfg.teacher_unet_ckpt, device, latent_channels=vae_cfg.latent_channels
    )

    model = Dwi_Cond_LatentUNet(
        unet_config=unet_cfg,
        unet_state_dict=unet_state,
        dwi_in_channels=1,
        dwi_encoder_kwargs={
            "enable_vae_adapter": cfg.use_vae_adapter_branch,
        },
    ).to(device)

    if cfg.module2_ckpt_path and os.path.exists(cfg.module2_ckpt_path):
        state = torch.load(
            cfg.module2_ckpt_path,
            map_location=device,
        )
        module2_sd = (
            state["model"] if isinstance(state, dict) and "model" in state else state
        )
        model.load_state_dict(module2_sd, strict=False)
        print(f"[Init] loaded module2 ckpt from {cfg.module2_ckpt_path}")
    else:
        print(
            "[Init] WARNING: module2_ckpt_path not found, training will start from scratch."
        )

    freeze_all(model)
    unfreeze_module(model.dwi_encoder)

    optimizer = torch.optim.AdamW(
        [p for p in model.dwi_encoder.parameters() if p.requires_grad], lr=cfg.lr
    )

    train_loader, eval_loader = create_dataloaders(cfg)
    fixed_eval_t1ce, fixed_eval_dwi = get_fixed_eval_images(
        eval_loader.dataset, device, cfg
    )

    best_eval = math.inf
    for epoch in range(1, cfg.epochs + 1):
        print(f"[Epoch {epoch:03d}]")

        train_loss = train_one_epoch_distill(
            model,
            vae,
            train_loader,
            optimizer,
            device,
            alphas_cumprod,
            cfg,
            epoch_idx=epoch - 1,
        )
        eval_loss = evaluate_distill(
            model, vae, eval_loader, device, alphas_cumprod, cfg
        )

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

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        }
        torch.save(ckpt, os.path.join(cfg.ckpt_dir, "latest.pt"))

        if epoch % cfg.save_interval == 0 or eval_loss < best_eval:
            torch.save(
                ckpt, os.path.join(cfg.ckpt_dir, f"distill_epoch_{epoch:03d}.pt")
            )
            best_eval = min(best_eval, eval_loss)

    swanlab.finish()


if __name__ == "__main__":
    main()
