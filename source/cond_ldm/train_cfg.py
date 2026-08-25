import glob
import math
import os
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from config import CondLDMTrainCFGConfig as TrainConfig
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
    candidates = sorted(glob.glob(os.path.join(ckpt_dir, "condcfg_epoch_*.pt")))
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
def get_fixed_eval_images(
    eval_dataset: torch.utils.data.Dataset, device: torch.device, cfg: TrainConfig
):
    loader = DataLoader(
        eval_dataset, batch_size=cfg.vis_num_images, shuffle=False, num_workers=0
    )
    batch = next(iter(loader))
    return batch["gt"].to(device), batch["image"].to(device)


def _denorm(v: torch.Tensor) -> torch.Tensor:
    return (v.clamp(-1, 1) + 1) * 0.5


@torch.no_grad()
def save_recon_vis_cfg(
    x_t1ce: torch.Tensor,
    x_dwi: torch.Tensor,
    zt: torch.Tensor,
    t: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    vae: VAE,
    model: Dwi_Cond_LatentUNet,
    cfg: TrainConfig,
    epoch: int,
):
    os.makedirs(cfg.recon_vis_dir, exist_ok=True)

    a_bar = alphas_cumprod[t].view(-1, 1, 1, 1)

    eps_cond = model(latents=zt, timesteps=t, dwi_image=x_dwi)

    if str(cfg.uncond_fill).lower().strip() == "noise":
        x_uncond = torch.randn_like(x_dwi)
    else:
        x_uncond = torch.zeros_like(x_dwi)

    eps_uncond = model(latents=zt, timesteps=t, dwi_image=x_uncond)
    s = float(cfg.vis_guidance_scale)
    eps_guided = eps_uncond + s * (eps_cond - eps_uncond)

    def _z0hat(eps: torch.Tensor) -> torch.Tensor:
        return (zt - torch.sqrt(1.0 - a_bar) * eps) / torch.sqrt(a_bar + 1e-8)

    x_hat_cond = vae.decode(_z0hat(eps_cond))
    x_hat_uncond = vae.decode(_z0hat(eps_uncond))
    x_hat_guided = vae.decode(_z0hat(eps_guided))

    n = (
        min(int(cfg.vis_num_images), x_t1ce.size(0))
        if cfg.vis_num_images > 0
        else min(4, x_t1ce.size(0))
    )
    grid = torch.cat(
        [
            _denorm(x_dwi[:n]),
            _denorm(x_t1ce[:n]),
            _denorm(x_hat_uncond[:n]),
            _denorm(x_hat_cond[:n]),
            _denorm(x_hat_guided[:n]),
        ],
        dim=0,
    )
    t0 = int(t[0].item()) if t.numel() > 0 else -1
    out_path = os.path.join(
        cfg.recon_vis_dir, f"e{epoch:03d}_t{t0:04d}_cfg{cfg.vis_guidance_scale:g}.png"
    )
    save_image(grid, out_path, nrow=n)


@torch.no_grad()
def visualize_on_fixed_eval_cfg(
    model: Dwi_Cond_LatentUNet,
    vae: VAE,
    x_t1ce: torch.Tensor,
    x_dwi: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    cfg: TrainConfig,
    epoch: int,
):
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

    save_recon_vis_cfg(
        x_t1ce=x_t1ce,
        x_dwi=x_dwi,
        zt=zt,
        t=t,
        alphas_cumprod=alphas_cumprod,
        vae=vae,
        model=model,
        cfg=cfg,
        epoch=epoch,
    )
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


def _make_uncond_fill(x_dwi: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    mode = str(cfg.uncond_fill).lower().strip()
    if mode == "noise":
        return torch.randn_like(x_dwi)
    if mode == "zeros":
        return torch.zeros_like(x_dwi)
    raise ValueError(f"uncond_fill 需为 'zeros' 或 'noise'，当前为 {cfg.uncond_fill!r}")


def _apply_cond_dropout(
    x_dwi: torch.Tensor, cfg: TrainConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    p = float(cfg.cond_drop_prob)
    if p <= 0.0:
        mask = torch.zeros((x_dwi.size(0),), device=x_dwi.device, dtype=torch.bool)
        return x_dwi, mask
    if p >= 1.0:
        mask = torch.ones((x_dwi.size(0),), device=x_dwi.device, dtype=torch.bool)
    else:
        mask = torch.rand((x_dwi.size(0),), device=x_dwi.device) < p

    if not mask.any():
        return x_dwi, mask

    x_out = x_dwi.clone()
    x_fill = _make_uncond_fill(x_dwi, cfg)
    x_out[mask] = x_fill[mask]
    return x_out, mask


def _param_groups_for_adamw(model: torch.nn.Module, cfg: TrainConfig):
    wd = float(cfg.weight_decay)
    gate_lr = float(cfg.lr) * float(cfg.gate_lr_mult)
    teacher_lr = float(cfg.lr) * float(cfg.teacher_lr_mult)
    teacher_wd = float(cfg.teacher_weight_decay)

    no_decay_keywords = (
        ".bias",
        "gate",
        "norm",
        "groupnorm",
    )

    decay_params = []
    no_decay_params = []
    gate_params = []

    teacher_decay_params = []
    teacher_no_decay_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if "gate" in lname:
            gate_params.append(p)
        elif lname.startswith("latent_unet."):
            if any(k in lname for k in no_decay_keywords):
                teacher_no_decay_params.append(p)
            else:
                teacher_decay_params.append(p)
        elif any(k in lname for k in no_decay_keywords):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    groups = []
    if decay_params:
        groups.append(
            {
                "name": "decay",
                "params": decay_params,
                "lr": float(cfg.lr),
                "weight_decay": wd,
            }
        )
    if no_decay_params:
        groups.append(
            {
                "name": "no_decay",
                "params": no_decay_params,
                "lr": float(cfg.lr),
                "weight_decay": 0.0,
            }
        )
    if teacher_decay_params:
        groups.append(
            {
                "name": "teacher_decay",
                "params": teacher_decay_params,
                "lr": teacher_lr,
                "weight_decay": teacher_wd,
            }
        )
    if teacher_no_decay_params:
        groups.append(
            {
                "name": "teacher_no_decay",
                "params": teacher_no_decay_params,
                "lr": teacher_lr,
                "weight_decay": 0.0,
            }
        )
    if gate_params:
        groups.append(
            {"name": "gate", "params": gate_params, "lr": gate_lr, "weight_decay": 0.0}
        )
    return groups


def _gate_stats(model: torch.nn.Module) -> Dict[str, float]:
    vals = []
    for n, p in model.named_parameters():
        if "gate" in n.lower() and p.requires_grad:
            vals.append(p.detach().float().view(-1))
    if not vals:
        return {"gate_mean": 0.0, "gate_abs_mean": 0.0, "gate_max": 0.0}
    v = torch.cat(vals, dim=0)
    return {
        "gate_mean": float(v.mean().item()),
        "gate_abs_mean": float(v.abs().mean().item()),
        "gate_max": float(v.max().item()),
    }


def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(requires_grad)


def apply_teacher_unfreeze(
    model: Dwi_Cond_LatentUNet, cfg: TrainConfig
) -> Dict[str, int]:
    """
    Selectively unfreeze parts of the teacher UNet (module1) with a lower LR.

    Strategy (defaults):
    - unfreeze mid_block (global structure)
    - unfreeze last N up_blocks (detail refinement)
    - unfreeze conv_norm_out/conv_out (final mapping)
    """
    # baseline: freeze all teacher params
    _set_requires_grad(model.latent_unet, False)

    if not bool(cfg.unfreeze_teacher):
        return {"teacher_trainable_params": 0, "teacher_trainable_tensors": 0}

    unet = getattr(model.latent_unet, "unet", None)
    if unet is None:
        raise RuntimeError("model.latent_unet.unet 不存在，无法解冻 teacher。")

    if (
        bool(cfg.unfreeze_teacher_mid)
        and hasattr(unet, "mid_block")
        and unet.mid_block is not None
    ):
        _set_requires_grad(unet.mid_block, True)

    n_up = int(cfg.unfreeze_teacher_up_blocks)
    if n_up > 0 and hasattr(unet, "up_blocks"):
        up_blocks = list(unet.up_blocks)
        if up_blocks:
            for blk in up_blocks[-min(n_up, len(up_blocks)) :]:
                _set_requires_grad(blk, True)

    if bool(cfg.unfreeze_teacher_out):
        for name in ("conv_norm_out", "conv_out"):
            if hasattr(unet, name) and getattr(unet, name) is not None:
                _set_requires_grad(getattr(unet, name), True)

    tensors = 0
    params = 0
    for p in model.latent_unet.parameters():
        if p.requires_grad:
            tensors += 1
            params += int(p.numel())
    return {"teacher_trainable_params": params, "teacher_trainable_tensors": tensors}


def train_one_epoch(
    model: Dwi_Cond_LatentUNet,
    vae: VAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alphas_cumprod: torch.Tensor,
    cfg: TrainConfig,
):
    model.train()
    total_loss, total_samples = 0.0, 0

    for step, batch in enumerate(loader):
        x_t1ce = batch["gt"].to(device)
        x_dwi = batch["image"].to(device)

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

        x_dwi_drop, drop_mask = _apply_cond_dropout(x_dwi, cfg)
        pred_noise = model(latents=zt, timesteps=t, dwi_image=x_dwi_drop)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        bs = x_t1ce.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        if (step + 1) % cfg.log_interval == 0:
            dropped = float(drop_mask.float().mean().item())
            gstats = _gate_stats(model)
            print(
                f"step {step + 1}: loss={loss.item():.4f} drop={dropped:.2f} "
                f"gate_mean={gstats['gate_mean']:.4g} gate_abs={gstats['gate_abs_mean']:.4g} gate_max={gstats['gate_max']:.4g}"
            )

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: Dwi_Cond_LatentUNet,
    vae: VAE,
    loader: DataLoader,
    device: torch.device,
    alphas_cumprod: torch.Tensor,
    cfg: TrainConfig,
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
        loss = F.mse_loss(pred_noise, noise)

        bs = x_t1ce.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / max(total_samples, 1)


def main():
    cfg = TrainConfig()
    setup_seed()
    device = torch.device(cfg.device)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    swanlab = None
    if bool(cfg.swanlab_enabled):
        try:
            import swanlab as _swanlab

            swanlab = _swanlab
            swanlab.init(project=str(cfg.swanlab_project), config=asdict(cfg))
        except Exception as e:
            print(f"Warning: SwanLab disabled due to import/init failure: {e}")
            swanlab = None

    _betas, alphas_cumprod = get_noise_schedule(cfg)
    alphas_cumprod = alphas_cumprod.to(device)

    vae_cfg = VAEConfig(device=cfg.device)
    vae = load_vae(vae_cfg, cfg.vae_ckpt_path, device)

    resume_path = _find_resume_ckpt(cfg.ckpt_dir)
    start_epoch = 1
    best_eval = math.inf

    teacher_unet_cfg, teacher_unet_state = load_teacher_unet(
        cfg.teacher_unet_ckpt, device, latent_channels=vae_cfg.latent_channels
    )

    stage_map = {str(k): int(v) for (k, v) in tuple(cfg.stage_cond_feat_indices)}

    def _build_model(unet_cfg: UNetConfig) -> Dwi_Cond_LatentUNet:
        return Dwi_Cond_LatentUNet(
            unet_config=unet_cfg,
            unet_state_dict=teacher_unet_state,
            dwi_in_channels=1,
            cross_attn_heads=int(cfg.cross_attn_heads),
            cross_attn_dim_head=int(cfg.cross_attn_dim_head),
            cross_attn_feat_indices=tuple(cfg.cross_attn_feat_indices),
            cond_proj_channels=cfg.cond_proj_channels,
            gate_init=float(cfg.gate_init),
            stage_cond_feat_indices=stage_map,
        )

    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt_state = _load_ckpt_state(resume_path, device)
        unet_cfg = ckpt_state.get("unet_config") or teacher_unet_cfg

        model = _build_model(unet_cfg).to(device)
        model.load_state_dict(ckpt_state.get("model", ckpt_state), strict=True)
        try:
            model.latent_unet.load_state_dict(teacher_unet_state, strict=True)
        except Exception as e:
            print(
                f"Warning: failed to refresh teacher UNet weights from {cfg.teacher_unet_ckpt}: {e}"
            )

        teacher_stats = apply_teacher_unfreeze(model, cfg)
        print(
            f"teacher_unfreeze={bool(cfg.unfreeze_teacher)} "
            f"teacher_trainable_tensors={teacher_stats['teacher_trainable_tensors']} "
            f"teacher_trainable_params={teacher_stats['teacher_trainable_params']}"
        )

        optimizer = torch.optim.AdamW(_param_groups_for_adamw(model, cfg))
        opt_state = ckpt_state.get("optimizer")
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
            except Exception as e:
                print(
                    f"Warning: failed to load optimizer state (param_groups changed?): {e}"
                )
            for group in optimizer.param_groups:
                name = str(group.get("name", "")).lower()
                if name == "gate":
                    group["lr"] = float(cfg.lr) * float(cfg.gate_lr_mult)
                    group["weight_decay"] = 0.0
                elif name == "no_decay":
                    group["lr"] = float(cfg.lr)
                    group["weight_decay"] = 0.0
                elif name == "teacher_no_decay":
                    group["lr"] = float(cfg.lr) * float(cfg.teacher_lr_mult)
                    group["weight_decay"] = 0.0
                elif name == "teacher_decay":
                    group["lr"] = float(cfg.lr) * float(cfg.teacher_lr_mult)
                    group["weight_decay"] = float(cfg.teacher_weight_decay)
                else:
                    group["lr"] = float(cfg.lr)
                    group["weight_decay"] = float(cfg.weight_decay)

        best_eval = float(ckpt_state.get("best_eval", math.inf))
        start_epoch = int(ckpt_state.get("epoch", 0)) + 1
    else:
        unet_cfg = teacher_unet_cfg
        model = _build_model(unet_cfg).to(device)
        teacher_stats = apply_teacher_unfreeze(model, cfg)
        print(
            f"teacher_unfreeze={bool(cfg.unfreeze_teacher)} "
            f"teacher_trainable_tensors={teacher_stats['teacher_trainable_tensors']} "
            f"teacher_trainable_params={teacher_stats['teacher_trainable_params']}"
        )
        optimizer = torch.optim.AdamW(_param_groups_for_adamw(model, cfg))

    train_loader, eval_loader = create_dataloaders(cfg)
    fixed_eval_t1ce, fixed_eval_dwi = get_fixed_eval_images(
        eval_loader.dataset, device, cfg
    )

    if start_epoch > cfg.epochs:
        print(f"start_epoch={start_epoch} > cfg.epochs={cfg.epochs}; nothing to do.")
        if swanlab is not None:
            swanlab.finish()
        return

    for epoch in range(start_epoch, cfg.epochs + 1):
        print(f"[Epoch {epoch:03d}]")
        train_loss = train_one_epoch(
            model, vae, train_loader, optimizer, device, alphas_cumprod, cfg
        )
        eval_loss = evaluate(model, vae, eval_loader, device, alphas_cumprod, cfg)
        gstats = _gate_stats(model)
        print(
            f"train_loss={train_loss:.4f} eval_loss={eval_loss:.4f} "
            f"gate_mean={gstats['gate_mean']:.4g} gate_abs={gstats['gate_abs_mean']:.4g} gate_max={gstats['gate_max']:.4g}"
        )

        if swanlab is not None:
            swanlab.log({"train_loss": train_loss, "eval_loss": eval_loss, **gstats})

        if cfg.vis_interval_epochs > 0 and (epoch % cfg.vis_interval_epochs == 0):
            visualize_on_fixed_eval_cfg(
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
            torch.save(
                ckpt, os.path.join(cfg.ckpt_dir, f"condcfg_epoch_{epoch:03d}.pt")
            )

    if swanlab is not None:
        swanlab.finish()


if __name__ == "__main__":
    main()
