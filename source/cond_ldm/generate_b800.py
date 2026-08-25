from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from config import CondLDMGenConfig, UNetConfig, VAEConfig
from config import CondLDMInferConfig as InferConfig

from ..data_pre_processing.dataset import MRIDataset, MRItransform
from ..Unet.model import LatentUNet
from .infer import (
    get_noise_schedule,
    load_cond_model,
    load_teacher_unet_config_and_state,
    load_vae,
)


def setup_seed(seed: int = 42):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) * 0.5


def _mean_std(x: torch.Tensor) -> Tuple[float, float]:
    x = x.detach()
    return float(x.mean().item()), float(x.std(unbiased=False).item())


def _mean_std_min_max(x: torch.Tensor) -> Tuple[float, float, float, float]:
    x = x.detach()
    return (
        float(x.mean().item()),
        float(x.std(unbiased=False).item()),
        float(x.min().item()),
        float(x.max().item()),
    )


def _clamp_ratios_to_unit(x: torch.Tensor) -> Tuple[float, float]:
    x = x.detach()
    below = float((x < -1).float().mean().item())
    above = float((x > 1).float().mean().item())
    return below, above


def _parse_id_from_path(path: str) -> str:
    parent = os.path.basename(os.path.dirname(path))
    if parent.endswith("_b800"):
        return parent[: -len("_b800")]
    return parent


def _pick_indices_from_distinct_ids(
    dataset: MRIDataset,
    num_ids: int,
    seed: int,
) -> List[Tuple[str, int]]:
    id_to_first_index: Dict[str, int] = {}
    for i, p in enumerate(getattr(dataset, "image_path_list", [])):
        id_ = _parse_id_from_path(str(p))
        if id_ not in id_to_first_index:
            id_to_first_index[id_] = int(i)

    ids = sorted(id_to_first_index.keys())
    if not ids:
        raise RuntimeError("无法从数据集中解析到任何 id（image_path_list 为空？）")
    if num_ids <= 0:
        raise ValueError(f"num_ids 需为正数，当前为 {num_ids}")
    if num_ids > len(ids):
        print(f"Warning: num_ids={num_ids} > available ids={len(ids)}; clamping.")
        num_ids = len(ids)

    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(len(ids), generator=g)[:num_ids].tolist()
    return [(ids[i], id_to_first_index[ids[i]]) for i in perm]


def _load_teacher_model(
    *,
    cfg: InferConfig,
    vae_cfg: VAEConfig,
    device: torch.device,
) -> LatentUNet:
    teacher_default_cfg = UNetConfig(latent_channels=vae_cfg.latent_channels)
    unet_cfg, unet_state = load_teacher_unet_config_and_state(
        teacher_default_cfg, cfg.teacher_unet_ckpt, device
    )
    teacher = LatentUNet(unet_cfg).to(device)
    teacher.load_state_dict(unet_state, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def _make_ddim_timesteps(
    num_train_timesteps: int, num_inference_steps: int
) -> torch.Tensor:
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps 需为正数。")
    # integer timesteps, descending
    timesteps = torch.linspace(
        num_train_timesteps - 1, 0, steps=int(num_inference_steps)
    )
    return torch.round(timesteps).to(torch.long)


@torch.no_grad()
def _ddim_sample(
    *,
    pred_eps_fn,
    z: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    eta: float = 0.0,
) -> torch.Tensor:
    if eta != 0.0:
        raise NotImplementedError("当前实现仅支持 eta=0 的确定性 DDIM。")
    if timesteps.numel() < 2:
        raise ValueError("timesteps 至少包含 2 个点。")

    for i in range(timesteps.numel() - 1):
        t = int(timesteps[i].item())
        t_prev = int(timesteps[i + 1].item())
        if t_prev >= t:
            continue

        t_batch = torch.full((z.size(0),), t, device=z.device, dtype=torch.long)
        t_prev_batch = torch.full(
            (z.size(0),), t_prev, device=z.device, dtype=torch.long
        )

        a_bar_t = alphas_cumprod[t_batch].view(-1, 1, 1, 1)
        a_bar_prev = alphas_cumprod[t_prev_batch].view(-1, 1, 1, 1)

        eps = pred_eps_fn(z, t_batch)
        x0_hat = (z - torch.sqrt(1.0 - a_bar_t) * eps) / torch.sqrt(a_bar_t + 1e-8)
        z = torch.sqrt(a_bar_prev) * x0_hat + torch.sqrt(1.0 - a_bar_prev) * eps

    return z


@torch.no_grad()
def _dpmpp_sample(
    *,
    pred_eps_fn,
    z: torch.Tensor,
    num_inference_steps: int,
    num_train_timesteps: int,
    beta_start: float,
    beta_end: float,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """
    Prefer DPMSolver++ (diffusers) when available; fall back to DDIM if not.

    Note: our model is trained with discrete integer timesteps, so we round scheduler timesteps
    for model conditioning, while keeping scheduler internal time for updates.
    """
    try:
        from diffusers import DPMSolverMultistepScheduler
    except Exception as e:  # pragma: no cover
        print(
            f"Warning: failed to import DPMSolverMultistepScheduler ({e}); falling back to DDIM."
        )
        timesteps = _make_ddim_timesteps(num_train_timesteps, num_inference_steps).to(
            z.device
        )
        return _ddim_sample(
            pred_eps_fn=pred_eps_fn,
            z=z,
            timesteps=timesteps,
            alphas_cumprod=alphas_cumprod,
            eta=0.0,
        )

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=int(num_train_timesteps),
        beta_start=float(beta_start),
        beta_end=float(beta_end),
        beta_schedule="linear",
        algorithm_type="dpmsolver++",
        solver_order=2,
        prediction_type="epsilon",
    )
    try:
        scheduler = DPMSolverMultistepScheduler.from_config(
            scheduler.config, use_karras_sigmas=True
        )
    except Exception:
        pass

    scheduler.set_timesteps(int(num_inference_steps), device=z.device)
    timesteps = scheduler.timesteps

    for t in timesteps:
        t_val = float(t) if hasattr(t, "item") else float(t)
        t_model = int(round(t_val))
        t_batch = torch.full((z.size(0),), t_model, device=z.device, dtype=torch.long)
        z_in = (
            scheduler.scale_model_input(z, t)
            if hasattr(scheduler, "scale_model_input")
            else z
        )
        eps = pred_eps_fn(z_in, t_batch)
        out = scheduler.step(eps, t, z)
        z = out.prev_sample

    return z


@torch.no_grad()
def main() -> None:
    infer_cfg = InferConfig()
    gen_cfg = CondLDMGenConfig()

    setup_seed(gen_cfg.seed)
    device = torch.device(infer_cfg.device)
    os.makedirs(gen_cfg.out_dir, exist_ok=True)

    _betas, alphas_cumprod = get_noise_schedule(infer_cfg)
    alphas_cumprod = alphas_cumprod.to(device)

    max_t = int(alphas_cumprod.numel()) - 1
    start_t = int(gen_cfg.start_t)
    if start_t < 1 or start_t > max_t:
        raise ValueError(f"start_t={start_t} 超出范围 [1, {max_t}]")

    init_mode = str(gen_cfg.init_noise_scale_mode).lower().strip()
    if init_mode not in {"unit", "match_t"}:
        raise ValueError(
            f"init_noise_scale_mode 需为 'unit' 或 'match_t'，当前为 {gen_cfg.init_noise_scale_mode!r}"
        )

    vae_cfg = VAEConfig(device=infer_cfg.device)
    vae = load_vae(vae_cfg, infer_cfg.vae_ckpt_path, device)
    cond = load_cond_model(cfg=infer_cfg, vae_cfg=vae_cfg, device=device)
    teacher = _load_teacher_model(cfg=infer_cfg, vae_cfg=vae_cfg, device=device)

    # Determine latent spatial shape from VAE (no need to use GT).
    dummy = torch.zeros(
        (1, vae_cfg.img_channels, vae_cfg.img_size, vae_cfg.img_size), device=device
    )
    latent_mean, _latent_logvar = vae.encode(dummy)
    latent_shape = latent_mean.shape[1:]  # [C, H', W']

    tf = MRItransform()
    eval_dataset = MRIDataset(
        data_folder=infer_cfg.data_root,
        sequence_list_txt=infer_cfg.eval_list,
        transforms=tf,
    )
    chosen = _pick_indices_from_distinct_ids(
        eval_dataset, num_ids=gen_cfg.num_ids, seed=gen_cfg.seed
    )

    if gen_cfg.sampler.lower() == "ddim":
        # truncated DDIM: start from start_t rather than max_t
        timesteps = torch.linspace(
            float(start_t), 0.0, steps=int(gen_cfg.num_inference_steps)
        )
        timesteps = torch.round(timesteps).to(torch.long)
        timesteps = torch.unique_consecutive(timesteps).to(device)
        if timesteps.numel() < 2:
            raise ValueError(
                "num_inference_steps 太小或 start_t 太小，导致 timesteps < 2。"
            )
        if int(timesteps[0].item()) != start_t:
            timesteps[0] = start_t
        if int(timesteps[-1].item()) != 0:
            timesteps = torch.cat(
                [timesteps, torch.zeros(1, device=device, dtype=torch.long)], dim=0
            )

    stats_lines = [
        "id,index,"
        "z0_mean_mean,z0_mean_std,z0_mean_min,z0_mean_max,"
        "z0_sample_mean,z0_sample_std,z0_sample_min,z0_sample_max,"
        "z0hat_teacher_mean,z0hat_teacher_std,z0hat_teacher_min,z0hat_teacher_max,"
        "mse_z0mean_teacher,mse_z0sample_teacher,"
        "xdec_teacher_mean,xdec_teacher_std,xdec_teacher_min,xdec_teacher_max,xdec_teacher_below1,xdec_teacher_above1,"
        "z0hat_cond_mean,z0hat_cond_std,z0hat_cond_min,z0hat_cond_max,"
        "mse_z0mean_cond,mse_z0sample_cond,"
        "xdec_cond_mean,xdec_cond_std,xdec_cond_min,xdec_cond_max,xdec_cond_below1,xdec_cond_above1"
    ]
    mse_teacher_mean_all: List[float] = []
    mse_cond_mean_all: List[float] = []
    mse_teacher_sample_all: List[float] = []
    mse_cond_sample_all: List[float] = []

    for i, (id_, dataset_index) in enumerate(chosen):
        item = eval_dataset[int(dataset_index)]
        x_dwi = item["image"].to(device).unsqueeze(0)  # [1,1,H,W]
        x_gt = item["gt"].to(device).unsqueeze(0)  # [1,1,H,W] (for reference only)

        z0_mean, z0_logvar = vae.encode(x_gt)
        # Match training-time latent definition (sampled z, not mean z).
        std = torch.exp(0.5 * z0_logvar)
        eps_gen = torch.Generator()
        eps_gen.manual_seed(int(gen_cfg.seed) + 10_000 + i)
        eps = torch.randn(
            std.shape, generator=eps_gen, device="cpu", dtype=std.dtype
        ).to(std.device)
        z0_sample = z0_mean + eps * std

        gen = torch.Generator()
        gen.manual_seed(int(gen_cfg.seed) + i)
        z_init = torch.randn((1, *latent_shape), generator=gen, device="cpu").to(device)
        if init_mode == "match_t":
            a_bar = float(alphas_cumprod[torch.tensor(start_t, device=device)].item())
            noise_scale = (1.0 - a_bar) ** 0.5
            z_init = z_init * float(noise_scale)

        def _pred_teacher(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return teacher(z, t)

        def _pred_cond(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return cond(latents=z, timesteps=t, dwi_image=x_dwi)

        if gen_cfg.sampler.lower() == "ddim":
            z_teacher = _ddim_sample(
                pred_eps_fn=_pred_teacher,
                z=z_init.clone(),
                timesteps=timesteps,
                alphas_cumprod=alphas_cumprod,
                eta=float(gen_cfg.eta),
            )
            z_cond = _ddim_sample(
                pred_eps_fn=_pred_cond,
                z=z_init.clone(),
                timesteps=timesteps,
                alphas_cumprod=alphas_cumprod,
                eta=float(gen_cfg.eta),
            )
        else:
            if start_t != max_t:
                print(
                    "Warning: sampler='dpmpp' currently ignores start_t; set sampler='ddim' to use truncated start_t."
                )
            z_teacher = _dpmpp_sample(
                pred_eps_fn=_pred_teacher,
                z=z_init.clone(),
                num_inference_steps=gen_cfg.num_inference_steps,
                num_train_timesteps=infer_cfg.num_train_timesteps,
                beta_start=infer_cfg.beta_start,
                beta_end=infer_cfg.beta_end,
                alphas_cumprod=alphas_cumprod,
            )
            z_cond = _dpmpp_sample(
                pred_eps_fn=_pred_cond,
                z=z_init.clone(),
                num_inference_steps=gen_cfg.num_inference_steps,
                num_train_timesteps=infer_cfg.num_train_timesteps,
                beta_start=infer_cfg.beta_start,
                beta_end=infer_cfg.beta_end,
                alphas_cumprod=alphas_cumprod,
            )

        mse_z0mean_teacher = float(F.mse_loss(z_teacher, z0_mean).item())
        mse_z0mean_cond = float(F.mse_loss(z_cond, z0_mean).item())
        mse_teacher_mean_all.append(mse_z0mean_teacher)
        mse_cond_mean_all.append(mse_z0mean_cond)

        mse_z0sample_teacher = float(F.mse_loss(z_teacher, z0_sample).item())
        mse_z0sample_cond = float(F.mse_loss(z_cond, z0_sample).item())
        mse_teacher_sample_all.append(mse_z0sample_teacher)
        mse_cond_sample_all.append(mse_z0sample_cond)

        z0m_mean, z0m_std, z0m_min, z0m_max = _mean_std_min_max(z0_mean)
        z0s_mean, z0s_std, z0s_min, z0s_max = _mean_std_min_max(z0_sample)
        zt_mean, zt_std, zt_min, zt_max = _mean_std_min_max(z_teacher)
        zc_mean, zc_std, zc_min, zc_max = _mean_std_min_max(z_cond)

        x_teacher = vae.decode(z_teacher)
        x_cond = vae.decode(z_cond)
        xt_mean, xt_std, xt_min, xt_max = _mean_std_min_max(x_teacher)
        xc_mean, xc_std, xc_min, xc_max = _mean_std_min_max(x_cond)
        xt_below, xt_above = _clamp_ratios_to_unit(x_teacher)
        xc_below, xc_above = _clamp_ratios_to_unit(x_cond)

        stats_lines.append(
            f"{id_},{int(dataset_index)},"
            f"{z0m_mean:.8f},{z0m_std:.8f},{z0m_min:.8f},{z0m_max:.8f},"
            f"{z0s_mean:.8f},{z0s_std:.8f},{z0s_min:.8f},{z0s_max:.8f},"
            f"{zt_mean:.8f},{zt_std:.8f},{zt_min:.8f},{zt_max:.8f},"
            f"{mse_z0mean_teacher:.8f},{mse_z0sample_teacher:.8f},"
            f"{xt_mean:.8f},{xt_std:.8f},{xt_min:.8f},{xt_max:.8f},{xt_below:.6f},{xt_above:.6f},"
            f"{zc_mean:.8f},{zc_std:.8f},{zc_min:.8f},{zc_max:.8f},"
            f"{mse_z0mean_cond:.8f},{mse_z0sample_cond:.8f},"
            f"{xc_mean:.8f},{xc_std:.8f},{xc_min:.8f},{xc_max:.8f},{xc_below:.6f},{xc_above:.6f}"
        )

        grid = torch.cat(
            [_denorm(x_dwi), _denorm(x_gt), _denorm(x_teacher), _denorm(x_cond)], dim=0
        )
        out_path = os.path.join(
            gen_cfg.out_dir,
            f"id_{id_}_idx{int(dataset_index):06d}_seed{gen_cfg.seed + i:06d}.png",
        )
        save_image(grid, out_path, nrow=4)

    with open(
        os.path.join(gen_cfg.out_dir, "gen_latent_stats.csv"), "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(stats_lines) + "\n")

    mean_mse_teacher_mean = sum(mse_teacher_mean_all) / max(
        len(mse_teacher_mean_all), 1
    )
    mean_mse_cond_mean = sum(mse_cond_mean_all) / max(len(mse_cond_mean_all), 1)
    mean_mse_teacher_sample = sum(mse_teacher_sample_all) / max(
        len(mse_teacher_sample_all), 1
    )
    mean_mse_cond_sample = sum(mse_cond_sample_all) / max(len(mse_cond_sample_all), 1)
    print(f"Saved results to: {gen_cfg.out_dir}")
    print(
        "mean MSE(z0_hat, z0_mean): "
        f"teacher={mean_mse_teacher_mean:.6f} cond={mean_mse_cond_mean:.6f}"
    )
    print(
        "mean MSE(z0_hat, z0_sample): "
        f"teacher={mean_mse_teacher_sample:.6f} cond={mean_mse_cond_sample:.6f}"
    )


if __name__ == "__main__":
    main()
