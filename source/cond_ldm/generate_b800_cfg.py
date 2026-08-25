from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch
from torchvision.utils import save_image

from config import CondLDMGenCFGConfig, UNetConfig, VAEConfig
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


def _make_ddim_timesteps(
    start_t: int, num_inference_steps: int, device: torch.device
) -> torch.Tensor:
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps 需为正数。")
    ts = torch.linspace(float(start_t), 0.0, steps=int(num_inference_steps))
    ts = torch.round(ts).to(torch.long)
    ts = torch.unique_consecutive(ts).to(device)
    if ts.numel() < 2:
        raise ValueError(
            "num_inference_steps 太小或 start_t 太小，导致 timesteps < 2。"
        )
    if int(ts[0].item()) != int(start_t):
        ts[0] = int(start_t)
    if int(ts[-1].item()) != 0:
        ts = torch.cat([ts, torch.zeros(1, device=device, dtype=torch.long)], dim=0)
    return ts


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
    """
    try:
        from diffusers import DPMSolverMultistepScheduler
    except Exception as e:  # pragma: no cover
        print(
            f"Warning: failed to import DPMSolverMultistepScheduler ({e}); falling back to DDIM."
        )
        timesteps = _make_ddim_timesteps(
            int(num_train_timesteps) - 1, int(num_inference_steps), z.device
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


def _make_uncond_image(
    *,
    x_like: torch.Tensor,
    mode: str,
    seed: int,
    sample_index: int,
) -> torch.Tensor:
    mode = str(mode).lower().strip()
    if mode == "zeros":
        return torch.zeros_like(x_like)
    if mode == "noise":
        gen = torch.Generator()
        gen.manual_seed(int(seed) + int(sample_index))
        return torch.randn(
            x_like.shape, device="cpu", generator=gen, dtype=x_like.dtype
        ).to(x_like.device)
    raise ValueError(f"uncond_fill 需为 'zeros' 或 'noise'，当前为 {mode!r}")


@torch.no_grad()
def main() -> None:
    infer_cfg = InferConfig()
    gen_cfg = CondLDMGenCFGConfig()

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

    s = float(gen_cfg.guidance_scale)
    if s < 0:
        raise ValueError("guidance_scale 需为非负数。")

    vae_cfg = VAEConfig(device=infer_cfg.device)
    vae = load_vae(vae_cfg, infer_cfg.vae_ckpt_path, device)
    cond = load_cond_model(cfg=infer_cfg, vae_cfg=vae_cfg, device=device)
    teacher = _load_teacher_model(cfg=infer_cfg, vae_cfg=vae_cfg, device=device)

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
        eval_dataset, num_ids=int(gen_cfg.num_ids), seed=int(gen_cfg.seed)
    )

    if gen_cfg.sampler.lower() == "ddim":
        timesteps = _make_ddim_timesteps(
            start_t=start_t,
            num_inference_steps=int(gen_cfg.num_inference_steps),
            device=device,
        )
    else:
        timesteps = None

    for i, (id_, dataset_index) in enumerate(chosen):
        item = eval_dataset[int(dataset_index)]
        x_dwi = item["image"].to(device).unsqueeze(0)  # [1,1,H,W]
        x_gt = item["gt"].to(device).unsqueeze(0)  # [1,1,H,W] (for reference only)

        x_uncond = _make_uncond_image(
            x_like=x_dwi,
            mode=str(gen_cfg.uncond_fill),
            seed=int(gen_cfg.uncond_noise_seed),
            sample_index=i,
        )

        gen = torch.Generator()
        gen.manual_seed(int(gen_cfg.seed) + i)
        z = torch.randn((1, *latent_shape), generator=gen, device="cpu").to(device)
        if init_mode == "match_t":
            a_bar = float(alphas_cumprod[torch.tensor(start_t, device=device)].item())
            z = z * float((1.0 - a_bar) ** 0.5)

        def _pred_teacher(z_in: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return teacher(z_in, t)

        def _pred_cond(z_in: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return cond(latents=z_in, timesteps=t, dwi_image=x_dwi)

        def _pred_cfg(z_in: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            eps_c = cond(latents=z_in, timesteps=t, dwi_image=x_dwi)
            eps_u = cond(latents=z_in, timesteps=t, dwi_image=x_uncond)
            return eps_u + s * (eps_c - eps_u)

        if gen_cfg.sampler.lower() == "ddim":
            z_teacher = _ddim_sample(
                pred_eps_fn=_pred_teacher,
                z=z.clone(),
                timesteps=timesteps,
                alphas_cumprod=alphas_cumprod,
                eta=float(gen_cfg.eta),
            )
            z_cond = _ddim_sample(
                pred_eps_fn=_pred_cond,
                z=z.clone(),
                timesteps=timesteps,
                alphas_cumprod=alphas_cumprod,
                eta=float(gen_cfg.eta),
            )
            z_cfg = _ddim_sample(
                pred_eps_fn=_pred_cfg,
                z=z.clone(),
                timesteps=timesteps,
                alphas_cumprod=alphas_cumprod,
                eta=float(gen_cfg.eta),
            )
        else:
            if start_t != max_t:
                print(
                    "Warning: sampler='dpmpp' currently ignores start_t; prefer sampler='ddim' for truncated start_t."
                )
            z_teacher = _dpmpp_sample(
                pred_eps_fn=_pred_teacher,
                z=z.clone(),
                num_inference_steps=int(gen_cfg.num_inference_steps),
                num_train_timesteps=int(infer_cfg.num_train_timesteps),
                beta_start=float(infer_cfg.beta_start),
                beta_end=float(infer_cfg.beta_end),
                alphas_cumprod=alphas_cumprod,
            )
            z_cond = _dpmpp_sample(
                pred_eps_fn=_pred_cond,
                z=z.clone(),
                num_inference_steps=int(gen_cfg.num_inference_steps),
                num_train_timesteps=int(infer_cfg.num_train_timesteps),
                beta_start=float(infer_cfg.beta_start),
                beta_end=float(infer_cfg.beta_end),
                alphas_cumprod=alphas_cumprod,
            )
            z_cfg = _dpmpp_sample(
                pred_eps_fn=_pred_cfg,
                z=z.clone(),
                num_inference_steps=int(gen_cfg.num_inference_steps),
                num_train_timesteps=int(infer_cfg.num_train_timesteps),
                beta_start=float(infer_cfg.beta_start),
                beta_end=float(infer_cfg.beta_end),
                alphas_cumprod=alphas_cumprod,
            )

        x_teacher = vae.decode(z_teacher)
        x_cond = vae.decode(z_cond)
        x_cfg = vae.decode(z_cfg)

        grid = torch.cat(
            [
                _denorm(x_dwi),
                _denorm(x_gt),
                _denorm(x_teacher),
                _denorm(x_cond),
                _denorm(x_cfg),
            ],
            dim=0,
        )
        out_path = os.path.join(
            gen_cfg.out_dir, f"id_{id_}_idx{int(dataset_index):06d}_cfg{s:g}.png"
        )
        save_image(grid, out_path, nrow=5)

    print(f"Saved results to: {gen_cfg.out_dir}")


if __name__ == "__main__":
    main()
