from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from config import CondLDMDiagCFGConfig, UNetConfig, VAEConfig
from config import CondLDMInferConfig as InferConfig

from ..data_pre_processing.dataset import MRIDataset, MRItransform
from ..Unet.model import LatentUNet
from .infer import (
    get_noise_schedule,
    load_cond_model,
    load_teacher_unet_config_and_state,
    load_vae,
)


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
        raise ValueError(f"num_ids={num_ids} 超过可用 id 数量 {len(ids)}")

    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(len(ids), generator=g)[:num_ids].tolist()
    return [(ids[i], id_to_first_index[ids[i]]) for i in perm]


def _make_reverse_schedule(t_start: int, num_steps: int) -> List[int]:
    if num_steps < 2:
        raise ValueError("num_steps 至少为 2（包含起点和终点）。")
    if t_start <= 0:
        raise ValueError("t_start 需为正数。")

    ts = torch.linspace(float(t_start), 0.0, steps=int(num_steps))
    ts = torch.round(ts).to(torch.long).tolist()
    out: List[int] = []
    last = None
    for t in ts:
        t = int(t)
        if last is None or t != last:
            out.append(t)
            last = t
    if out[-1] != 0:
        out.append(0)
    out = sorted(set(out), reverse=True)
    return out


@torch.no_grad()
def _encode_latent_mean(vae, x: torch.Tensor) -> torch.Tensor:
    mean, _logvar = vae.encode(x)
    return mean


@torch.no_grad()
def _add_noise(
    z0: torch.Tensor,
    t: int,
    alphas_cumprod: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    t_idx = torch.full((z0.size(0),), int(t), device=z0.device, dtype=torch.long)
    a_bar = alphas_cumprod[t_idx].view(-1, 1, 1, 1)
    return torch.sqrt(a_bar) * z0 + torch.sqrt(1.0 - a_bar) * noise


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
def _ddim_reverse_from_t(
    *,
    zt: torch.Tensor,
    t_start: int,
    timesteps: Iterable[int],
    alphas_cumprod: torch.Tensor,
    pred_eps_fn,
) -> torch.Tensor:
    ts = [int(t) for t in timesteps]
    if not ts or ts[0] != int(t_start):
        raise ValueError("timesteps 必须以 t_start 开头。")

    z = zt
    pred_x0 = None
    for i in range(len(ts) - 1):
        t = int(ts[i])
        t_prev = int(ts[i + 1])
        if t_prev >= t:
            continue

        t_batch = torch.full((z.size(0),), t, device=z.device, dtype=torch.long)
        a_bar_t = alphas_cumprod[t_batch].view(-1, 1, 1, 1)

        t_prev_batch = torch.full(
            (z.size(0),), t_prev, device=z.device, dtype=torch.long
        )
        a_bar_prev = alphas_cumprod[t_prev_batch].view(-1, 1, 1, 1)

        pred_eps = pred_eps_fn(z, t_batch)
        pred_x0 = (z - torch.sqrt(1.0 - a_bar_t) * pred_eps) / torch.sqrt(
            a_bar_t + 1e-8
        )
        z = torch.sqrt(a_bar_prev) * pred_x0 + torch.sqrt(1.0 - a_bar_prev) * pred_eps

    if pred_x0 is None:
        raise RuntimeError("timesteps 长度过短或不合法，未执行任何反向步。")
    return pred_x0


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


@torch.no_grad()
def main() -> None:
    diag_cfg = CondLDMDiagCFGConfig()
    infer_cfg = InferConfig()

    device = torch.device(infer_cfg.device)
    os.makedirs(diag_cfg.out_dir, exist_ok=True)

    _betas, alphas_cumprod = get_noise_schedule(infer_cfg)
    alphas_cumprod = alphas_cumprod.to(device)

    t_fixed = int(diag_cfg.t)
    if t_fixed <= 0 or t_fixed >= int(alphas_cumprod.numel()):
        raise ValueError(f"t={t_fixed} 超出范围 [1, {int(alphas_cumprod.numel()) - 1}]")

    tf = MRItransform()
    eval_dataset = MRIDataset(
        data_folder=infer_cfg.data_root,
        sequence_list_txt=infer_cfg.eval_list,
        transforms=tf,
    )

    chosen = _pick_indices_from_distinct_ids(
        eval_dataset,
        num_ids=int(diag_cfg.num_ids),
        seed=int(diag_cfg.seed),
    )

    vae_cfg = VAEConfig(device=infer_cfg.device)
    vae = load_vae(vae_cfg, infer_cfg.vae_ckpt_path, device)
    teacher = _load_teacher_model(cfg=infer_cfg, vae_cfg=vae_cfg, device=device)
    cond = load_cond_model(cfg=infer_cfg, vae_cfg=vae_cfg, device=device)

    reverse_ts = _make_reverse_schedule(t_start=t_fixed, num_steps=int(diag_cfg.steps))
    s = float(diag_cfg.guidance_scale)

    summary_lines = [
        "id,index,mse_teacher,mse_cond,mse_uncond,mse_cfg,l1_teacher,l1_cond,l1_cfg"
    ]
    mse_teacher_all: List[float] = []
    mse_cond_all: List[float] = []
    mse_uncond_all: List[float] = []
    mse_cfg_all: List[float] = []
    l1_teacher_all: List[float] = []
    l1_cond_all: List[float] = []
    l1_cfg_all: List[float] = []

    for i, (id_, dataset_index) in enumerate(chosen):
        item = eval_dataset[int(dataset_index)]
        x_t1ce = item["gt"].to(device).unsqueeze(0)  # [1,1,H,W]
        x_dwi = item["image"].to(device).unsqueeze(0)  # [1,1,H,W]

        x_uncond = _make_uncond_image(
            x_like=x_dwi,
            mode=str(diag_cfg.uncond_fill),
            seed=int(diag_cfg.uncond_noise_seed),
            sample_index=i,
        )

        z0 = _encode_latent_mean(vae, x_t1ce)

        gen = torch.Generator()
        gen.manual_seed(int(diag_cfg.seed) + i)
        noise = torch.randn(z0.shape, device="cpu", generator=gen, dtype=z0.dtype).to(
            z0.device
        )
        zt = _add_noise(z0, t=t_fixed, alphas_cumprod=alphas_cumprod, noise=noise)

        t_batch = torch.full((zt.size(0),), t_fixed, device=device, dtype=torch.long)
        eps_teacher = teacher(zt, t_batch)
        eps_cond = cond(latents=zt, timesteps=t_batch, dwi_image=x_dwi)
        eps_uncond = cond(latents=zt, timesteps=t_batch, dwi_image=x_uncond)
        eps_cfg = eps_uncond + s * (eps_cond - eps_uncond)

        mse_teacher = float(F.mse_loss(eps_teacher, noise).item())
        mse_cond = float(F.mse_loss(eps_cond, noise).item())
        mse_uncond = float(F.mse_loss(eps_uncond, noise).item())
        mse_cfg = float(F.mse_loss(eps_cfg, noise).item())

        mse_teacher_all.append(mse_teacher)
        mse_cond_all.append(mse_cond)
        mse_uncond_all.append(mse_uncond)
        mse_cfg_all.append(mse_cfg)

        def _pred_eps_teacher(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return teacher(z, t)

        def _pred_eps_cond(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return cond(latents=z, timesteps=t, dwi_image=x_dwi)

        def _pred_eps_cfg(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            eps_c = cond(latents=z, timesteps=t, dwi_image=x_dwi)
            eps_u = cond(latents=z, timesteps=t, dwi_image=x_uncond)
            return eps_u + s * (eps_c - eps_u)

        z0_hat_teacher = _ddim_reverse_from_t(
            zt=zt,
            t_start=t_fixed,
            timesteps=reverse_ts,
            alphas_cumprod=alphas_cumprod,
            pred_eps_fn=_pred_eps_teacher,
        )
        z0_hat_cond = _ddim_reverse_from_t(
            zt=zt,
            t_start=t_fixed,
            timesteps=reverse_ts,
            alphas_cumprod=alphas_cumprod,
            pred_eps_fn=_pred_eps_cond,
        )
        z0_hat_cfg = _ddim_reverse_from_t(
            zt=zt,
            t_start=t_fixed,
            timesteps=reverse_ts,
            alphas_cumprod=alphas_cumprod,
            pred_eps_fn=_pred_eps_cfg,
        )

        x_hat_teacher = vae.decode(z0_hat_teacher)
        x_hat_cond = vae.decode(z0_hat_cond)
        x_hat_cfg = vae.decode(z0_hat_cfg)

        l1_teacher = float(F.l1_loss(x_hat_teacher, x_t1ce).item())
        l1_cond = float(F.l1_loss(x_hat_cond, x_t1ce).item())
        l1_cfg = float(F.l1_loss(x_hat_cfg, x_t1ce).item())
        l1_teacher_all.append(l1_teacher)
        l1_cond_all.append(l1_cond)
        l1_cfg_all.append(l1_cfg)

        summary_lines.append(
            f"{id_},{int(dataset_index)},"
            f"{mse_teacher:.8f},{mse_cond:.8f},{mse_uncond:.8f},{mse_cfg:.8f},"
            f"{l1_teacher:.8f},{l1_cond:.8f},{l1_cfg:.8f}"
        )

        grid = torch.cat(
            [
                _denorm(x_dwi),
                _denorm(x_t1ce),
                _denorm(x_hat_teacher),
                _denorm(x_hat_cond),
                _denorm(x_hat_cfg),
            ],
            dim=0,
        )
        out_path = os.path.join(
            diag_cfg.out_dir,
            f"id_{id_}_idx{int(dataset_index):06d}_t{t_fixed}_steps{len(reverse_ts)}_cfg{s:g}.png",
        )
        save_image(grid, out_path, nrow=5)

    def _mean(xs: List[float]) -> float:
        return sum(xs) / max(len(xs), 1)

    summary_lines.append(
        "MEAN,_,"
        f"{_mean(mse_teacher_all):.8f},{_mean(mse_cond_all):.8f},{_mean(mse_uncond_all):.8f},{_mean(mse_cfg_all):.8f},"
        f"{_mean(l1_teacher_all):.8f},{_mean(l1_cond_all):.8f},{_mean(l1_cfg_all):.8f}"
    )

    with open(
        os.path.join(diag_cfg.out_dir, f"t{t_fixed}_cfg{s:g}_summary.csv"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Saved results to: {diag_cfg.out_dir}")


if __name__ == "__main__":
    main()
