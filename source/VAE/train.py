import copy
import glob
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from config import VAEConfig

from ..data_pre_processing.dataset import MRIDataset, MRItransform
from .model import VAE, vae_loss


def train_one_epoch(
    model: VAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: VAEConfig,
):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    n_samples = len(dataloader.dataset)

    for batch in dataloader:
        # MRIDataset 中 'gt' 的格式: [B, 1, H, W]，数值范围约 [-1,1]
        x = batch["gt"].to(device)

        optimizer.zero_grad()

        x_recon, mean, logvar = model(x)
        loss, recon, kl = vae_loss(x_recon, x, mean, logvar, config)

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_recon += recon.item() * batch_size
        total_kl += kl.item() * batch_size

    return (
        total_loss / n_samples,
        total_recon / n_samples,
        total_kl / n_samples,
    )


@torch.no_grad()
def eval_one_epoch(
    model: VAE,
    dataloader: DataLoader,
    device: torch.device,
    config: VAEConfig,
):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    n_samples = len(dataloader.dataset)

    for batch in dataloader:
        x = batch["gt"].to(device)
        x_recon, mean, logvar = model(x, sample=False)
        loss, recon, kl = vae_loss(x_recon, x, mean, logvar, config)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_recon += recon.item() * batch_size
        total_kl += kl.item() * batch_size

    return (
        total_loss / n_samples,
        total_recon / n_samples,
        total_kl / n_samples,
    )


@torch.no_grad()
def save_recon_samples(
    model: VAE,
    dataloader: DataLoader,
    device: torch.device,
    config: VAEConfig,
    num_images: int = 8,
):
    """
    从 dataloader 中取若干样本，保存原图和重建图，便于肉眼检查 VAE 效果。

    输出目录：
        config.recon_vis_dir/real_x.png
        config.recon_vis_dir/recon_x.png
    """

    os.makedirs(config.recon_vis_dir, exist_ok=True)
    model.eval()

    saved = 0
    for batch in dataloader:
        x = batch["gt"].to(device)
        x_recon, _, _ = model(x, sample=False)

        x_np = x.cpu().numpy()
        x_recon_np = x_recon.cpu().numpy()

        x_np = (x_np + 1.0) / 2.0
        x_recon_np = (x_recon_np + 1.0) / 2.0

        B = x_np.shape[0]
        for i in range(B):
            if saved >= num_images:
                return

            img_real = x_np[i, 0, :, :]  # [H, W]
            img_recon = x_recon_np[i, 0, :, :]

            plt.imsave(
                os.path.join(config.recon_vis_dir, f"real_{saved}.png"),
                img_real,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            plt.imsave(
                os.path.join(config.recon_vis_dir, f"recon_{saved}.png"),
                img_recon,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            saved += 1


def _infer_epoch_from_path(path: str) -> Optional[int]:
    base = os.path.basename(path)
    # e.g. vae_epoch_10.pth / vae_epoch_010.pth
    if not base.startswith("vae_epoch_"):
        return None
    digits = "".join(ch for ch in base[len("vae_epoch_") :] if ch.isdigit())
    return int(digits) if digits else None


def _find_resume_ckpt(ckpt_dir: str) -> Optional[str]:
    latest_pt = os.path.join(ckpt_dir, "latest.pt")
    if os.path.isfile(latest_pt):
        return latest_pt
    latest_pth = os.path.join(ckpt_dir, "latest.pth")
    if os.path.isfile(latest_pth):
        return latest_pth
    candidates = glob.glob(os.path.join(ckpt_dir, "vae_epoch_*.pth")) + glob.glob(
        os.path.join(ckpt_dir, "vae_epoch_*.pt")
    )
    if candidates:

        def _key(p: str) -> Tuple[int, str]:
            epoch = _infer_epoch_from_path(p)
            return (epoch if epoch is not None else -1, p)

        return sorted(candidates, key=_key)[-1]
    best = os.path.join(ckpt_dir, "vae_best.pth")
    if os.path.isfile(best):
        return best
    return None


def train():
    config = VAEConfig()
    device = torch.device(config.device)

    os.makedirs(config.ckpt_dir, exist_ok=True)
    os.makedirs(config.recon_vis_dir, exist_ok=True)

    tf = MRItransform()
    train_dataset = MRIDataset(
        data_folder=config.data_root,
        sequence_list_txt=config.train_list,
        transforms=tf,
        expand_train_fixed3=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    eval_dataset = MRIDataset(
        data_folder=config.data_root,
        sequence_list_txt=config.eval_list,
        transforms=tf,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = VAE(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    resume_path = _find_resume_ckpt(config.ckpt_dir)
    start_epoch = 1
    best_eval_loss = float("inf")

    best_path = os.path.join(config.ckpt_dir, "vae_best.pth")
    best_model_wts = copy.deepcopy(model.state_dict())
    if os.path.isfile(best_path):
        try:
            best_state = torch.load(best_path, map_location="cpu")
            if isinstance(best_state, dict) and "model" in best_state:
                best_state = best_state["model"]
            if isinstance(best_state, dict):
                best_model_wts = best_state
        except Exception as e:
            print(f"Warning: failed to load {best_path}: {e}")

    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        state = torch.load(resume_path, map_location=device)
        last_epoch = 0
        opt_state = None
        best_eval_from_ckpt = None

        if isinstance(state, dict) and (
            "model" in state or "epoch" in state or "optimizer" in state
        ):
            model_state = state.get("model")
            if isinstance(model_state, dict):
                model.load_state_dict(model_state, strict=True)
            else:
                model.load_state_dict(state, strict=True)
            opt_state = state.get("optimizer")
            last_epoch = int(state.get("epoch", 0))
            if "best_eval_loss" in state:
                best_eval_from_ckpt = float(state["best_eval_loss"])
        elif isinstance(state, dict):
            model.load_state_dict(state, strict=True)
            inferred = _infer_epoch_from_path(resume_path)
            if inferred is not None:
                last_epoch = inferred
        else:
            raise ValueError(f"Unexpected checkpoint format at {resume_path}")

        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
            for group in optimizer.param_groups:
                group["lr"] = config.lr

        if best_eval_from_ckpt is not None:
            best_eval_loss = best_eval_from_ckpt

        start_epoch = last_epoch + 1

    if start_epoch > config.epochs:
        print(
            f"start_epoch={start_epoch} > config.epochs={config.epochs}; nothing to do."
        )
        return

    for epoch in range(start_epoch, config.epochs + 1):
        print(f"[Epoch {epoch:03d}] ")
        train_loss, train_recon, train_kl = train_one_epoch(
            model, train_loader, optimizer, device, config
        )
        eval_loss, eval_recon, eval_kl = eval_one_epoch(
            model, eval_loader, device, config
        )

        print(
            f"train_loss={train_loss:.4f}, train_recon={train_recon:.4f}, train_kl={train_kl:.4f} "
        )
        print(
            f"eval_loss={eval_loss:.4f}, eval_recon={eval_recon:.4f}, eval_kl={eval_kl:.4f}"
        )

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "best_eval_loss": best_eval_loss,
        }
        torch.save(ckpt, os.path.join(config.ckpt_dir, "latest.pt"))

        if epoch % 10 == 0:
            ckpt_path = os.path.join(config.ckpt_dir, f"vae_epoch_{epoch}.pth")
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
            save_recon_samples(model, eval_loader, device, config, num_images=8)

    torch.save(best_model_wts, os.path.join(config.ckpt_dir, "vae_best.pth"))


if __name__ == "__main__":
    train()
