import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import Decoder, Encoder

from config import VAEConfig


def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std


def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
    return kl.mean()


def vae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    config: VAEConfig,
):
    if config.recon_loss_type.lower() == "l1":
        recon = F.l1_loss(x_recon, x)
    else:
        recon = F.mse_loss(x_recon, x)

    kl = kl_loss(mean, logvar)

    total = recon + config.kl_weight * kl
    return total, recon, kl


class VAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        z_channels = config.latent_channels

        self.encoder = Encoder(
            in_channels=config.img_channels,  # 1 通道
            # diffusers 的 Encoder 会输出 2 * out_channels（内部已经为 mean/logvar 预留通道）
            # 因此这里设为 z_channels，使得最终输出通道数为 2 * z_channels，方便 chunk
            out_channels=z_channels,
            down_block_types=(
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ),
            block_out_channels=(64, 128, 256, 256),
        )

        self.decoder = Decoder(
            in_channels=z_channels,
            out_channels=config.img_channels,  # 输出仍然是 1 通道图像
            up_block_types=(
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ),
            block_out_channels=(256, 256, 128, 64),
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)  # [B, 2*z_channels, H', W']
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar

    def decode(self, z: torch.Tensor):
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x: torch.Tensor, sample: bool = True):
        mean, logvar = self.encode(x)
        z = reparameterize(mean, logvar) if sample else mean
        x_recon = self.decode(z)
        return x_recon, mean, logvar
