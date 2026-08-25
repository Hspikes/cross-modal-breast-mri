from typing import Optional, Union

import torch
from diffusers import UNet2DModel
from torch import nn

from config import UNetConfig


class LatentUNet(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config
        self.unet = UNet2DModel(
            sample_size=config.sample_size,  # latent H' (可为 None)
            in_channels=config.latent_channels,  # C_z
            out_channels=config.latent_channels,  # C_z
            block_out_channels=config.block_out_channels,
            down_block_types=config.down_block_types,
            up_block_types=config.up_block_types,
            layers_per_block=config.num_res_blocks,
        )

    @staticmethod
    def _prepare_timesteps(
        timesteps: Union[int, float, torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        config: UNetConfig,
    ) -> torch.Tensor:
        # int/float -> broadcast
        if isinstance(timesteps, (int, float)):
            t = torch.tensor([timesteps] * batch_size, device=device, dtype=dtype)
        elif isinstance(timesteps, torch.Tensor):
            t = timesteps.to(device=device)
            if t.dim() == 0:
                t = t.view(1).expand(batch_size)
            elif t.dim() == 1:
                if t.numel() == 1:
                    t = t.expand(batch_size)
                elif t.numel() == batch_size:
                    pass
                else:
                    raise ValueError(
                        f"timesteps 1-D tensor 长度为 {t.numel()}，"
                        f"与 batch_size={batch_size} 不匹配。"
                    )
            else:
                raise ValueError(f"timesteps 维度为 {t.dim()}，期望为 0 或 1 维。")

            if t.dtype != dtype:
                t = t.to(dtype)
        else:
            raise TypeError(
                f"Unsupported timesteps type: {type(timesteps)}. "
                f"仅支持 int/float/torch.Tensor。"
            )

        # 检查时间步范围
        if t.max().item() >= config.max_timesteps:
            raise ValueError(
                f"时间步 {t.max().item()} 超过最大 {config.max_timesteps - 1}"
            )
        return t

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: Union[int, float, torch.Tensor],
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents.dim() != 4:
            raise ValueError(
                f"期望 latents 形状为 [B, C, H, W]，当前为 {latents.shape}"
            )

        batch_size = latents.shape[0]
        device = latents.device

        t = self._prepare_timesteps(
            timesteps=timesteps,
            batch_size=batch_size,
            device=device,
            dtype=torch.long,
            config=self.config,
        )

        unet_output = self.unet(latents, t)

        return unet_output.sample
