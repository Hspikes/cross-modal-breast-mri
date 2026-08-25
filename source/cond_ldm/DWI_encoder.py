from typing import List, Optional, Sequence, Tuple, Union

import torch
from diffusers import UNet2DModel
from torch import nn

from config import UNetConfig


class AdapterToVAE(nn.Module):
    def __init__(
        self,
        vae_block_out_channels: Sequence[int] = (64, 128, 256, 256),
        use_norm_act: bool = True,
        num_groups: int = 8,
    ):
        super().__init__()
        self.vae_block_out_channels = tuple(vae_block_out_channels)

        self.proj = nn.ModuleList()
        for cout in self.vae_block_out_channels:
            layers: List[nn.Module] = [
                nn.LazyConv2d(cout, kernel_size=1, stride=1, padding=0)
            ]
            if use_norm_act:
                # GroupNorm 对小 batch 更友好
                g = min(num_groups, cout)
                if cout % g != 0:
                    g = 1
                layers += [nn.GroupNorm(num_groups=g, num_channels=cout), nn.SiLU()]
            self.proj.append(nn.Sequential(*layers))

    def forward(self, selected_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(selected_feats) != len(self.proj):
            raise ValueError(
                f"AdapterToVAE 期望 {len(self.proj)} 个尺度特征，"
                f"但得到 {len(selected_feats)} 个"
            )

        out: List[torch.Tensor] = []
        for feat, proj_i in zip(selected_feats, self.proj):
            out.append(proj_i(feat))
        return out


class DWIEncoder(nn.Module):
    """
    - 模块二兼容（保留 UNet-style 编码主干输出）
    - 模块三扩展（可选 UNet -> VAE adapter 蒸馏支路）

    输出:
        unet_feats: List[Tensor]
            feats[0]: conv_in 输出
            feats[1..]: 每个 down_block 输出

    模块二用法:
        unet_feats = dwi_encoder(x, t)

    模块三用法:
        unet_feats, vae_like_feats = dwi_encoder(x, t, return_vae_feats=True)
    """

    def __init__(
        self,
        unet_config: UNetConfig,
        in_channels: int = 1,
        # --- Module3 distill branch ---
        enable_vae_adapter: bool = False,
        vae_adapter_feat_indices: Sequence[int] = (1, 2, 3, 4),
        vae_block_out_channels: Sequence[int] = (64, 128, 256, 256),
        use_norm_act_in_adapter: bool = True,
    ):
        super().__init__()

        self.config = unet_config
        self.in_channels = in_channels

        self.unet_backbone = UNet2DModel(
            sample_size=unet_config.sample_size,
            in_channels=in_channels,
            out_channels=unet_config.latent_channels,  # 模块二里不直接用
            block_out_channels=unet_config.block_out_channels,
            down_block_types=unet_config.down_block_types,
            up_block_types=unet_config.up_block_types,
            layers_per_block=unet_config.num_res_blocks,
        )

        self.enable_vae_adapter = enable_vae_adapter
        self.vae_adapter_feat_indices = tuple(vae_adapter_feat_indices)
        self.vae_block_out_channels = tuple(vae_block_out_channels)

        if enable_vae_adapter:
            self.adapter_to_vae = AdapterToVAE(
                vae_block_out_channels=self.vae_block_out_channels,
                use_norm_act=use_norm_act_in_adapter,
            )
        else:
            self.adapter_to_vae = None

    def _build_temb(
        self,
        x: torch.Tensor,
        timesteps: Optional[Union[torch.Tensor, int, List[int]]],
    ) -> torch.Tensor:
        # 统一到 device/dtype，并检查 batch 尺寸
        if timesteps is None:
            timesteps = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
        else:
            timesteps = torch.as_tensor(timesteps, device=x.device, dtype=torch.long)
            if timesteps.dim() == 0:
                timesteps = timesteps.expand(x.size(0))
            elif timesteps.dim() == 1:
                if timesteps.numel() == 1:
                    timesteps = timesteps.expand(x.size(0))
                elif timesteps.numel() != x.size(0):
                    raise ValueError(
                        f"timesteps 长度为 {timesteps.numel()}，与 batch={x.size(0)} 不匹配"
                    )
            else:
                raise ValueError(f"timesteps 期望 0/1 维，当前 {timesteps.dim()} 维")

        temb = self.unet_backbone.time_proj(timesteps)
        temb = self.unet_backbone.time_embedding(temb)
        return temb

    def forward(
        self,
        x: torch.Tensor,
        timesteps: Optional[Union[torch.Tensor, int, List[int]]] = None,
        return_vae_feats: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        if x.dim() != 4:
            raise ValueError(f"期望 x 形状为 [B, C, H, W]，当前为 {x.shape}")

        feats: List[torch.Tensor] = []

        # stem
        h = self.unet_backbone.conv_in(x)
        feats.append(h)

        temb = self._build_temb(x, timesteps)

        # down path
        for down_block in self.unet_backbone.down_blocks:
            out = down_block(h, temb=temb)
            h = out[0] if isinstance(out, tuple) else out
            feats.append(h)

        # -------- Module3 distill branch --------
        if return_vae_feats:
            if not self.enable_vae_adapter:
                raise RuntimeError("return_vae_feats=True 但未启用 enable_vae_adapter")
            if self.adapter_to_vae is None:
                raise RuntimeError("enable_vae_adapter=True 但 adapter_to_vae 未初始化")

            max_idx = len(feats) - 1
            for i in self.vae_adapter_feat_indices:
                if i < 0 or i > max_idx:
                    raise IndexError(
                        f"vae_adapter_feat_indices 包含非法 index={i}，"
                        f"当前 feats 长度={len(feats)}（最大 index={max_idx}）"
                    )

            selected = [feats[i] for i in self.vae_adapter_feat_indices]

            vae_like_feats = self.adapter_to_vae(selected)

            return feats, vae_like_feats

        return feats
