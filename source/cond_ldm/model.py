from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from config import UNetConfig
from source.Unet.model import LatentUNet

from .DWI_encoder import DWIEncoder


class GatedCrossAttention2D(nn.Module):
    def __init__(
        self,
        query_channels: int,
        cond_channels: int,
        heads: int = 4,
        dim_head: int = 64,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.query_channels = query_channels
        self.cond_channels = cond_channels
        self.heads = heads
        self.dim_head = dim_head
        self.hidden_dim = heads * dim_head

        self.to_q = nn.Conv2d(query_channels, self.hidden_dim, kernel_size=1)
        self.to_k = nn.Conv2d(cond_channels, self.hidden_dim, kernel_size=1)
        self.to_v = nn.Conv2d(cond_channels, self.hidden_dim, kernel_size=1)
        self.to_out = nn.Conv2d(self.hidden_dim, query_channels, kernel_size=1)

        self.scale = dim_head**-0.5
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

        nn.init.zeros_(self.to_out.weight)
        if self.to_out.bias is not None:
            nn.init.zeros_(self.to_out.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        if x.dim() != 4 or cond.dim() != 4:
            raise ValueError(
                f"x/cond 期望为 [B,C,H,W]，当前 x={tuple(x.shape)} cond={tuple(cond.shape)}"
            )

        b, _, h, w = x.shape
        _, _, h_c, w_c = cond.shape
        if (h_c != h) or (w_c != w):
            cond = F.interpolate(
                cond, size=(h, w), mode="bilinear", align_corners=False
            )

        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)

        def _reshape_for_heads(t: torch.Tensor) -> torch.Tensor:
            t = t.view(b, self.heads, self.dim_head, h * w)
            return t.permute(0, 1, 3, 2)

        q = _reshape_for_heads(q)
        k = _reshape_for_heads(k)
        v = _reshape_for_heads(v)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = attn_scores.softmax(dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.permute(0, 1, 3, 2).contiguous().view(b, self.hidden_dim, h, w)
        out = self.to_out(out)
        return x + self.gate * out


class _UNetMacroInjector:
    """
    通过 forward hooks 在冻结 UNet 的 conv_in/down/mid/up 输出处注入条件残差。

    设计目标：
    - 不改动 teacher UNet 的参数与 forward 接口
    - 注入逻辑独立于 diffusers UNet2DModel.forward 的具体实现细节（尽量只依赖模块树结构）
    """

    def __init__(
        self,
        unet: nn.Module,
        stem_injector: Optional[nn.Module],
        down_injectors: Sequence[nn.Module],
        mid_injector: Optional[nn.Module],
        up_injectors: Sequence[nn.Module],
        apply_to_down_res_samples: bool = True,
    ):
        self.unet = unet
        self.stem_injector = stem_injector
        self.down_injectors = list(down_injectors)
        self.mid_injector = mid_injector
        self.up_injectors = list(up_injectors)
        self.apply_to_down_res_samples = apply_to_down_res_samples

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._active = False
        self._cond_by_key: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        if hasattr(self.unet, "conv_in") and self.stem_injector is not None:
            self._handles.append(
                self.unet.conv_in.register_forward_hook(self._hook_stem)
            )

        if hasattr(self.unet, "down_blocks"):
            for i, block in enumerate(self.unet.down_blocks):
                if i < len(self.down_injectors):
                    self._handles.append(
                        block.register_forward_hook(self._make_hook_down(i))
                    )

        if hasattr(self.unet, "mid_block") and self.mid_injector is not None:
            self._handles.append(
                self.unet.mid_block.register_forward_hook(self._hook_mid)
            )

        if hasattr(self.unet, "up_blocks"):
            for i, block in enumerate(self.unet.up_blocks):
                if i < len(self.up_injectors):
                    self._handles.append(
                        block.register_forward_hook(self._make_hook_up(i))
                    )

    @contextmanager
    def use(self, cond_by_key: Dict[str, torch.Tensor]):
        self._active = True
        self._cond_by_key = cond_by_key
        try:
            yield
        finally:
            self._active = False
            self._cond_by_key = {}

    def _get_cond(self, key: str) -> Optional[torch.Tensor]:
        return self._cond_by_key.get(key)

    def _apply(
        self, x: torch.Tensor, injector: nn.Module, cond: torch.Tensor
    ) -> torch.Tensor:
        if not torch.is_tensor(x) or x.dim() != 4:
            return x
        if hasattr(injector, "query_channels") and x.size(1) != getattr(
            injector, "query_channels"
        ):
            return x
        return injector(x, cond)

    def _hook_stem(self, module: nn.Module, inputs: tuple, output: torch.Tensor):
        if not self._active or self.stem_injector is None:
            return output
        cond = self._get_cond("stem")
        if cond is None:
            return output
        return self._apply(output, self.stem_injector, cond)

    def _make_hook_down(self, i: int):
        injector = self.down_injectors[i]

        def _hook(module: nn.Module, inputs: tuple, output):
            if not self._active:
                return output
            cond = self._get_cond(f"down_{i}")
            if cond is None:
                return output

            if isinstance(output, tuple) and len(output) >= 2:
                hidden = output[0]
                rest = list(output[1:])
                hidden = self._apply(hidden, injector, cond)

                if self.apply_to_down_res_samples:
                    for j, item in enumerate(rest):
                        if isinstance(item, (tuple, list)):
                            rest[j] = tuple(
                                self._apply(t, injector, cond) for t in item
                            )
                return (hidden, *rest)

            if torch.is_tensor(output):
                return self._apply(output, injector, cond)
            return output

        return _hook

    def _hook_mid(self, module: nn.Module, inputs: tuple, output: torch.Tensor):
        if not self._active or self.mid_injector is None:
            return output
        cond = self._get_cond("mid")
        if cond is None:
            return output
        return self._apply(output, self.mid_injector, cond)

    def _make_hook_up(self, i: int):
        injector = self.up_injectors[i]

        def _hook(module: nn.Module, inputs: tuple, output):
            if not self._active:
                return output
            cond = self._get_cond(f"up_{i}")
            if cond is None:
                return output
            if (
                isinstance(output, tuple)
                and len(output) >= 1
                and torch.is_tensor(output[0])
            ):
                hidden = self._apply(output[0], injector, cond)
                return (hidden, *output[1:])
            if torch.is_tensor(output):
                return self._apply(output, injector, cond)
            return output

        return _hook


class Dwi_Cond_LatentUNet(nn.Module):
    """
    - 冻结 teacher UNet（模块一）
    - DWIEncoder 提供多尺度条件特征（模块二）
    - 在 teacher UNet 的 stem/down/mid/up 多处注入 gated cross-attention 残差（宏观深度）

    对外接口尽量与 `cond_ldm/model.py:Dwi_Cond_LatentUNet` 保持一致：
    forward(latents, timesteps, dwi_image, cond=None) -> pred_noise
    """

    def __init__(
        self,
        unet_config: UNetConfig,
        unet_state_dict: dict,
        dwi_in_channels: int = 1,
        cross_attn_heads: int = 4,
        cross_attn_dim_head: int = 64,
        dwi_encoder_kwargs: Optional[dict] = None,
        cross_attn_feat_indices: Tuple[int, ...] = (-1, -2, -3),
        *,
        cond_proj_channels: Optional[int] = None,
        gate_init: float = 1e-3,
        enable_unet_stem: bool = True,
        enable_unet_down: bool = True,
        enable_unet_mid: bool = True,
        enable_unet_up: bool = True,
        apply_to_down_res_samples: bool = True,
        stage_cond_feat_indices: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.unet_config = unet_config
        self.cross_attn_feat_indices = tuple(cross_attn_feat_indices)
        if len(self.cross_attn_feat_indices) == 0:
            raise ValueError("cross_attn_feat_indices 不能为空。")

        self.stage_cond_feat_indices = stage_cond_feat_indices or {}
        gate_init = float(gate_init)

        self.latent_unet = LatentUNet(unet_config)
        self.latent_unet.load_state_dict(unet_state_dict, strict=True)
        for p in self.latent_unet.parameters():
            p.requires_grad = False
        self.latent_unet.eval()

        dwi_encoder_kwargs = dwi_encoder_kwargs or {}
        self.dwi_encoder = DWIEncoder(
            unet_config=unet_config,
            in_channels=dwi_in_channels,
            **dwi_encoder_kwargs,
        )

        boc = list(unet_config.block_out_channels)
        cond_proj_channels = int(cond_proj_channels or boc[-1])
        self.cond_proj_channels = cond_proj_channels

        # 将若干尺度的 DWI feats 统一投影到同一通道数，便于不同注入点复用 cond_channels
        self.cond_projs = nn.ModuleDict()
        self.cond_proj_norms = nn.ModuleDict()
        for idx in self.cross_attn_feat_indices:
            key = str(int(idx))
            self.cond_projs[key] = nn.LazyConv2d(
                cond_proj_channels, kernel_size=1, stride=1, padding=0
            )
            g = min(8, cond_proj_channels)
            if cond_proj_channels % g != 0:
                g = 1
            self.cond_proj_norms[key] = nn.Sequential(
                nn.GroupNorm(num_groups=g, num_channels=cond_proj_channels), nn.SiLU()
            )

        # 入口（latent）注入：保持与旧实现一致的“先注入再送入 UNet”的能力（但加入 gate/zero-init 更稳）
        self.input_injectors = nn.ModuleList(
            [
                GatedCrossAttention2D(
                    query_channels=unet_config.latent_channels,
                    cond_channels=cond_proj_channels,
                    heads=cross_attn_heads,
                    dim_head=cross_attn_dim_head,
                    gate_init=gate_init,
                )
                for _ in self.cross_attn_feat_indices
            ]
        )

        # 宏观深度：UNet 内部多点注入（stem/down/mid/up）
        # 注意：这些注入器必须注册为子模块，否则不会随着 `.to(device)` 移动，且 optimizer 也看不到它们。
        self.macro_stem_injector: Optional[nn.Module] = None
        if enable_unet_stem:
            self.macro_stem_injector = GatedCrossAttention2D(
                query_channels=boc[0],
                cond_channels=cond_proj_channels,
                heads=cross_attn_heads,
                dim_head=cross_attn_dim_head,
                gate_init=gate_init,
            )

        self.macro_down_injectors = nn.ModuleList()
        if enable_unet_down:
            for ch in boc:
                self.macro_down_injectors.append(
                    GatedCrossAttention2D(
                        query_channels=ch,
                        cond_channels=cond_proj_channels,
                        heads=cross_attn_heads,
                        dim_head=cross_attn_dim_head,
                        gate_init=gate_init,
                    )
                )

        self.macro_mid_injector: Optional[nn.Module] = None
        if enable_unet_mid:
            self.macro_mid_injector = GatedCrossAttention2D(
                query_channels=boc[-1],
                cond_channels=cond_proj_channels,
                heads=cross_attn_heads,
                dim_head=cross_attn_dim_head,
                gate_init=gate_init,
            )

        self.macro_up_injectors = nn.ModuleList()
        if enable_unet_up:
            for ch in reversed(boc):
                self.macro_up_injectors.append(
                    GatedCrossAttention2D(
                        query_channels=ch,
                        cond_channels=cond_proj_channels,
                        heads=cross_attn_heads,
                        dim_head=cross_attn_dim_head,
                        gate_init=gate_init,
                    )
                )

        self._macro_injector = _UNetMacroInjector(
            unet=self.latent_unet.unet,
            stem_injector=self.macro_stem_injector,
            down_injectors=self.macro_down_injectors,
            mid_injector=self.macro_mid_injector,
            up_injectors=self.macro_up_injectors,
            apply_to_down_res_samples=apply_to_down_res_samples,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # teacher UNet 永远冻结且保持 eval，避免 dropout 等训练态随机性影响条件分支学习
        self.latent_unet.eval()
        return self

    def _get_feat_by_index(
        self, feats: Sequence[torch.Tensor], idx: int
    ) -> torch.Tensor:
        if not isinstance(feats, (list, tuple)):
            raise TypeError(f"DWIEncoder 输出期望为 list/tuple，当前 {type(feats)}")
        if idx >= len(feats) or idx < -len(feats):
            raise IndexError(f"dwi_feats 长度={len(feats)}，但索引 {idx} 越界。")
        return feats[idx]

    def _project_cond_feats(
        self, dwi_feats: Sequence[torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        projected: Dict[int, torch.Tensor] = {}
        for idx in self.cross_attn_feat_indices:
            key = str(int(idx))
            feat = self._get_feat_by_index(dwi_feats, idx)
            feat = self.cond_projs[key](feat)
            feat = self.cond_proj_norms[key](feat)
            projected[int(idx)] = feat
        return projected

    def _cond_for_key(
        self, projected: Dict[int, torch.Tensor], key: str
    ) -> torch.Tensor:
        # stage_cond_feat_indices 支持：
        # - "stem"/"mid"/"down"/"up"：组级别
        # - "down_0"..."down_k"、"up_0"..."up_k"：更细粒度
        # - "default"：兜底
        default_idx = int(
            self.stage_cond_feat_indices.get("default", self.cross_attn_feat_indices[0])
        )
        idx = int(self.stage_cond_feat_indices.get(key, default_idx))
        if idx not in projected:
            raise KeyError(
                f"stage_cond_feat_indices[{key!r}]={idx} 未包含在 cross_attn_feat_indices={self.cross_attn_feat_indices} 中"
            )
        return projected[idx]

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: Union[int, float, torch.Tensor],
        dwi_image: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents.dim() != 4:
            raise ValueError(f"latents 期望为 [B, C, H, W]，当前 {latents.shape}")
        if dwi_image.dim() != 4:
            raise ValueError(f"dwi_image 期望为 [B, C, H, W]，当前 {dwi_image.shape}")

        dwi_feats = self.dwi_encoder(dwi_image, timesteps)
        projected = self._project_cond_feats(dwi_feats)

        # 入口注入（latent 空间）
        latents_cond = latents
        for injector, idx in zip(self.input_injectors, self.cross_attn_feat_indices):
            latents_cond = injector(latents_cond, projected[int(idx)])

        batch_size = latents_cond.size(0)
        t = LatentUNet._prepare_timesteps(
            timesteps=timesteps,
            batch_size=batch_size,
            device=latents_cond.device,
            dtype=torch.long,
            config=self.unet_config,
        )

        # 宏观深度注入：stem/down/mid/up
        cond_by_key: Dict[str, torch.Tensor] = {}
        if self._macro_injector.stem_injector is not None:
            cond_by_key["stem"] = self._cond_for_key(projected, "stem")

        for i, _ in enumerate(self._macro_injector.down_injectors):
            cond_by_key[f"down_{i}"] = (
                self._cond_for_key(
                    projected,
                    f"down_{i}",
                )
                if f"down_{i}" in self.stage_cond_feat_indices
                else self._cond_for_key(projected, "down")
            )

        if self._macro_injector.mid_injector is not None:
            cond_by_key["mid"] = self._cond_for_key(projected, "mid")

        for i, _ in enumerate(self._macro_injector.up_injectors):
            cond_by_key[f"up_{i}"] = (
                self._cond_for_key(
                    projected,
                    f"up_{i}",
                )
                if f"up_{i}" in self.stage_cond_feat_indices
                else self._cond_for_key(projected, "up")
            )

        with self._macro_injector.use(cond_by_key):
            unet_output = self.latent_unet.unet(latents_cond, t)

        return unet_output.sample
