import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================
# 1. Swish
# ======================================
class Swish(nn.Module):
    def __init__(self, beta=1.0, name="swish"):
        super().__init__()
        self.beta = beta
        self.name = name

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


# ======================================
# 2. 基础 1D 卷积块
#    输入输出统一使用 [B, L, C]
# ======================================
class ConvBNAct1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        groups=1,
        dropout_prob=0.0,
        name="conv_bn_act"
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 建议使用奇数"
        padding = (kernel_size - 1) // 2
        self.name = name

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=groups,
            bias=False
        )
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

        self.bn = nn.BatchNorm1d(out_channels)
        self.act = Swish(name=f"{name}_swish")
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # x: [B, L, C]
        x = x.permute(0, 2, 1).contiguous()   # [B, C, L]
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1).contiguous()   # [B, L, C]
        return x


# ======================================
# 3. GRN
# ======================================
class GRN1D(nn.Module):
    def __init__(self, dim, eps=1e-6, name="grn1d"):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))
        self.eps = eps
        self.name = name

    def forward(self, x):
        # x: [B, L, C]
        gx = torch.norm(x, p=2, dim=1, keepdim=True)          # [B, 1, C]
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)  # [B, 1, C]
        return x + self.gamma * (x * nx) + self.beta


# ======================================
# 4. DPRBP 风格双卷积块 + GRN
# ======================================
class DPRBPDoubleConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=192,
        kernel_size=3,
        dropout_prob=0.1,
        name="dprbp_double_conv_block"
    ):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = ConvBNAct1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=1,
            dropout_prob=dropout_prob,
            name=f"{name}_conv1"
        )
        self.conv2 = ConvBNAct1D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=1,
            dropout_prob=dropout_prob,
            name=f"{name}_conv2"
        )
        self.grn = GRN1D(out_channels, name=f"{name}_grn")

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Linear(in_channels, out_channels, bias=False)
            nn.init.xavier_uniform_(self.shortcut.weight)

        self.final_act = Swish(name=f"{name}_final_swish")

    def forward(self, x):
        # x: [B, L, Cin]
        residual = self.shortcut(x)      # [B, L, Cout]
        main = self.conv1(x)
        main = self.conv2(main)
        
        feat = self.final_act(residual + main)
        return feat


# ======================================
# 5. 门控选择性压缩块
#    核心思想：
#    - feature branch: 提取候选判别特征
#    - gate branch: 在 GRN 校准后生成重要性信号
#    - 同一 gate 同时用于特征重加权与后续下采样指导
# ======================================
class GatedSelectiveCompressionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=192,
        kernel_size=3,
        gate_kernel_size=3,
        dropout_prob=0.1,
        name="gated_selective_compression_block"
    ):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ---------- feature branch ----------
        self.conv1 = ConvBNAct1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=1,
            dropout_prob=dropout_prob,
            name=f"{name}_feat_conv1"
        )
        self.conv2 = ConvBNAct1D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=1,
            dropout_prob=dropout_prob,
            name=f"{name}_feat_conv2"
        )

        # ---------- gate branch ----------
        self.gate_proj = nn.Linear(in_channels, out_channels, bias=False)
        nn.init.xavier_uniform_(self.gate_proj.weight)

        #self.gate_grn = GRN1D(out_channels, name=f"{name}_gate_grn")
        self.gate_grn = nn.Identity()
        self.gate_dw = ConvBNAct1D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=gate_kernel_size,
            groups=out_channels,
            dropout_prob=0.0,
            name=f"{name}_gate_dw"
        )
        self.gate_pw = nn.Linear(out_channels, out_channels, bias=True)
        nn.init.xavier_uniform_(self.gate_pw.weight)
        nn.init.constant_(self.gate_pw.bias, 1.0)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Linear(in_channels, out_channels, bias=False)
            nn.init.xavier_uniform_(self.shortcut.weight)

        self.final_act = Swish(name=f"{name}_final_swish")

    def forward(self, x, return_gate=False):
        # x: [B, L, Cin]
        residual = self.shortcut(x)                  # [B, L, Cout]

        # feature branch
        feat = self.conv1(x)
        feat = self.conv2(feat)                      # [B, L, Cout]

        # gate branch
        gate = self.gate_proj(x)                     # [B, L, Cout]
        gate = self.gate_grn(gate)
        gate = self.gate_dw(gate)
        gate = self.gate_pw(gate)
        gate = torch.sigmoid(gate)                   # [B, L, Cout], in (0, 1)

        fused = residual + feat * gate
        out = self.final_act(fused)

        if return_gate:
            return out, gate
        return out



# ======================================
# 7. 标准 MaxPool 下采样
# ======================================
class StandardPeakMaxPool1D(nn.Module):
    def __init__(
        self,
        channels,
        pool_kernel_size=3,
        stride=2,
        name="standard_peak_maxpool"
    ):
        super().__init__()
        self.name = name
        self.channels = channels
        self.pool_kernel_size = pool_kernel_size
        self.stride = stride

    def _pad_right(self, x_chw):
        pad_right = max(self.pool_kernel_size - self.stride, 0)
        return F.pad(x_chw, (0, pad_right), mode="constant", value=0.0)

    def forward(self, feat, external_importance=None):
        # feat: [B, L, C]
        x = feat.permute(0, 2, 1).contiguous()  # [B, C, L]
        x = self._pad_right(x)
        x = F.max_pool1d(
            x,
            kernel_size=self.pool_kernel_size,
            stride=self.stride,
            padding=0
        )
        x = x.permute(0, 2, 1).contiguous()  # [B, L', C]
        return x


# ======================================
# 8. LIP 风格 1D 下采样
#    支持外部 importance 引导：
#    - 原始 learned weight: 由本层轻量卷积生成
#    - external_importance: 由 gate branch 提供
#    最终 weight = learned_weight * external_importance
# ======================================
class LIPPool1D(nn.Module):
    def __init__(
        self,
        channels,
        pool_kernel_size=3,
        stride=2,
        logit_kernel_size=3,
        eps=1e-6,
        use_bn=False,
        name="lip_pool1d"
    ):
        super().__init__()
        self.name = name
        self.channels = channels
        self.pool_kernel_size = pool_kernel_size
        self.stride = stride
        self.eps = eps

        assert logit_kernel_size % 2 == 1
        logit_padding = (logit_kernel_size - 1) // 2

        self.logit_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=logit_kernel_size,
            stride=1,
            padding=logit_padding,
            groups=channels,
            bias=True
        )
        nn.init.kaiming_normal_(self.logit_conv.weight, mode="fan_out", nonlinearity="relu")
        if self.logit_conv.bias is not None:
            nn.init.zeros_(self.logit_conv.bias)

        self.use_bn = use_bn
        if use_bn:
            self.logit_bn = nn.BatchNorm1d(channels)

    def _pad_right(self, x_chw):
        pad_right = max(self.pool_kernel_size - self.stride, 0)
        return F.pad(x_chw, (0, pad_right), mode="constant", value=0.0)

    def forward(self, feat, external_importance=None):
        """
        feat: [B, L, C]
        external_importance: [B, L, C] or None
        return: [B, L', C]
        """
        external_importance=None
        x = feat.permute(0, 2, 1).contiguous()  # [B, C, L]

        # learned local importance from current feature
        logit = self.logit_conv(x)              # [B, C, L]
        if self.use_bn:
            logit = self.logit_bn(logit)
        learned_weight = torch.exp(logit)       # > 0

        # external gate guidance (来自 block 的 gate branch)
        if external_importance is not None:
            ext = external_importance.permute(0, 2, 1).contiguous()   # [B, C, L]
            ext = ext.clamp_min(self.eps)
            weight =  learned_weight * ext#*ext #learned_weight *
        else:
            weight = learned_weight

        x_pad = self._pad_right(x)
        w_pad = self._pad_right(weight)

        num = F.avg_pool1d(
            x_pad * w_pad,
            kernel_size=self.pool_kernel_size,
            stride=self.stride,
            padding=0,
            count_include_pad=False
        )

        den = F.avg_pool1d(
            w_pad,
            kernel_size=self.pool_kernel_size,
            stride=self.stride,
            padding=0,
            count_include_pad=False
        )

        out = num / (den + self.eps)            # [B, C, L']
        return out.permute(0, 2, 1).contiguous()  # [B, L', C]


# ======================================
# 9. DPRBPStyleMSRBPyramid
#
# block_type:
#   - "pcse"
#   - "dprbp"
#   - "gsc"   # 新增：门控选择性压缩块
#
# downsample_type:
#   - "maxpool"
#   - "lip"
# ======================================
class DPRBPStyleMSRBPyramid(nn.Module):
    def __init__(
        self,
        in_channels,
        num_scales=3,
        share_cbam_params=False,
        out_channels=192,
        hidden_channels=192,
        local_kernel_size=3,
        context_kernel_size=5,
        dropout_prob=0.1,
        block_type="dprbp",      # "pcse" or "dprbp" or "gsc"
        downsample_type="lip",   # "lip" or "maxpool"
        name="dprbp_style_msrb_pyramid"
    ):
        super().__init__()
        self.name = name
        self.num_scales = num_scales
        self.share_cbam_params = share_cbam_params
        self.out_channels = out_channels
        self.block_type = block_type
        self.downsample_type = downsample_type

        if block_type not in {"pcse", "dprbp", "gsc"}:
            raise ValueError("block_type must be 'pcse', 'dprbp', or 'gsc'")
        if downsample_type not in {"lip", "maxpool"}:
            raise ValueError("downsample_type must be 'lip' or 'maxpool'")

        self.blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        current_in_channels = in_channels
        for scale_idx in range(num_scales):
            
            if block_type == "dprbp":
                block = DPRBPDoubleConvBlock(
                    in_channels=current_in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    dropout_prob=dropout_prob,
                    name=f"{name}_dprbp_scale_{scale_idx}"
                )
            else:
                block = GatedSelectiveCompressionBlock(
                    in_channels=current_in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    gate_kernel_size=3,
                    dropout_prob=dropout_prob,
                    name=f"{name}_gsc_scale_{scale_idx}"
                )
            self.blocks.append(block)

            if scale_idx < num_scales - 1:
                if downsample_type == "lip":
                    downsample = LIPPool1D(
                        channels=out_channels,
                        pool_kernel_size=3,
                        stride=2,
                        logit_kernel_size=3,
                        use_bn=False,
                        name=f"{name}_lip_scale_{scale_idx}"
                    )
                else:
                    downsample = StandardPeakMaxPool1D(
                        channels=out_channels,
                        pool_kernel_size=3,
                        stride=2,
                        name=f"{name}_maxpool_scale_{scale_idx}"
                    )
                self.downsample_blocks.append(downsample)

            current_in_channels = out_channels

    def forward(self, x, return_all_scales=False, return_gate_maps=False):
        """
        x: [B, L, C]
        """
        scale_features = []
        gate_maps = []
        current_x = x

        for scale_idx in range(self.num_scales):
            if self.block_type == "gsc":
                feat, gate = self.blocks[scale_idx](current_x, return_gate=True)
                gate_maps.append(gate)
            else:
                feat = self.blocks[scale_idx](current_x)
                gate = None

            scale_features.append(feat)

            if scale_idx < self.num_scales - 1:
                if self.block_type == "gsc" and self.downsample_type == "lip":
                    current_x = self.downsample_blocks[scale_idx](feat, external_importance=gate)
                else:
                    current_x = self.downsample_blocks[scale_idx](feat)

        if return_all_scales and return_gate_maps:
            return scale_features, gate_maps
        if return_all_scales:
            return scale_features
        if return_gate_maps:
            return scale_features[-1], gate_maps
        return scale_features[-1]


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return (x * weights).sum(dim=1), weights.squeeze(-1)


class SampleGatedExpert(nn.Module):
    """One DAMF candidate branch."""

    def __init__(self, group_dims, out_dim):
        super().__init__()
        self.out_dim = int(out_dim)
        self.group_dims = [int(dim) for dim in group_dims]
        self.num_groups = len(self.group_dims)
        self.last_gate_mean = [1.0 for _ in range(self.num_groups)]
        self.gate_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2, 8),
                    nn.GELU(),
                    nn.Linear(8, 1),
                )
                for _ in self.group_dims
            ]
        )
        self.out_norm = nn.LayerNorm(self.out_dim)

    def forward(self, groups):
        gate_logits = []
        for group, gate_head in zip(groups, self.gate_heads):
            summary = torch.cat(
                [
                    group.mean(dim=-1, keepdim=True),
                    group.var(dim=-1, unbiased=False, keepdim=True).sqrt(),
                ],
                dim=-1,
            )
            gate_logits.append(gate_head(summary))

        gates = torch.softmax(torch.cat(gate_logits, dim=-1), dim=-1)
        fused_groups = []
        gate_means = []
        for idx, group in enumerate(groups):
            gate = gates[..., idx : idx + 1]
            gate_means.append(float(gate.mean().detach().cpu().item()))
            fused_groups.append(group * gate)
        self.last_gate_mean = gate_means
        return self.out_norm(torch.cat(fused_groups, dim=-1))


class DualLevelAdaptiveFusion(nn.Module):
    def __init__(self, out_dim, num_experts=3, router_temperature=1.5):
        super().__init__()
        self.group_names = ["kmer1", "kmer2", "kmer3", "ncp", "dpcp", "circ2vec"]
        self.group_dims = [4, 16, 64, 3, 11, 30]
        self.num_groups = len(self.group_names)
        self.out_dim = int(out_dim)
        self.num_experts = int(num_experts)
        self.router_temperature = float(router_temperature)
        self.last_router_mean = [0.0 for _ in range(self.num_experts)]
        self.last_router_entropy = 0.0
        self.last_expert_gate_means = [
            [0.0 for _ in range(self.num_groups)] for _ in range(self.num_experts)
        ]
        self.router = nn.Sequential(
            nn.Linear(self.num_groups * 2, 32),
            nn.GELU(),
            nn.Linear(32, self.num_experts),
        )
        self.experts = nn.ModuleList(
            [SampleGatedExpert(self.group_dims, self.out_dim) for _ in range(self.num_experts)]
        )
        self.fused_out_norm = nn.LayerNorm(self.out_dim)

    def forward(self, batch):
        groups = [
            batch.kmer1,
            batch.kmer2,
            batch.kmer3,
            batch.ncp,
            batch.dpcp,
            batch.circ2vec_embed,
        ]
        router_summary = torch.cat(
            [group.mean(dim=-1, keepdim=True) for group in groups]
            + [group.var(dim=-1, unbiased=False, keepdim=True).sqrt() for group in groups],
            dim=-1,
        )
        logits = self.router(router_summary) / max(self.router_temperature, 1e-6)
        router_weights = torch.softmax(logits, dim=-1)
        expert_outputs = torch.stack([expert(groups) for expert in self.experts], dim=1)
        mixture_weights = router_weights.transpose(1, 2).unsqueeze(-1)
        fused = self.fused_out_norm((expert_outputs * mixture_weights).sum(dim=1))

        self.last_router_mean = router_weights.mean(dim=(0, 1)).detach().cpu().tolist()
        entropy = -(router_weights * (router_weights + 1e-8).log()).sum(dim=-1).mean()
        self.last_router_entropy = float(entropy.detach().cpu().item())
        self.last_expert_gate_means = [list(expert.last_gate_mean) for expert in self.experts]
        return fused


class MAPNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.damf = DualLevelAdaptiveFusion(
            out_dim=config["fusion_out_dim"],
            num_experts=config["num_experts"],
            router_temperature=config["router_temperature"],
        )
        self.glip = DPRBPStyleMSRBPyramid(
            in_channels=128,
            num_scales=int(config["num_scales"]),
            share_cbam_params=False,
            name="seq_pyramid",
            block_type="gsc",
            downsample_type="lip",
        )
        # Retained to preserve the initialization sequence used in the experiments.
        self.pro = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, batch):
        return self.glip(self.damf(batch), return_all_scales=False)

    def debug_stats(self):
        return {
            "router_mean": list(self.damf.last_router_mean),
            "router_entropy": self.damf.last_router_entropy,
            "expert_gate_means": [list(row) for row in self.damf.last_expert_gate_means],
        }


class MAPNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.encoder = MAPNetEncoder(self.config)
        self.seq_proj = nn.Linear(self.config["seq_input_dim"], self.config["token_dim"])
        self.seq_pool = AttentionPool(self.config["token_dim"])
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.config["token_dim"]),
            nn.Linear(self.config["token_dim"], self.config["token_dim"]),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Dropout(0.1),
            nn.Linear(self.config["token_dim"], self.config["num_classes"]),
        )

    def forward(self, batch):
        seq_feature = self.encoder(batch)
        seq_tokens = self.seq_proj(seq_feature)
        pooled, _ = self.seq_pool(seq_tokens)
        return self.classifier(pooled)

    def debug_stats(self):
        return self.encoder.debug_stats()
