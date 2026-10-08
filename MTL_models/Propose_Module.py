
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # 新增

# =========================================================
# MSAG: Multi-scale Attention Gate
# 来自 CMUNet，用于增强 skip feature
# =========================================================
class MSAG(nn.Module):
    def __init__(self, channel):
        super(MSAG, self).__init__()

        self.pointwiseConv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(channel),
        )

        self.ordinaryConv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(channel),
        )

        self.dilationConv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.BatchNorm2d(channel),
        )

        self.voteConv = nn.Sequential(
            nn.Conv2d(channel * 3, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.pointwiseConv(x)
        x2 = self.ordinaryConv(x)
        x3 = self.dilationConv(x)

        attn = torch.cat((x1, x2, x3), dim=1)
        attn = self.relu(attn)
        attn = self.voteConv(attn)

        out = x + x * attn
        return out


# =========================================================
# ConvMixer Block
# 放在 bottleneck，用于增强深层语义特征
# =========================================================
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixerBlock(nn.Module):
    def __init__(self, dim=1024, depth=3, k=7):
        super(ConvMixerBlock, self).__init__()

        self.block = nn.Sequential(
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                        nn.Conv2d(
                            dim,
                            dim,
                            kernel_size=k,
                            groups=dim,
                            padding=k // 2),nn.GELU(), nn.BatchNorm2d(dim))
                        ),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim))
                for _ in range(depth)])

    def forward(self, x):
        return self.block(x)


# =========================================================
# Basic Residual Conv Block
# =========================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = out + self.residual(x)
        return self.relu(out)


# =========================================================
# Up Conv
# =========================================================
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


# =========================================================
# Attention Gate for skip connection
# =========================================================
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients):
        super().__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        out = skip_connection * psi
        return out


# =========================================================
# Edge Attention
# alpha 建议从 0.1 开始，不要一开始 0.3
# =========================================================
class EdgeAttention(nn.Module):
    def __init__(self, in_channels, alpha=0.1):
        super().__init__()

        self.alpha = alpha

        self.conv = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feat, edge_map):
        attn = self.conv(edge_map)
        out = feat + self.alpha * feat * attn
        return out


# =========================================================
# Class Semantic Guidance
# 先保留模块，但默认不强行使用，避免分类干扰分割
# =========================================================
class ClassGuidedChannelAttention(nn.Module):
    def __init__(self, feat_channels, cls_dim, alpha=0.1):
        super().__init__()

        self.alpha = alpha

        self.mlp = nn.Sequential(
            nn.Linear(cls_dim, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.Sigmoid()
        )

    def forward(self, feat, cls_feat):
        weight = self.mlp(cls_feat).unsqueeze(-1).unsqueeze(-1)
        out = feat + self.alpha * feat * weight
        return out

