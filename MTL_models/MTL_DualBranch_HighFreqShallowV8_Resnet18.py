import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from MTL_models.Propose_Module import (
    MSAG,
    ConvMixerBlock,
    ConvBlock,
    UpConv,
    AttentionBlock,
    EdgeAttention,
)


class HighFrequencyDetail(nn.Module):
    """Normalized high-frequency residual with reflection-padded blur."""

    def __init__(self, channels=1, kernel_size=5, strength=1.0):
        super().__init__()
        self.strength = strength
        self.pad = kernel_size // 2
        weight = torch.ones(channels, 1, kernel_size, kernel_size, dtype=torch.float32)
        weight = weight / weight[0, 0].numel()
        self.register_buffer("weight", weight)
        self.channels = channels

    def forward(self, x):
        if self.strength <= 0:
            return torch.zeros_like(x)
        padded = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        blurred = F.conv2d(padded, self.weight, padding=0, groups=self.channels)
        detail = x - blurred
        scale = detail.flatten(1).std(dim=1, keepdim=True).clamp_min(1e-4)
        detail = detail / scale.view(-1, 1, 1, 1)
        return torch.tanh(self.strength * detail)


class SharedInputStem(nn.Module):
    def __init__(self, in_channels=1, stem_channels=16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def _group_count(channels):
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class DetailConvBlock(nn.Module):
    """Lightweight detail block with GroupNorm, independent of raw-image BN."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LightweightDetailEncoder(nn.Module):
    """Produce only e1/e2-aligned detail scales; deeper semantics stay raw-only."""

    def __init__(self, in_channels, feature_channels):
        super().__init__()
        c1, c2 = feature_channels[:2]
        self.stage1 = DetailConvBlock(in_channels, c1, stride=2)
        self.stage2 = DetailConvBlock(c1, c2, stride=2)

    def forward(self, x):
        h1 = self.stage1(x)
        h2 = self.stage2(h1)
        return h1, h2


class ResidualDetailFusionBlock(nn.Module):
    """Inject detail as a bounded residual; gamma=0 makes this an exact identity."""

    def __init__(self, channels, raw_branch_bias=0.50, fusion_strength=1.0):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.fusion_strength = fusion_strength
        self.raw_branch_bias = float(raw_branch_bias)
        if not 0.0 <= self.raw_branch_bias <= 1.0:
            raise ValueError("raw_branch_bias must be in [0, 1]")
        self.residual = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat_raw, feat_detail):
        # At the paper default raw_branch_bias=0.50, this is the direct
        # concatenation Concat(feat_raw, feat_detail). Non-default values are
        # retained only for backward compatibility with legacy checkpoints.
        pair = torch.cat([
            2.0 * self.raw_branch_bias * feat_raw,
            2.0 * (1.0 - self.raw_branch_bias) * feat_detail,
        ], dim=1)
        residual = self.residual(pair)
        residual = residual * self.spatial_gate(pair) * self.channel_gate(pair)
        scale = self.fusion_strength * torch.tanh(self.gamma)
        return feat_raw + scale * residual


class DirectDetailFusionBlock(nn.Module):
    """Parameter-free control for testing adaptive fusion against direct addition."""

    def __init__(self, fusion_strength=1.0):
        super().__init__()
        self.fusion_strength = float(fusion_strength)

    def forward(self, feat_raw, feat_detail):
        return feat_raw + self.fusion_strength * feat_detail


class DualBranchHighFreqShallowV8_Resnet18(nn.Module):
    def __init__(
        self,
        img_ch=1,
        output_ch=1,
        cls_classes=3,
        use_cls=True,
        stem_channels=16,
        mixer_depth=4,
        mixer_kernel=7,
        edge_alpha=0.1,
        backbone="resnet18",
        use_mixer=True,
        use_msag=True,
        use_att_gate=True,
        use_edge_attention=True,
        detail_strength=1.0,
        raw_branch_bias=0.50,
        fusion_strength=1.0,
        use_hfd=True,
        use_detail_fusion=True,
        hfd_input_injection=False,
        fusion_mode="adf",
    ):
        super().__init__()
        self.use_cls = use_cls
        self.use_mixer = use_mixer
        self.use_msag = use_msag
        self.use_att_gate = use_att_gate
        self.use_edge_attention = use_edge_attention
        self.raw_branch_bias = raw_branch_bias
        self.fusion_strength = fusion_strength
        self.use_hfd = bool(use_hfd)
        self.use_detail_fusion = bool(use_detail_fusion)
        self.hfd_input_injection = bool(hfd_input_injection)
        self.fusion_mode = str(fusion_mode)
        if self.fusion_mode not in {"none", "direct_add", "adf"}:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")
        if not self.use_hfd:
            self.use_detail_fusion = False
            self.hfd_input_injection = False
            self.fusion_mode = "none"
        elif not self.use_detail_fusion:
            self.fusion_mode = "none"

        encoder_in_channels = stem_channels if stem_channels > 1 else img_ch
        self.detail = HighFrequencyDetail(channels=img_ch, kernel_size=5, strength=detail_strength) if use_hfd else None
        self.shared_stem = (
            SharedInputStem(in_channels=img_ch, stem_channels=stem_channels)
            if stem_channels > 1
            else nn.Identity()
        )
        self.encoder = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True,
            in_chans=encoder_in_channels,
        )
        self.enc_channels = self.encoder.feature_info.channels()
        if self.use_detail_fusion:
            self.detail_encoder = LightweightDetailEncoder(img_ch, self.enc_channels)
            if self.fusion_mode == "direct_add":
                self.fuse1 = DirectDetailFusionBlock(fusion_strength=fusion_strength)
                self.fuse2 = DirectDetailFusionBlock(fusion_strength=fusion_strength)
            else:
                self.fuse1 = ResidualDetailFusionBlock(
                    self.enc_channels[0], raw_branch_bias=raw_branch_bias,
                    fusion_strength=fusion_strength,
                )
                self.fuse2 = ResidualDetailFusionBlock(
                    self.enc_channels[1], raw_branch_bias=raw_branch_bias,
                    fusion_strength=fusion_strength,
                )
        else:
            self.detail_encoder = None
            self.fuse1 = None
            self.fuse2 = None

        self.msag1 = MSAG(self.enc_channels[0])
        self.msag2 = MSAG(self.enc_channels[1])
        self.msag3 = MSAG(self.enc_channels[2])
        self.msag4 = MSAG(self.enc_channels[3])

        self.ConvMixer = ConvMixerBlock(
            dim=self.enc_channels[4],
            depth=mixer_depth,
            k=mixer_kernel,
        )

        self.Up5 = UpConv(self.enc_channels[4], self.enc_channels[3])
        self.Att4 = AttentionBlock(self.enc_channels[3], self.enc_channels[3], 512)
        self.UpConv5 = ConvBlock(self.enc_channels[3] * 2, self.enc_channels[3])

        self.Up4 = UpConv(self.enc_channels[3], self.enc_channels[2])
        self.Att3 = AttentionBlock(self.enc_channels[2], self.enc_channels[2], 256)
        self.UpConv4 = ConvBlock(self.enc_channels[2] * 2, self.enc_channels[2])

        self.Up3 = UpConv(self.enc_channels[2], self.enc_channels[1])
        self.Att2 = AttentionBlock(self.enc_channels[1], self.enc_channels[1], 128)
        self.UpConv3 = ConvBlock(self.enc_channels[1] * 2, self.enc_channels[1])

        self.Up2 = UpConv(self.enc_channels[1], self.enc_channels[0])
        self.Att1 = AttentionBlock(self.enc_channels[0], self.enc_channels[0], 32)
        self.UpConv2 = ConvBlock(self.enc_channels[0] * 2, self.enc_channels[0])

        self.seg_head = nn.Conv2d(self.enc_channels[0], output_ch, kernel_size=1)
        self.edge_head = nn.Conv2d(self.enc_channels[0], 1, kernel_size=1)
        self.aux_head2 = nn.Conv2d(self.enc_channels[1], output_ch, kernel_size=1)
        self.aux_head3 = nn.Conv2d(self.enc_channels[2], output_ch, kernel_size=1)
        self.aux_head4 = nn.Conv2d(self.enc_channels[3], output_ch, kernel_size=1)
        self.aux_outputs = ()

        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_fc1 = nn.Linear(self.enc_channels[4], 256)
        self.cls_relu = nn.ReLU(inplace=True)
        self.cls_drop = nn.Dropout(0.3)
        self.cls_fc2 = nn.Linear(256, cls_classes)

        self.edge_att = EdgeAttention(self.enc_channels[0], alpha=edge_alpha)

    def _encode_branch(self, x):
        x = self.shared_stem(x)
        return self.encoder(x)

    def forward(self, x, use_edge_attention=True, return_feat=False):
        input_size = x.shape[-2:]
        detail = self.detail(x) if self.use_hfd else None
        encoder_input = x + detail if self.hfd_input_injection else x
        raw_feats = self._encode_branch(encoder_input)

        if self.use_detail_fusion:
            detail_feats = self.detail_encoder(detail)
            detail_feats = tuple(
                feat if feat.shape[-2:] == raw_feats[index].shape[-2:]
                else F.interpolate(feat, size=raw_feats[index].shape[-2:], mode="bilinear", align_corners=False)
                for index, feat in enumerate(detail_feats)
            )
            e1 = self.fuse1(raw_feats[0], detail_feats[0])
            e2 = self.fuse2(raw_feats[1], detail_feats[1])
        else:
            e1, e2 = raw_feats[0], raw_feats[1]
        e3 = raw_feats[2]
        e4 = raw_feats[3]
        raw_deepest = raw_feats[4]
        deepest = raw_deepest

        if self.use_msag:
            e1 = self.msag1(e1)
            e2 = self.msag2(e2)
            e3 = self.msag3(e3)
            e4 = self.msag4(e4)

        if self.use_mixer:
            deepest = self.ConvMixer(deepest)

        cls_feat = self.cls_pool(raw_deepest).flatten(1)
        cls_feat = self.cls_fc1(cls_feat)
        cls_feat = self.cls_relu(cls_feat)
        cls_feat = self.cls_drop(cls_feat)
        cls_out = self.cls_fc2(cls_feat)

        d4 = self.Up5(deepest)
        if d4.shape[-2:] != e4.shape[-2:]:
            d4 = F.interpolate(d4, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        s4 = self.Att4(gate=d4, skip_connection=e4) if self.use_att_gate else e4
        d4 = self.UpConv5(torch.cat((s4, d4), dim=1))

        d3 = self.Up4(d4)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        s3 = self.Att3(gate=d3, skip_connection=e3) if self.use_att_gate else e3
        d3 = self.UpConv4(torch.cat((s3, d3), dim=1))

        d2 = self.Up3(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        s2 = self.Att2(gate=d2, skip_connection=e2) if self.use_att_gate else e2
        d2 = self.UpConv3(torch.cat((s2, d2), dim=1))

        d1 = self.Up2(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        s1 = self.Att1(gate=d1, skip_connection=e1) if self.use_att_gate else e1
        d1 = self.UpConv2(torch.cat((s1, d1), dim=1))

        self.aux_outputs = (
            F.interpolate(self.aux_head2(d2), size=input_size, mode="bilinear", align_corners=False),
            F.interpolate(self.aux_head3(d3), size=input_size, mode="bilinear", align_corners=False),
            F.interpolate(self.aux_head4(d4), size=input_size, mode="bilinear", align_corners=False),
        )

        edge_out = self.edge_head(d1)
        seg_feat = d1
        if use_edge_attention and self.use_edge_attention:
            seg_feat = self.edge_att(seg_feat, torch.sigmoid(edge_out))
        seg_out = self.seg_head(seg_feat)

        if edge_out.shape[-2:] != input_size:
            edge_out = F.interpolate(edge_out, size=input_size, mode="bilinear", align_corners=False)
        if seg_out.shape[-2:] != input_size:
            seg_out = F.interpolate(seg_out, size=input_size, mode="bilinear", align_corners=False)

        if return_feat:
            return seg_out, edge_out, cls_out, d1, deepest
        if self.use_cls:
            return seg_out, edge_out, cls_out
        return seg_out, edge_out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DualBranchHighFreqShallowV8_Resnet18().to(device)
    x = torch.randn(2, 1, 256, 256).to(device)
    seg_out, edge_out, cls_out = model(x)
    print("Input:", x.shape)
    print("seg_out:", seg_out.shape)
    print("edge_out:", edge_out.shape)
    print("cls_out:", cls_out.shape)
