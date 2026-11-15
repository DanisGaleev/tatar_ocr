"""
v3.architecture
Architectural definitions for Tatar OCR Net v3.
Features:
- Residual skip connections with projection shortcuts
- Squeeze-and-Excitation (SE) channel attention modules for fine diacritic stroke discrimination
- Parameter budget constrained to ~750K - 950K parameters (near parity with v2's 822K)
- Fast on-device inference compatible with PyTorch FP32, ONNX FP32, and ONNX INT8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention module."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid_channels = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, mid_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid_channels, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = x.mean(dim=(2, 3))  # Global average pooling -> [B, C]
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class ResSEBlock(nn.Module):
    """
    Standard Residual block with Squeeze-and-Excitation attention.
    Maintains `.se.` and `.shortcut.` parameter paths for automated scanner detection.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels, reduction=reduction)
        self.act2 = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out = out + res
        return self.act2(out)


# ---------------------------------------------------------------------------
# Candidate 1: TatarOCRNetV3_ResSE (Standard Deep ResSE-Net)
# ---------------------------------------------------------------------------

class TatarOCRNetV3_ResSE(nn.Module):
    """
    Deep Residual network with SE channel attention at each stage.
    ~939K parameters (v2 baseline is 822K).
    """
    def __init__(self, num_classes: int = 39):
        super().__init__()
        # Stem: 64x64 -> 64x64
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 64x64 -> 32x32
        self.stage1 = nn.Sequential(
            ResSEBlock(32, 40, stride=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.08)
        )

        # Stage 2: 32x32 -> 16x16
        self.stage2 = nn.Sequential(
            ResSEBlock(40, 64, stride=1),
            ResSEBlock(64, 64, stride=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.12)
        )

        # Stage 3: 16x16 -> 8x8
        self.stage3 = nn.Sequential(
            ResSEBlock(64, 104, stride=1),
            ResSEBlock(104, 104, stride=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15)
        )

        # Stage 4: 8x8 -> 4x4
        self.stage4 = nn.Sequential(
            ResSEBlock(104, 136, stride=1),
            nn.AdaptiveAvgPool2d((2, 2))
        )

        # Classifier: 136 * 2 * 2 = 544 -> 160 -> num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(136 * 2 * 2, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(160, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Candidate 2: TatarOCRNetV3_Mobile (Inverted Residuals + SE)
# ---------------------------------------------------------------------------

class InvertedResSEBlock(nn.Module):
    """Inverted Residual block with SE and shortcut."""
    def __init__(self, in_ch: int, out_ch: int, exp_ratio: int = 3, stride: int = 1):
        super().__init__()
        self.use_res = (stride == 1 and in_ch == out_ch)
        hidden_ch = in_ch * exp_ratio

        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU6(inplace=True)
        ) if exp_ratio != 1 else nn.Identity()

        self.dw = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=stride, padding=1, groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU6(inplace=True)
        )

        self.se = SEBlock(hidden_ch, reduction=6)

        self.project = nn.Sequential(
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        if not self.use_res:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.expand(x)
        out = self.dw(out)
        out = self.se(out)
        out = self.project(out)
        return out + res


class TatarOCRNetV3_Mobile(nn.Module):
    """
    Lightweight Mobile inverted residual architecture with SE.
    ~756K parameters.
    """
    def __init__(self, num_classes: int = 39):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        self.stage1 = nn.Sequential(
            InvertedResSEBlock(32, 48, exp_ratio=2, stride=1),
            nn.MaxPool2d(2, 2)  # 32x32
        )
        self.stage2 = nn.Sequential(
            InvertedResSEBlock(48, 80, exp_ratio=3, stride=1),
            InvertedResSEBlock(80, 80, exp_ratio=3, stride=1),
            nn.MaxPool2d(2, 2)  # 16x16
        )
        self.stage3 = nn.Sequential(
            InvertedResSEBlock(80, 128, exp_ratio=3, stride=1),
            InvertedResSEBlock(128, 128, exp_ratio=3, stride=1),
            nn.MaxPool2d(2, 2)  # 8x8
        )
        self.stage4 = nn.Sequential(
            InvertedResSEBlock(128, 176, exp_ratio=4, stride=1),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(176 * 2 * 2, 192),
            nn.BatchNorm1d(192),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(192, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Candidate 3: TatarOCRNetV3_MultiScale (Dual-Scale Dilated ResSE)
# ---------------------------------------------------------------------------

class MultiScaleResSEBlock(nn.Module):
    """
    Dual-branch residual block:
    Branch A: Standard 3x3 conv (local strokes, diacritic dots)
    Branch B: Dilated 3x3 conv (dilation=2, receptive field 5x5 for letter silhouettes)
    Fused with SE channel attention.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        mid_ch = out_channels // 2
        self.conv_std = nn.Conv2d(in_channels, mid_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_std = nn.BatchNorm2d(mid_ch)

        self.conv_dil = nn.Conv2d(in_channels, out_channels - mid_ch, kernel_size=3, stride=stride, padding=2, dilation=2, bias=False)
        self.bn_dil = nn.BatchNorm2d(out_channels - mid_ch)

        self.act1 = nn.ReLU(inplace=True)

        self.conv_fuse = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels, reduction=8)
        self.act2 = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        a = self.bn_std(self.conv_std(x))
        b = self.bn_dil(self.conv_dil(x))
        out = self.act1(torch.cat([a, b], dim=1))

        out = self.bn_fuse(self.conv_fuse(out))
        out = self.se(out)

        return self.act2(out + res)


class TatarOCRNetV3_MultiScale(nn.Module):
    """
    Multi-Scale ResSE architecture capturing both stroke-level serifs and overall glyph geometry.
    ~937K parameters.
    """
    def __init__(self, num_classes: int = 39):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(
            MultiScaleResSEBlock(32, 40, stride=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.08)
        )
        self.stage2 = nn.Sequential(
            MultiScaleResSEBlock(40, 64, stride=1),
            MultiScaleResSEBlock(64, 64, stride=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.12)
        )
        self.stage3 = nn.Sequential(
            MultiScaleResSEBlock(64, 104, stride=1),
            MultiScaleResSEBlock(104, 104, stride=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15)
        )
        self.stage4 = nn.Sequential(
            MultiScaleResSEBlock(104, 136, stride=1),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(136 * 2 * 2, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(160, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.classifier(x)
        return x


# Default alias for pipeline integration
TatarOCRNetV3 = TatarOCRNetV3_MultiScale


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    dummy = torch.randn(2, 1, 64, 64)
    models = {
        "ResSE": TatarOCRNetV3_ResSE(),
        "Mobile": TatarOCRNetV3_Mobile(),
        "MultiScale": TatarOCRNetV3_MultiScale(),
    }
    print("=" * 65)
    print("Tatar OCR v3 Architecture Candidates Evaluation")
    print("=" * 65)
    for name, m in models.items():
        params = count_parameters(m)
        out = m(dummy)
        state_dict_keys = list(m.state_dict().keys())
        has_se_or_shortcut = any(".se." in k or ".shortcut." in k for k in state_dict_keys)
        print(f"  • {name:12s} | Params: {params:9,d} ({params*4/(1024*1024):.2f} MB FP32, ~{params/(1024):.1f} KB INT8) | Scanner Hook: {has_se_or_shortcut}")
    print("=" * 65)
