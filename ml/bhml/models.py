"""Neural network models for the black-hole ML tasks.

A small MLP for the photon capture classifier and the deflection surrogate, and,
for the emission -> image operator, a 2D Fourier Neural Operator with a U-Net
baseline. The FNO and the U-Net follow the from-scratch versions in
https://github.com/amark-23/fno-pde.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """A small fully-connected network that outputs a single logit.

    For binary classification: apply torch.sigmoid to the output to get
    P(capture). Trained with BCEWithLogitsLoss, which expects the raw logit.
    """

    def __init__(self, in_dim: int = 2, hidden: tuple[int, ...] = (32, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]  # final logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # shape (N,) logits


# --------------------------------------------------------------------------- #
# Fourier Neural Operator
# --------------------------------------------------------------------------- #

class SpectralConv2d(nn.Module):
    """Learned multiply in Fourier space, keeping the lowest (modes1, modes2) modes.

    The weights act on frequencies, not grid points, so the layer is defined at
    any resolution: that is what lets one trained FNO run on finer grids.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_channels * out_channels)
        # Two blocks: the low positive-kx and low negative-kx corners of the spectrum.
        self.weight1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weight2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, H, W)
        batch, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)                                   # (batch, in, H, W//2+1)
        out_ft = torch.zeros(batch, self.weight1.shape[1], H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        m1, m2 = self.modes1, self.modes2
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2],
                                              self.weight1)
        out_ft[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2],
                                               self.weight2)
        return torch.fft.irfft2(out_ft, s=(H, W))


class FNO2d(nn.Module):
    """2D FNO: lift -> [spectral conv + 1x1 conv + GELU] x depth -> project.

    Images are not periodic, but the FFT assumes they are, so the lifted field is
    zero-padded by a fixed FRACTION of the grid (pad_frac) before the spectral
    layers and cropped after. Padding by a fraction, not a pixel count, keeps the
    padded domain the same physical size at every resolution, so each Fourier
    mode means the same thing on a 64x64 grid and a 256x256 one.

    Input (batch, H, W, in_channels) -> output (batch, H, W, out_channels).
    """

    def __init__(self, modes1: int = 12, modes2: int = 12, width: int = 32, depth: int = 4,
                 in_channels: int = 5, out_channels: int = 1, pad_frac: float = 0.125):
        super().__init__()
        self.pad_frac = pad_frac
        self.fc_in = nn.Linear(in_channels, width)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(depth)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.fc_out = nn.Sequential(nn.Linear(width, 128), nn.GELU(),
                                    nn.Linear(128, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape
        x = self.fc_in(x).permute(0, 3, 1, 2)                       # (batch, width, H, W)
        ph, pw = round(self.pad_frac * H), round(self.pad_frac * W)
        x = F.pad(x, (0, pw, 0, ph))
        for spec, loc in zip(self.spectral, self.local):
            x = F.gelu(spec(x) + loc(x))                            # global + local mixing
        x = x[..., :H, :W].permute(0, 2, 3, 1)                      # crop, channels last
        return self.fc_out(x)


# --------------------------------------------------------------------------- #
# U-Net baseline
# --------------------------------------------------------------------------- #

def _groups(c: int) -> int:
    """Largest group count (<= 8) that divides c, for GroupNorm."""
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


class ResBlock2d(nn.Module):
    """Two 3x3 convs with GroupNorm + GELU, plus a residual skip."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(_groups(cout), cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(cout), cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class UNet2d(nn.Module):
    """U-Net with GroupNorm residual blocks: a strong convolutional baseline.

    Its kernels are tied to the pixel grid, so on a finer grid each one covers a
    smaller patch of sky; the resolution-transfer test measures what that costs.
    Needs the grid divisible by 2**n_levels.
    Input (batch, H, W, in_channels) -> output (batch, H, W, out_channels).
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 1, width: int = 16,
                 n_levels: int = 3):
        super().__init__()
        chs = [width * (2 ** i) for i in range(n_levels)]
        self.enc = nn.ModuleList()
        cin = in_channels
        for c in chs:
            self.enc.append(ResBlock2d(cin, c))
            cin = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ResBlock2d(chs[-1], width * (2 ** n_levels))
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        cin = width * (2 ** n_levels)
        for c in reversed(chs):
            self.up.append(nn.ConvTranspose2d(cin, c, 2, stride=2))
            self.dec.append(ResBlock2d(2 * c, c))
            cin = c
        self.head = nn.Conv2d(chs[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.permute(0, 3, 1, 2)
        skips = []
        for enc in self.enc:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            h = dec(torch.cat([up(h), skip], dim=1))
        return self.head(h).permute(0, 2, 3, 1)


def count_params(model: nn.Module) -> int:
    """Number of real-valued parameters (a complex weight counts as two)."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
