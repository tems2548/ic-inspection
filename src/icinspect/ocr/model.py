"""CRNN: CNN feature extractor -> BiLSTM -> per-frame class logits (CTC)."""

from __future__ import annotations

import torch
from torch import nn


class CRNN(nn.Module):
    """Expects grayscale input of shape (N, 1, 32, W); emits (T, N, num_classes).

    The CNN downsamples width by 4 and then trims 1 frame, so T = W // 4 - 1
    (W = 160 -> T = 39). Keep W >= 4 * (max_label_len + 1) + 4 or CTC will not
    have enough frames to fit the longest label.
    """

    def __init__(
        self,
        num_classes: int,
        img_height: int = 32,
        in_channels: int = 1,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if img_height != 32:
            raise ValueError(f"CRNN expects img_height=32, got {img_height}")
        self.num_classes = num_classes
        self.img_height = img_height

        def conv(i: int, o: int, k: int = 3, s: int = 1, p: int = 1, bn: bool = False) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(i, o, k, s, p)]
            if bn:
                layers.append(nn.BatchNorm2d(o))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.cnn = nn.Sequential(
            *conv(in_channels, 64),
            nn.MaxPool2d(2, 2),                        # 16 x W/2
            *conv(64, 128),
            nn.MaxPool2d(2, 2),                        # 8  x W/4
            *conv(128, 256, bn=True),
            *conv(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),              # 4  x W/4
            *conv(256, 512, bn=True),
            *conv(512, 512),
            nn.MaxPool2d((2, 1), (2, 1)),              # 2  x W/4
            *conv(512, 512, k=2, s=1, p=0, bn=True),   # 1  x W/4-1
        )

        # Stacked as separate single-layer LSTMs with explicit dropout between them,
        # rather than one nn.LSTM(num_layers=2, dropout=...). The maths is identical
        # (PyTorch applies inter-layer dropout exactly here), but a multi-layer cuDNN
        # LSTM with dropout > 0 allocates a cuDNN dropout state whose teardown crashes
        # the process at interpreter shutdown on Windows/CUDA (0xC0000409), turning
        # every successful run into a nonzero exit code.
        self.rnn = nn.ModuleList()
        in_size = 512
        for _ in range(rnn_layers):
            self.rnn.append(nn.LSTM(in_size, rnn_hidden, bidirectional=True, batch_first=False))
            in_size = rnn_hidden * 2

        self.rnn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.cnn(x)                       # (N, C, H', W')
        n, c, h, w = feats.shape
        if h != 1:
            raise RuntimeError(f"Expected feature height 1 after CNN, got {h}")
        seq = feats.squeeze(2).permute(2, 0, 1)    # (T, N, C)
        for i, rnn in enumerate(self.rnn):
            seq, _ = rnn(seq)
            if i < len(self.rnn) - 1:
                seq = self.rnn_dropout(seq)
        return self.fc(self.dropout(seq))          # (T, N, num_classes)

    @torch.no_grad()
    def output_length(self, img_width: int) -> int:
        """Number of CTC time steps produced for a given input width."""
        device = next(self.parameters()).device
        probe = torch.zeros(1, self.cnn[0].in_channels, self.img_height, img_width, device=device)
        return int(self.cnn(probe).shape[3])


def build_model(cfg: dict, num_classes: int) -> CRNN:
    m = cfg.get("model", {})
    return CRNN(
        num_classes=num_classes,
        img_height=cfg.get("image", {}).get("height", 32),
        in_channels=1 if cfg.get("image", {}).get("grayscale", True) else 3,
        rnn_hidden=m.get("rnn_hidden", 256),
        rnn_layers=m.get("rnn_layers", 2),
        dropout=m.get("dropout", 0.1),
    )
