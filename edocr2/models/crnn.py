"""eDOCr2's CRNN recognizer in PyTorch, read from eDOCr2's ``.keras`` files without TensorFlow.

The network is keras-ocr's recognizer (github.com/faustomorales/keras-ocr, MIT License): seven
convolutions, a spatial transformer, two bidirectional LSTMs and a CTC output, at 31 x 200 in
grayscale. Two of keras-ocr's quirks are kept on purpose, because eDOCr2 trained its weights
with them and changing either changes the answers: the transformer samples with keras-ocr's own
bilinear sampler (`_stn_sample`), not ``grid_sample``; and each backward LSTM's output is added to
the forward one in reversed time order, as Keras ``go_backwards`` leaves it.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _stn_sample(img: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """keras-ocr's ``_transform``: ``img`` (B, H, W, C) sampled through affine ``theta`` (B, 6)."""
    b, h, w, c = img.shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, h, device=img.device), torch.linspace(-1, 1, w, device=img.device), indexing="ij"
    )
    grid = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.ones(h * w, device=img.device)])
    t = theta.view(b, 2, 3) @ grid
    # Scaled by the width and height, not width - 1, and clipped after flooring, as keras-ocr does.
    x, y = 0.5 * (t[:, 0] + 1.0) * w, 0.5 * (t[:, 1] + 1.0) * h
    x0, y0 = torch.floor(x).long(), torch.floor(y).long()
    x1, y1 = (x0 + 1).clamp(0, w - 1), (y0 + 1).clamp(0, h - 1)
    x0, y0 = x0.clamp(0, w - 1), y0.clamp(0, h - 1)
    flat = img.reshape(b, h * w, c)

    def at(yy: torch.Tensor, xx: torch.Tensor) -> torch.Tensor:
        return torch.gather(flat, 1, (yy * w + xx).unsqueeze(-1).expand(-1, -1, c))

    x0f, x1f, y0f, y1f = x0.float(), x1.float(), y0.float(), y1.float()
    out = (
        ((x1f - x) * (y1f - y)).unsqueeze(-1) * at(y0, x0)
        + ((x1f - x) * (y - y0f)).unsqueeze(-1) * at(y1, x0)
        + ((x - x0f) * (y1f - y)).unsqueeze(-1) * at(y0, x1)
        + ((x - x0f) * (y - y0f)).unsqueeze(-1) * at(y1, x1)
    )
    return out.view(b, h, w, c)


class CRNN(nn.Module):
    """Input (B, 31, 200, 1) in [0, 1], as keras-ocr's; output (B, 48, n_classes) softmax, blank last."""

    height, width = 31, 200

    def __init__(self, n_classes: int, filters=(64, 128, 256, 256, 512, 512, 512), units: int = 128):
        super().__init__()
        self.n_classes = n_classes
        chans = (1, *filters)
        self.convs = nn.ModuleList(nn.Conv2d(chans[i], chans[i + 1], 3, padding=1) for i in range(7))
        self.bns = nn.ModuleDict({str(i): nn.BatchNorm2d(filters[i], eps=1e-3) for i in (2, 4, 6)})
        self.loc1, self.loc2 = nn.Conv2d(512, 16, 5, padding=2), nn.Conv2d(16, 32, 5, padding=2)
        self.loc3, self.loc4 = nn.Linear(50 * 7 * 32, 64), nn.Linear(64, 6)
        self.fc9 = nn.Linear(7 * 512, units)
        self.lstms = nn.ModuleList(nn.LSTM(units, units, batch_first=True) for _ in range(4))
        self.fc12 = nn.Linear(2 * units, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3).flip(2).permute(0, 3, 1, 2)  # Keras Permute((2, 1, 3)), x[:, :, ::-1], to NCHW
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x))
            if str(i) in self.bns:
                x = self.bns[str(i)](x)
                x = F.max_pool2d(x, 2) if i < 6 else x
        loc = F.relu(self.loc2(F.relu(self.loc1(x)))).permute(0, 2, 3, 1).flatten(1)
        theta = self.loc4(F.relu(self.loc3(loc)))
        x = F.relu(self.fc9(_stn_sample(x.permute(0, 2, 3, 1), theta).flatten(2)))
        x = self.lstms[0](x)[0] + self.lstms[1](x.flip(1))[0]  # go_backwards: the output stays reversed
        x = torch.cat([self.lstms[2](x)[0], self.lstms[3](x.flip(1))[0]], -1)
        return F.softmax(self.fc12(x), -1)[:, 2:]  # keras-ocr discards the first two steps


# Where each module's weights sit in a Keras 3 ``model.weights.h5``, which names layers by class.
_KERAS_LAYERS = {
    **{f"conv2d{'' if i == 0 else f'_{i}'}": f"convs.{i}" for i in range(7)},
    "batch_normalization": "bns.2",
    "batch_normalization_1": "bns.4",
    "batch_normalization_2": "bns.6",
    "functional/layers/conv2d": "loc1",
    "functional/layers/conv2d_1": "loc2",
    "functional/layers/dense": "loc3",
    "functional/layers/dense_1": "loc4",
    "dense": "fc9",
    "dense_1": "fc12",
    "lstm": "lstms.0",
    "lstm_1": "lstms.1",
    "lstm_2": "lstms.2",
    "lstm_3": "lstms.3",
}


def load_crnn(path: str | Path) -> CRNN:
    """The recognizer in an eDOCr2 ``.keras`` file, read with h5py."""
    import h5py

    with zipfile.ZipFile(path) as archive:
        weights = archive.read("model.weights.h5")
    with h5py.File(io.BytesIO(weights), "r") as h5:
        layers = h5["layers"]

        def arrays(name: str) -> list[np.ndarray]:
            group = layers[name]
            values = group["cell"]["vars"] if "cell" in group else group["vars"]
            return [np.asarray(values[str(i)]) for i in range(len(values))]

        tensors = {name: arrays(name) for name in _KERAS_LAYERS}
    model = CRNN(n_classes=tensors["dense_1"][1].shape[0])
    with torch.no_grad():
        for name, target in _KERAS_LAYERS.items():
            module = model.get_submodule(target)
            values = [torch.from_numpy(np.ascontiguousarray(v)) for v in tensors[name]]
            if isinstance(module, nn.LSTM):  # kernel, recurrent kernel, bias; gate order i, f, c, o in both
                module.weight_ih_l0.copy_(values[0].T)
                module.weight_hh_l0.copy_(values[1].T)
                module.bias_ih_l0.copy_(values[2])
                module.bias_hh_l0.zero_()
            elif isinstance(module, nn.BatchNorm2d):  # gamma, beta, moving mean, moving variance
                for param, value in zip(
                    (module.weight, module.bias, module.running_mean, module.running_var), values, strict=True
                ):
                    param.copy_(value)
            elif isinstance(module, nn.Conv2d):  # Keras (kh, kw, in, out)
                module.weight.copy_(values[0].permute(3, 2, 0, 1))
                module.bias.copy_(values[1])
            else:  # Linear: Keras (in, out)
                module.weight.copy_(values[0].T)
                module.bias.copy_(values[1])
    return model.eval()
