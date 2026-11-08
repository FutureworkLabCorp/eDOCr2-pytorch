"""Train eDOCr2's text recognizer on synthetic data, in PyTorch.

This is eDOCr2's ``train_synth_recognizer`` without TensorFlow. Balanced random strings over the
alphabet are drawn with the given fonts at 20 to 40 px on 256 x 256 canvases, warped to 31 x 200,
and samples with too little ink are dropped (`training.synth`). The loss is keras-ocr's CTC over
the 48 output steps with the blank last, averaged over the batch; the optimizer is Keras' default
RMSprop. Training stops when the validation loss has not improved for five epochs and keeps the
best weights. Given an eDOCr2 ``.keras`` recognizer, it fine-tunes that one. Otherwise it starts, as
eDOCr2's did, from keras-ocr's pretrained recognizer without its output layer (``kurapan``, which
keras-ocr loads for any alphabet but its own), unless told to start from scratch.

The result is a state dict ``OUT.pt`` with its alphabet in ``OUT.txt``, which `edocr2.Recognizer`
reads like eDOCr2's ``.keras`` files, and the losses per epoch in ``OUT.csv``.

    python -m training.train_recognizer --alphabet dimensions --fonts training/dimension_fonts \\
        --samples 25000 --epochs 3 --batch-size 256 --out models/recognizer_dimensions
"""

from __future__ import annotations

import argparse
import csv
import math
import string
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from edocr2.device import resolve_device
from edocr2.models import CRNN, load_crnn
from edocr2.models.crnn import copy_keras_weights
from edocr2.recognizer import _collapse, read_alphabet
from edocr2.weights import Weight, fetch
from training import metrics, synth

# keras-ocr's recognizer trained on synthetic text, without its output layer (Keras 2 HDF5).
KURAPAN_NOTOP = Weight(
    "crnn_kurapan_notop.h5",
    "https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/crnn_kurapan_notop.h5",
    "027fd2cced3cbea0c4f5894bb8e9e85bac04f11daf96b8fdcf1e4ee95dcf51b9",
)
# Where each module's weights sit in it: keras-ocr's layer names, the transformer's under "model".
_KURAPAN_LAYERS = {
    **{f"conv_{i + 1}": f"convs.{i}" for i in range(7)},
    "bn_3": "bns.2",
    "bn_5": "bns.4",
    "bn_7": "bns.6",
    "model/conv2d": "loc1",
    "model/conv2d_1": "loc2",
    "model/dense": "loc3",
    "model/dense_1": "loc4",
    "fc_9": "fc9",
    "lstm_10": "lstms.0",
    "lstm_10_back": "lstms.1",
    "lstm_11": "lstms.2",
    "lstm_11_back": "lstms.3",
}
_KERAS_ORDER = ("kernel", "recurrent_kernel", "bias", "gamma", "beta", "moving_mean", "moving_variance")

# The alphabets eDOCr2 trained its two recognizers on (eDOCr2's test_train.py).
_GDT_SYMBOLS = "⏤⏥○⌭⌒⌓⏊∠⫽⌯⌖◎↗⌰"
_FCF_SYMBOLS = "ⒺⒻⓁⓂⓅⓈⓉⓊ"
ALPHABETS = {
    "dimensions": string.digits + "AaBCDRGHhMmnxZtd" + '(),.+-±:/°"⌀=',
    "gdts": string.digits + ",.⌀ABCDZ" + _GDT_SYMBOLS + _FCF_SYMBOLS,
}


def load_kurapan_backbone(model: CRNN, path: str | Path) -> CRNN:
    """Everything but the output layer from keras-ocr's ``crnn_kurapan_notop.h5``."""
    import h5py

    tensors = {}
    with h5py.File(path, "r") as h5:
        for name, target in _KURAPAN_LAYERS.items():
            group = h5[name if "/" in name else f"{name}/{name}"]
            arrays = {key.split(":")[0]: np.asarray(value) for key, value in group.items()}
            tensors[target] = [arrays[key] for key in _KERAS_ORDER if key in arrays]
    copy_keras_weights(model, tensors)
    return model


def recognizer_samples(
    alphabet: str, fonts: list[str], string_length=(5, 10), bias_chars: str = ""
) -> Iterator[tuple[np.ndarray, str]]:
    """(31 x 200 RGB image, text) pairs, generated as eDOCr2's ``train_synth_recognizer`` did."""
    text = synth.get_balanced_text_generator(alphabet, string_length, bias_chars=bias_chars)
    images = synth.get_image_generator(
        height=256, width=256, text_generator=text, font_groups={alphabet: fonts}, font_size=(20, 40), margin=10
    )
    lines = synth.convert_image_generator_to_recognizer_input(
        image_generator=images,
        max_string_length=min(CRNN.steps, string_length[1]),
        target_width=CRNN.width,
        target_height=CRNN.height,
        margin=1,
    )
    return synth.filter_wrong_samples(lines)


def batch(samples: Iterator[tuple[np.ndarray, str]], size: int, alphabet: str):
    """Images (B, 31, 200, 1) in [0, 1] and CTC targets, as keras-ocr's ``get_batch_generator`` built them."""
    pairs = [next(samples) for _ in range(size)]
    images = np.stack([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[..., np.newaxis] for image, _ in pairs])
    texts = [text.strip() for _, text in pairs]
    if not all(texts) or any(c not in alphabet for c in "".join(texts)):
        raise ValueError(f"a sample is empty or outside the alphabet: {texts}")
    targets = torch.tensor([alphabet.index(c) for c in "".join(texts)])
    lengths = torch.tensor([len(text) for text in texts])
    return torch.from_numpy(images.astype("float32") / 255), targets, lengths, texts


def ctc_loss(model: CRNN, images: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """keras-ocr's ``ctc_batch_cost`` averaged over the batch: blank last, every output step used."""
    log_probs = model.logits(images).log_softmax(-1).permute(1, 0, 2)
    # CTC runs on the CPU: MPS has no CTC kernel, and the cost is small next to the network's.
    log_probs = log_probs.float().cpu()
    steps = torch.full((log_probs.shape[1],), log_probs.shape[0], dtype=torch.long)
    loss = nn.functional.ctc_loss(
        log_probs, targets, steps, lengths, blank=model.n_classes - 1, reduction="none", zero_infinity=True
    )
    return loss.mean()


def evaluate(model: CRNN, samples, steps: int, batch_size: int, alphabet: str, device: str) -> tuple[float, float]:
    """Mean CTC loss and character error rate over ``steps`` batches."""
    model.eval()
    losses, predictions, truths = [], [], []
    with torch.inference_mode():
        for _ in range(steps):
            images, targets, lengths, texts = batch(samples, batch_size, alphabet)
            images = images.to(device)
            losses.append(ctc_loss(model, images, targets, lengths).item())
            best = model(images).argmax(-1).cpu().tolist()
            predictions += [_collapse(row, alphabet) for row in best]
            truths += texts
    model.train()
    return float(np.mean(losses)), metrics.get_cer(predictions, truths)


def train(
    alphabet: str,
    fonts: list[str],
    out: str | Path,
    pretrained: str | Path | None = None,
    samples: int = 1000,
    batch_size: int = 256,
    epochs: int = 10,
    string_length=(5, 10),
    bias_chars: str = "",
    val_split: float = 0.2,
    patience: int = 5,
    lr: float = 1e-3,
    from_scratch: bool = False,
    device: str | None = None,
) -> Path:
    """Train or fine-tune a recognizer; returns the path of the best state dict."""
    device = resolve_device(device)
    if pretrained:
        model = load_crnn(pretrained)
    else:
        model = CRNN(n_classes=len(alphabet) + 1)
        if not from_scratch:
            load_kurapan_backbone(model, fetch(KURAPAN_NOTOP))
    if model.n_classes != len(alphabet) + 1:
        raise ValueError(f"{pretrained} has {model.n_classes} classes; the alphabet needs {len(alphabet) + 1}")
    model.to(device).train()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.9, eps=1e-7)  # Keras' defaults but lr
    train_samples = recognizer_samples(alphabet, fonts, string_length, bias_chars)
    val_samples = recognizer_samples(alphabet, fonts, string_length, bias_chars)
    train_steps = math.ceil((1 - val_split) * samples / batch_size)
    val_steps = math.ceil(val_split * samples / batch_size)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".txt").write_text(alphabet + "\n", encoding="utf-8")
    best, since_best = math.inf, 0
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "loss", "val_loss", "val_cer", "seconds"])
        for epoch in range(epochs):
            start, losses = time.time(), []
            for _ in range(train_steps):
                images, targets, lengths, _ = batch(train_samples, batch_size, alphabet)
                loss = ctc_loss(model, images.to(device), targets, lengths)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            val_loss, val_cer = evaluate(model, val_samples, val_steps, batch_size, alphabet, device)
            seconds = time.time() - start
            writer.writerow([epoch, np.mean(losses), val_loss, val_cer, round(seconds, 1)])
            log.flush()
            print(f"epoch {epoch}: loss {np.mean(losses):.4f}, val_loss {val_loss:.4f}, val_cer {val_cer:.4f}")
            if val_loss < best:
                best, since_best = val_loss, 0
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out.with_suffix(".pt"))
            else:
                since_best += 1
                if since_best >= patience:
                    break
    return out.with_suffix(".pt")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alphabet", default="dimensions", help="dimensions, gdts, or the characters themselves")
    parser.add_argument("--fonts", required=True, help="directory of .ttf fonts that draw every character")
    parser.add_argument("--out", required=True, help="output path without suffix")
    parser.add_argument("--pretrained", help="an eDOCr2 .keras recognizer or a .pt from this script")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--min-length", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=10)
    parser.add_argument("--bias-chars", default="", help="characters to draw more often, as eDOCr2's bias_char")
    parser.add_argument("--lr", type=float, default=1e-3, help="RMSprop learning rate; Keras' default is 1e-3")
    parser.add_argument("--from-scratch", action="store_true", help="without --pretrained, skip keras-ocr's backbone")
    parser.add_argument("--device", help="torch device; by default as edocr2 picks it")
    args = parser.parse_args(argv)
    if args.pretrained:
        alphabet = read_alphabet(args.pretrained)
    else:
        alphabet = ALPHABETS.get(args.alphabet, args.alphabet)
    fonts = sorted(str(p) for p in Path(args.fonts).glob("*.[tT][tT][fF]"))
    if not fonts:
        raise SystemExit(f"no .ttf fonts in {args.fonts}")
    path = train(
        alphabet,
        fonts,
        args.out,
        pretrained=args.pretrained,
        samples=args.samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        string_length=(args.min_length, args.max_length),
        bias_chars=args.bias_chars,
        lr=args.lr,
        from_scratch=args.from_scratch,
        device=args.device,
    )
    print(f"best weights: {path}")


if __name__ == "__main__":
    main()
