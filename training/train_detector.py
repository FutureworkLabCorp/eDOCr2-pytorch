"""Fine-tune the CRAFT text detector on synthetic drawings, in PyTorch.

This is eDOCr2's ``train_synth_detector`` without TensorFlow. Balanced random strings over the
alphabet are drawn at 25 to 50 px, turned by up to 90 degrees either way, and pasted onto 640 x 640
crops of eDOCr2's drawing backgrounds where they miss the drawing's lines
(`training.synth.generate_drawing_imgs`). The targets are keras-ocr's character-region and link maps
at half resolution; the loss is their mean squared error and the optimizer Keras' default Adam, as
keras-ocr compiled its detector. Training starts from CRAFT's general weights, as eDOCr2's did, stops
when the validation loss has not improved for five epochs and keeps the best weights.

The result is a state dict ``OUT.pth`` that `edocr2.Detector` reads like ``craft_mlt_25k.pth``, and
the losses per epoch in ``OUT.csv``.

    python -m training.train_detector --fonts training/dimension_fonts --samples 100 --batch-size 8 \\
        --out models/detector
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from edocr2 import Detector
from edocr2.ocr_utils import compute_input
from training import synth
from training.train_recognizer import ALPHABETS

SIZE = 640


def drawing_samples(alphabet: str, fonts: list[str], samples: int, string_length=(1, 10)):
    """(640 x 640 image, lines) pairs, generated as eDOCr2's ``train_synth_detector`` did."""
    image_gen_params = {
        "height": SIZE,
        "width": SIZE,
        "text_generator": synth.get_balanced_text_generator(alphabet, string_length),
        "font_groups": {alphabet: fonts},
        "font_size": (25, 50),
        "margin": 0,
        "rotationZ": (-90, 90),
    }
    return synth.generate_drawing_imgs(image_gen_params, synth.get_backgrounds(SIZE, SIZE, samples))


def batch(samples, size: int, heatmap: np.ndarray):
    """Normalized images (B, 3, 640, 640) and target maps (B, 320, 320, 2), as keras-ocr's ``get_batch_generator``."""
    pairs = [next(samples) for _ in range(size)]
    images = np.array([image for image, _ in pairs])
    maps = np.array([synth.compute_maps(heatmap, SIZE, SIZE, lines) for _, lines in pairs])
    x = torch.from_numpy(compute_input(images).astype("float32")).permute(0, 3, 1, 2).contiguous()
    return x, torch.from_numpy(maps.astype("float32"))


def train(
    alphabet: str,
    fonts: list[str],
    out: str | Path,
    pretrained: str | Path | None = None,
    samples: int = 100,
    batch_size: int = 8,
    epochs: int = 1,
    string_length=(1, 10),
    val_split: float = 0.2,
    patience: int = 5,
    lr: float = 1e-3,
    device: str | None = None,
) -> Path:
    """Fine-tune CRAFT; returns the path of the best state dict."""
    detector = Detector(pretrained, device)
    model, device = detector.model.train(), detector.device
    # keras-ocr gives its VGG blocks PyTorch's batch-norm momentum 0.1, which Keras reads the other way
    # round: torch's 0.9. Its upsampling blocks' Keras 0.9 is torch's default already.
    for name, module in model.named_modules():
        if name.startswith("basenet.") and isinstance(module, torch.nn.BatchNorm2d):
            module.momentum = 0.9
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)  # Keras' defaults but lr
    heatmap = synth.get_gaussian_heatmap(size=512, distanceRatio=1.5)  # keras-ocr's batch generator's
    train_samples = drawing_samples(alphabet, fonts, samples, string_length)
    val_samples = drawing_samples(alphabet, fonts, samples, string_length)
    train_steps = math.ceil((1 - val_split) * samples / batch_size)
    val_steps = math.ceil(val_split * samples / batch_size)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best, since_best = math.inf, 0
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "loss", "val_loss", "seconds"])
        for epoch in range(epochs):
            start, losses = time.time(), []
            for _ in range(train_steps):
                x, y = batch(train_samples, batch_size, heatmap)
                loss = torch.nn.functional.mse_loss(model(x.to(device))[0], y.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            model.eval()
            with torch.inference_mode():
                val_loss = float(
                    np.mean(
                        [
                            torch.nn.functional.mse_loss(model(x.to(device))[0], y.to(device)).item()
                            for x, y in (batch(val_samples, batch_size, heatmap) for _ in range(val_steps))
                        ]
                    )
                )
            model.train()
            seconds = time.time() - start
            writer.writerow([epoch, np.mean(losses), val_loss, round(seconds, 1)])
            log.flush()
            print(f"epoch {epoch}: loss {np.mean(losses):.5f}, val_loss {val_loss:.5f}")
            if val_loss < best:
                best, since_best = val_loss, 0
                torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out.with_suffix(".pth"))
            else:
                since_best += 1
                if since_best >= patience:
                    break
    return out.with_suffix(".pth")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alphabet", default="dimensions", help="dimensions, gdts, or the characters themselves")
    parser.add_argument("--fonts", required=True, help="directory of .ttf fonts that draw every character")
    parser.add_argument("--out", required=True, help="output path without suffix")
    parser.add_argument("--pretrained", help="a CRAFT state dict; by default craft_mlt_25k.pth")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate; Keras' default is 1e-3")
    parser.add_argument("--device", help="torch device; by default as edocr2 picks it")
    args = parser.parse_args(argv)
    fonts = sorted(str(p) for p in Path(args.fonts).glob("*.[tT][tT][fF]"))
    if not fonts:
        raise SystemExit(f"no .ttf fonts in {args.fonts}")
    path = train(
        ALPHABETS.get(args.alphabet, args.alphabet),
        fonts,
        args.out,
        pretrained=args.pretrained,
        samples=args.samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )
    print(f"best weights: {path}")


if __name__ == "__main__":
    main()
