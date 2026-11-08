"""Training on PyTorch: eDOCr2's synthetic data, CTC steps that learn, and weights the package reads back.

The detector, the ``.keras`` round trip and keras-ocr's backbone need their weights, fetched once into
the cache or read from EDOCR2_WEIGHTS_DIR, as the regression tests do. Everything runs on the CPU.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
import torch

from edocr2 import Detector, Recognizer
from edocr2.models import CRNN, load_crnn
from edocr2.weights import DIMENSIONS, DIMENSIONS_ALPHABET, fetch
from training import synth
from training import train_detector as td
from training import train_recognizer as tr

FONTS = sorted(str(p) for p in (Path(__file__).parents[1] / "training" / "dimension_fonts").glob("*.ttf"))
ALPHABET = tr.ALPHABETS["dimensions"]


def _seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def test_recognizer_samples_are_lines_of_the_alphabet():
    _seed()
    samples = tr.recognizer_samples(ALPHABET, FONTS)
    for _ in range(16):
        image, text = next(samples)
        assert image.shape == (31, 200, 3) and image.dtype == np.uint8
        assert 1 <= len(text) <= 10 and set(text) <= set(ALPHABET)


def test_ctc_steps_lower_the_loss():
    _seed()
    model = CRNN(len(ALPHABET) + 1).train()
    images, targets, lengths, _ = tr.batch(tr.recognizer_samples(ALPHABET, FONTS), 8, ALPHABET)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.9, eps=1e-7)
    losses = []
    for _ in range(30):
        loss = tr.ctc_loss(model, images, targets, lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] / 2, losses


def test_kurapan_backbone_fills_everything_but_the_output_layer():
    import h5py

    model = CRNN(len(ALPHABET) + 1)
    head = model.fc12.weight.clone()
    path = fetch(tr.KURAPAN_NOTOP)
    tr.load_kurapan_backbone(model, path)
    state = model.state_dict()
    with h5py.File(path, "r") as h5:
        assert np.array_equal(state["convs.0.weight"].permute(2, 3, 1, 0).numpy(), h5["conv_1/conv_1/kernel:0"][()])
        assert np.array_equal(state["loc4.bias"].numpy(), h5["model/dense_1/bias:0"][()])
        assert np.array_equal(
            state["lstms.3.weight_hh_l0"].T.numpy(), h5["lstm_11_back/lstm_11_back/recurrent_kernel:0"][()]
        )
        assert np.array_equal(state["bns.6.running_var"].numpy(), h5["bn_7/bn_7/moving_variance:0"][()])
    assert torch.equal(state["fc12.weight"], head)


def test_trained_recognizer_loads_into_the_package(tmp_path):
    _seed()
    path = tr.train(
        ALPHABET, FONTS, tmp_path / "rec", samples=16, batch_size=8, epochs=1, from_scratch=True, device="cpu"
    )
    assert path.with_suffix(".txt").read_text(encoding="utf-8").strip() == ALPHABET
    assert len(path.with_suffix(".csv").read_text(encoding="utf-8").splitlines()) == 2
    x = torch.rand(1, 31, 200, 1)
    with torch.inference_mode():
        assert torch.equal(Recognizer(path, device="cpu").model(x), load_crnn(path)(x))


def test_keras_recognizer_survives_a_state_dict(tmp_path):
    keras = fetch(DIMENSIONS)
    pt = tmp_path / "recognizer.pt"
    torch.save(load_crnn(keras).state_dict(), pt)
    shutil.copy(fetch(DIMENSIONS_ALPHABET), pt.with_suffix(".txt"))
    _seed()
    images = [image for image, _ in (next(tr.recognizer_samples(ALPHABET, FONTS)) for _ in range(4))]
    original, copied = Recognizer(keras, device="cpu"), Recognizer(pt, device="cpu")
    assert [original.recognize(image) for image in images] == [copied.recognize(image) for image in images]


def test_detector_targets_match_the_output():
    _seed()
    image, lines = next(td.drawing_samples(ALPHABET, FONTS, 2))
    maps = synth.compute_maps(synth.get_gaussian_heatmap(512, 1.5), td.SIZE, td.SIZE, lines)
    assert image.shape == (td.SIZE, td.SIZE, 3) and lines
    assert maps.shape == (td.SIZE // 2, td.SIZE // 2, 2) and 0 < maps.max() <= 1


def test_trained_detector_loads_into_the_package(tmp_path):
    _seed()
    path = td.train(ALPHABET, FONTS, tmp_path / "det", samples=2, batch_size=1, epochs=1, device="cpu")
    assert len(path.with_suffix(".csv").read_text(encoding="utf-8").splitlines()) == 2
    x = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        trained = Detector(path, device="cpu").model(x)[0]
    assert trained.shape == (1, 32, 32, 2)
