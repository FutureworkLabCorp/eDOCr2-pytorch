# eDOCr2-pytorch

Read the dimensions, GD&T frames and title blocks of mechanical engineering drawings, in PyTorch.

eDOCr2-pytorch is [eDOCr2](https://github.com/javvi51/edocr2) by Javier Villena Toro (Design
Automation Lab, Linköping University), the successor of [eDOCr](https://github.com/javvi51/eDOCr),
ported from TensorFlow to PyTorch. eDOCr2 set out to give production quality control an open,
end-to-end way to digitize engineering drawings. This port keeps its pipeline, its trained
recognizers and its output, and changes only what running without TensorFlow requires. It is not an
official release of eDOCr2.

> **Status: 0.1.0.dev0.** On eDOCr2's seven sample drawings this port gives eDOCr2's output exactly,
> every text and every box, on the CPU and on Apple MPS. CUDA has not been tried yet.

## What changed from eDOCr2

- **No TensorFlow, Keras or imgaug.** The CRAFT detector runs from
  [craft-detector](https://github.com/wkpark/CRAFT-pytorch) (NAVER's CRAFT) and the CRNN
  recognizer is ported to PyTorch. NumPy 2 and Python 3.10 to 3.13 work.
- **CPU, CUDA or MPS**, picked automatically or set with `EDOCR2_DEVICE`. On TensorFlow's Metal
  plugin eDOCr2 read almost every dimension wrong; on PyTorch MPS the answers match the CPU's.
- **Weights are read from the releases that published them**, checked against their sha256 and
  never converted: eDOCr2's `.keras` recognizers are read with h5py, CRAFT's `.pth` as it is. Set
  `EDOCR2_WEIGHTS_DIR` to read them from a directory instead and download nothing.
- **Reading many drawings in one process works.** On Keras, a predict graph traced on one
  drawing's patches could fail on the next drawing's.
- The symbol templates load from the installed package, not from the working directory.
- Removed: recognizer and detector training and synthetic data generation, which were TensorFlow
  code (training may come back in PyTorch), and the VLM helpers (`llm_tools.py`: Qwen2-VL through
  transformers, GPT-4o through the OpenAI API), which sit outside the OCR pipeline.

## Install

```bash
pip install edocr2-pytorch   # once released; until then: pip install -e .
```

Tesseract is needed too, with the `eng` and `nor` languages (the dimension filter asks for both):

```bash
sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-nor   # Debian, Ubuntu
```

Elsewhere, put `eng.traineddata` and `nor.traineddata` from
[tessdata](https://github.com/tesseract-ocr/tessdata) in a directory and point `TESSDATA_PREFIX` at it.

## Use

```python
import cv2
import edocr2

models = edocr2.load_models()                    # fetches the weights once (about 140 MB)
result = edocr2.read_drawing(cv2.imread("drawing.jpg"), models)

for text, box in result.dimensions:              # box: 4 corners inside the drawing's frame
    print(text, box.astype(int).tolist())
print(result.gdt)                                # [(text, (x, y)), ...], cells joined by "|"
print(result.tables)                             # title block words from Tesseract
```

`read_drawing` takes eDOCr2's settings as keyword arguments (`frame_thres`, `gdt_thres`,
`cluster_thres`, `max_img_size`, `language`, ...), with eDOCr2's `test_drawing.py` defaults.
Dimension and other-text boxes are in the pixels of the region inside `result.frame`, as in eDOCr2;
add `frame.x` and `frame.y` to place them on the page.

## Weights

| File | Network | Published by |
|---|---|---|
| `craft_mlt_25k.pth` | CRAFT text detector (SynthText, IC13, IC17) | NAVER Clova AI; fetched from craft-detector v0.0.2 |
| `recognizer_dimensions_2.keras`, `.txt` | dimension recognizer and its alphabet | eDOCr2 v1.0.0 |
| `recognizer_gdts.keras`, `.txt` | GD&T recognizer and its alphabet | eDOCr2 v1.0.0 |

They are cached in `~/.cache/edocr2` (`EDOCR2_CACHE_DIR`). For an offline install, download the
five files into a directory and set `EDOCR2_WEIGHTS_DIR`.

## Test

```bash
pip install -e ".[dev]"
pytest
```

`tests/test_regression.py` reads the seven samples and compares the outputs with eDOCr2's
(`tests/golden/`). It needs the weights and Tesseract with `eng` and `nor` from
[tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast) 4.1.0. GD&T frames, dimensions
and text boxes must match on any machine; the text Tesseract reads is compared only under
Tesseract 5.5.4, the build the expected outputs were read with.

## Citing

eDOCr2-pytorch is eDOCr2's work carried over to PyTorch. Please cite the papers behind it:

- J. Villena Toro, A. Wiberg, M. Tarkian, *Optical character recognition on engineering drawings
  to achieve automation in production quality control*, Frontiers in Manufacturing Technology 3
  (2023). [doi:10.3389/fmtec.2023.1154132](https://doi.org/10.3389/fmtec.2023.1154132)
- The eDOCr2 paper, preprint: [doi:10.2139/ssrn.5045921](http://dx.doi.org/10.2139/ssrn.5045921)

## License

MIT. See [LICENSE](LICENSE), and [NOTICE](NOTICE) for eDOCr2, keras-ocr and CRAFT.
