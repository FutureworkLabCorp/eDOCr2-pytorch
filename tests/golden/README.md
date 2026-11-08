# Golden outputs

One JSON per sample in `../test_samples`: what eDOCr2 read from it, which eDOCr2-pytorch must reproduce.

Made on 2026-11-08 with eDOCr2 f6f9651 and its v1.0.0 recognizers, on TensorFlow 2.18.1 and
Keras 3.15.1 on the CPU (NumPy 1.26.4), with Tesseract 5.5.4 and the `eng` and `nor` models of
tessdata_fast 4.1.0, by running `test_drawing.py`'s pipeline with the
detector's and both recognizers' `predict` replaced by a PyTorch port. The port's output was
compared with Keras' own for every sample, on the CPU and on MPS, and matched in every field.

Each file holds:

- `tables`: the words Tesseract read in each title block or table, joined by spaces;
- `gdt`: each GD&T frame as `text @x,y`, cells joined by `|`;
- `dimensions` and `other`: `text @x0,y0,...,x3,y3`, the box's corners truncated to integers in
  the pixels of the region inside the drawing's frame.

Lists other than `tables` are sorted.

Tesseract reads the title blocks and the other text, so another Tesseract build reads some of
their characters differently: on Ubuntu's 5.5.0 a few title block characters change. The
regression test therefore compares that text only under Tesseract 5.5.4, and the rest (GD&T,
dimensions, the boxes of the other text) everywhere. A box's starting corner also differs
between x86 and ARM (shapely's floating point), so boxes are compared as the same corners in the
same cyclic order.
