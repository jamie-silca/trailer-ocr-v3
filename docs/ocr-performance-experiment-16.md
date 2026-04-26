# OCR Performance Experiment 16 — Unroll-to-Horizontal (Virtual Horizontalisation)

**Status:** **SKIPPED — duplicates EXP-13A.**

## 1. Why this experiment was not run

When EXP-16 was drafted, the intent was to feed PP-OCRv4's recognition head its
native input shape by slicing a stacked-vertical column into per-character bands
and composing them into a synthetic horizontal strip. On review of the existing
codebase before implementation, this is **exactly what EXP-13A ("stitch")
already does**.

Implementation reference: [`_stitch_letters`](../tests/preprocessing.py#L349) in
`tests/preprocessing.py` resizes each uniform-slice band to 48 px height and
pastes them side-by-side with a 4 px neutral-gray gap, then feeds the composite
strip to `OcrProcessor.process_image()`. This is the EXP-16 pipeline verbatim.

### EXP-13A result (already known)

| Metric | EXP-09 baseline | EXP-13A stitch |
|---|---|---|
| Portrait correct | 2 / 156 (1.3%) | **2 / 156 (1.3%)** |
| Overall correct | 257 / 672 | 257 / 672 |
| Precision | 57.6% | 60.9% (+3.3 pp, suppression only) |
| Portrait median latency | ~110 ms | **660 ms** (+550 ms) |

The composite strip helps precision (the decoder returns `None` on 13 portrait
crops the baseline would have mis-read) but produces **zero new correct
portrait answers**.

### Why EXP-13A didn't gain, per its Finding 1

EXP-13 instrumented the rec model directly on the stitched strips. Result on
ground-truth `JBHZ672061`:

| Input to rec model | Output | Confidence |
|---|---|---|
| Stitched strip (9 slices, 48 px) | `'3RH'` | 0.20 |
| Stitched strip (10 slices, no gap) | `'RSAHR'` | 0.21 |
| Perspective-rectified + 10-slice stitch | `'P1L575AA'` | 0.18 |
| Column rotated 90° CW | `'PONNNIWL'` | 0.76 (confidently wrong) |

**The PP-OCRv4 rec model itself is the bottleneck, not the layout.** Its CRNN
has learned character priors too entangled with horizontal word context to
decode the trailer-yard stencil font when presented one character at a time,
even inside a well-formed horizontal strip.

## 2. Marginal deltas EXP-16 could still have brought

Things EXP-13A did not explore:

1. **Per-slice tight-bbox trim** — EXP-13A sliced at the full column width;
   trimming each band to its own glyph bbox might reduce whitespace padding
   that confuses CTC decoding.
2. **Gap / height / padding sweep** — EXP-13A used a single combo (48 px,
   4 px gap, 3 px side pad). Other points may score higher.
3. **Format-gate + fallback on output** — safer (precision) but does not
   lift correct count.

Expected gain from these marginals is small. EXP-13's Finding 1 suggests the
rec model's character-level accuracy on this font is single-digit regardless
of strip quality, which caps any PP-OCR-based stitch approach at or near
the 2/156 floor.

## 3. Decision

**Skip EXP-16. The real bottleneck is the recogniser, not the stitcher.**

Move to **EXP-20 (TrOCR on the stitched strip)** — reuse EXP-13A's
`decode_stacked_vertical(variant="stitch")` infrastructure verbatim and feed
the resulting strip to a different recognition model. This attacks the
diagnosed blocker directly.

If TrOCR also fails on the stitched strip, the stitch infrastructure itself
gets retired in favour of EXP-18 (VLM fallback), which does not require a
pre-composed strip — it reads the raw portrait crop end-to-end.

## 4. References

- [docs/ocr-performance-experiment-13.md](./ocr-performance-experiment-13.md) — prior run and root-cause diagnosis.
- [tests/preprocessing.py](../tests/preprocessing.py) — existing stitch implementation.
- [docs/ocr-performance-experiment-20.md](./ocr-performance-experiment-20.md) — successor experiment.
