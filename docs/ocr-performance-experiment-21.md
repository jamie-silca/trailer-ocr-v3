# OCR Performance Experiment 21 — Tiny Custom Single-Character Classifier

**Status:** PLANNED. Depends on EXP-17 (needs per-character box detection).

## 1. Goal

Sidestep sequence modelling entirely. The trailer-ID alphabet is tiny — at most
**16 classes**: `J B H Z U R 0 1 2 3 4 5 6 7 8 9`. Train a small CNN single-char
classifier on this alphabet and apply it per-box to the char detections from
EXP-17. Concatenate classifications in Y-order to form the final string.

Target: on top of EXP-17, **match or beat** the PP-OCR recogniser on portrait
at a fraction of the latency. Portrait ≥ 50/156 is the success line. Overall
**+3–4pp** over EXP-17.

## 2. Hypothesis

Modern CRNNs (PP-OCR) and VLMs (TrOCR) are generalists. On a closed 16-class
alphabet with clean printed glyphs, a purpose-built classifier with ~100k
parameters can hit **> 99% per-char accuracy** after training on a few thousand
synthetic + real samples. With N=10 chars per ID, 99% per-char = ~90% per-ID
— enough to dominate this bucket.

The existing reluctance to train is that we don't have a labelled char-level
dataset. But **we do** — the ground-truth ID strings tell us each char; we
just need to pair each with a char-level crop. EXP-17's per-character detection
provides that pairing automatically for portrait crops that already happen to
OCR correctly, and we can synthesise the rest by rendering GT strings in a
matching font.

## 3. Methodology

### Dataset construction

1. **Real positives:** for every correctly-OCR'd portrait crop under EXP-16/17,
   pair EXP-17's N char boxes with the N characters of the ground-truth string.
   ~40 crops × 6–10 chars = 300–400 labelled samples to start.
2. **Synthetic:** render each of the ~40 known GT IDs in stacked-vertical layout
   using candidate fonts (Arial Bold, DIN, Impact, commercial trailer-label
   fonts if available). Vary contrast, blur, JPEG quality. 2000+ samples.
3. **Augmentation:** random perspective, Gaussian blur, motion blur, brightness.
4. **Held-out test:** reserve 20% of real positives (not used in training) as
   the per-char test set.

### Model

- Input: 32×32 grayscale char crop (upscaled/letterboxed from box detection).
- Architecture: small CNN — 3 conv blocks (16→32→64 channels, 3×3, ReLU,
  maxpool) → GAP → 16-class softmax. ~30k params.
- Loss: cross-entropy. Optimiser: AdamW lr=1e-3, cosine schedule. 30 epochs.
- Deployment: export to ONNX. CPU inference < 1 ms per char.

### Pipeline

1. EXP-17 produces ordered per-char boxes for the portrait crop.
2. Crop each box, normalise to 32×32.
3. Classify each with the custom model. Record confidence.
4. Concatenate Y-ordered predictions. Run EXP-15's `format_rescore` on the
   string.
5. Format-gate + fallback to EXP-17 output on failure.

## 4. Critical Files

- `tools/train_charnet.py` (new) — dataset builder, trainer, ONNX exporter.
- `tests/charnet_portrait.py` (new) — `classify_chars(boxes, image, onnx_session) -> str`.
- [tests/benchmark_ocr.py](../tests/benchmark_ocr.py) — add `--portrait-strategy charnet`.
- `models/charnet.onnx` (new, ~200 kB checked in or fetched).
- `tools/render_trailer_ids.py` (new) — synthetic dataset generator.

## 5. Implementation Steps

1. Write `tools/render_trailer_ids.py`: renders a stacked-vertical
   trailer-ID image from a font + string, with augmentations. Validate
   output visually against real crops.
2. Build the real-positive dataset by running EXP-17 on the current
   benchmark and filtering to correctly-OCR'd portraits.
3. Merge synthetic + real; train `charnet` with `tools/train_charnet.py`.
   Hold out 20% of real for test.
4. Confirm per-char test accuracy ≥ 99%. If not, revisit font choice and
   augmentation.
5. Export to ONNX. Implement `tests/charnet_portrait.py` inference path.
6. Wire `--portrait-strategy charnet` into the benchmark.
7. Run full 672 benchmark.
8. Post-mortem: portrait delta vs EXP-17, per-char confusion matrix on the
   held-out test, latency (expected to beat every other variant).

## 6. Verification

- **Primary:** portrait bucket ≥ 50/156.
- **Guardrail 1:** per-char test accuracy ≥ 99% before running the full benchmark.
- **Guardrail 2:** non-portrait buckets unchanged.
- **Guardrail 3:** latency < 5 ms per portrait crop (detection + N×1 ms classification).

## 7. Risks

- **Font mismatch:** if synthetic fonts don't match the real trailer labels,
  the model overfits to synthetic. Mitigation: real-positive set must be the
  majority of training data, not a minority. If we can't bootstrap enough
  real positives, this experiment stalls until EXP-16/17 expand that set.
- **Distribution shift:** real crops have motion blur, glare, scratches that
  are hard to simulate. Augmentation helps but won't close the gap.
- **Alphabet drift:** if production ever encounters a character outside
  `{JBHZUR0-9}` (e.g. a new operator prefix), the model silently mis-classifies.
  Mitigation: reject when max-softmax confidence < 0.7 and fall back to EXP-17.
- **Operational overhead:** now we own a trained model, weights file, training
  pipeline, and a retraining cadence. Justified only if EXP-16–20 can't hit
  the portrait target.

## 8. Out of Scope

- Multi-char sequence models (PP-OCR and TrOCR already cover that).
- End-to-end detection+recognition training.
- Fine-tuning an existing model — simpler to train from scratch at this size.
- Using this for non-portrait buckets.
