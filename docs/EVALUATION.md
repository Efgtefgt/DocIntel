# OCR accuracy evaluation

## Metrics

* **CER (character error rate)** = Levenshtein edit distance between the reference and OCR
  characters, divided by the number of reference characters.
* **WER (word error rate)** is the same calculation over whitespace-separated words.
* Both can exceed 1.0 when the OCR output contains many extra characters.
* Lower is better. 0.0 means the text matches exactly after normalization.

## Normalization used before scoring

Normalization changes the scores, so it is fixed and reported with every benchmark run.
`EvalNormalization` defaults:

| Rule | Default | CLI flag |
| --- | --- | --- |
| Unicode NFC | on | — |
| Collapse all whitespace runs (including newlines) to one space, then trim | on | — |
| Lower-case (case-insensitive scoring) | **off** | `--lowercase` |
| Remove ASCII punctuation | **off** | `--ignore-punctuation` |

By default, case and punctuation errors count as errors, but line-break and spacing differences
don't.

## Benchmark suites

`python scripts/generate_samples.py` (re)creates all suites under `data/samples`:

| Suite | Folder | Contents | Role |
| --- | --- | --- | --- |
| basic | `data/samples` | 5 clean renders with single degradations | regression check (used during tuning) |
| hard | `data/samples/hard` | 10 conditions: shadow, low contrast, blur+JPEG, dark-mode screenshot, small noisy print, sideways scan, coloured paper, scanner border+skew, phone photo, clean control | used to design and tune the smart pipeline |
| holdout | `data/samples/holdout` | 12 random combinations of degradations with unseen text (seed 2026) | examined once; led to the orientation fixes below |
| holdout2 | `data/samples/holdout2` | as above, seed 2027 | examined once |
| **holdout3** | `data/samples/holdout3` | as above, seed 2028 | **never examined; the fairest figure** |
| documents | `data/samples/documents` | 12 typed documents with photo/scan effects, plus `<name>.type.txt` | document intelligence benchmark (see below) |

All suites are synthetic. They show whether preprocessing reacts correctly to each condition, not
how DocIntel performs on your documents (see "Adding samples").

## Running the benchmark

```powershell
python -m docintel.evaluation.benchmark data\samples\hard                     # default preset: smart
python -m docintel.evaluation.benchmark data\samples\hard --preset auto --out outputs\bench_auto
python -m docintel.evaluation.benchmark data\samples\holdout3 --engine paddle --lang en   # PaddleOCR env
```

The benchmark prints per-document CER, WER, mean confidence and duration, plus an aggregate summary.
With `--out` it also writes `report.json` and `report.csv`.

## Measured results

Measured on 2026-10-23 on the development machine (8 logical CPUs), with Tesseract 5.4.0 `eng`.
The figures are mean CER, lower is better.

| Suite | original | auto | **smart** |
| --- | --- | --- | --- |
| basic (5) | 0.150 | 0.000 | **0.000** |
| hard (10, tuning set) | 0.268 | 0.194 | **0.000** |
| holdout (12, examined) | 0.357 | 0.449 | **0.106** |
| holdout2 (12, examined) | 0.442 | 0.323 | **0.182** |
| **holdout3 (12, never examined)** | **0.486** | **0.296** | **0.189** |
| time per page (hard) | ~0.2 s | ~0.55 s | ~0.65 s |

On the never-examined set, `smart` has about 61% fewer character errors than no preprocessing and
36% fewer than `auto`. It is now the default preset. `auto` was *worse* than no preprocessing on
the first held-out set. On shadowed photos it can remove all text, because deskew filled the
corners with white; that fill bug is now fixed for every preset.

**Engine comparison** (from the PaddleOCR environment; PaddleOCR 2.10 / PaddlePaddle 3.0, `en`):

| Suite, CER / WER | Tesseract + smart | PaddleOCR (original) | PaddleOCR + smart |
| --- | --- | --- | --- |
| basic | **0.000 / 0.000** | 0.019 / 0.131 | 0.022 / 0.147 |
| hard | **0.000 / 0.000** | 0.029 / 0.243 | 0.037 / 0.331 |
| holdout3 | 0.189 / **0.230** | 0.414 / 0.704 | **0.140** / 0.578 |

* PaddleOCR + smart makes the fewest character errors on unseen degraded images.
* PaddleOCR's word error rate is much higher, because it often joins words and drops full stops.
* PaddleOCR is more robust to shadows without preprocessing.

Tesseract + smart stays the default: lowest WER, no extra install, less memory. Try PaddleOCR on
difficult photos (Image enhancement > Compare engines).

### What the measurements changed

* **Shadows and uneven lighting:** these were the main failure. The new *flatten illumination*
  step divides out the estimated background, and fixed both photo samples (CER 0.79 and 0.44 → 0.00).
* **Deskew:** it used to fill rotated corners with white. Next to a shadowed page that wedge broke
  Tesseract's thresholding and removed all text. Corners are now filled with the page's edge colour.
* **Border cropping:** it fixes scanned pages with a dark frame (0.56 → 0.00) but hurts shadowed
  photos, so `smart` crops only when a dark border is detected.
* **Orientation detection:** it failed on shadowed and small-text pages, so it now runs on a cleaned
  (flattened, inverted, enlarged) copy.
* **Confidence:** it was misleading. One shadowed photo scored 95 while missing about half its text.
  OCR output is therefore also judged by *coverage*: recognised characters compared with an
  independent count of glyph-like shapes on the page. Weak results trigger up to two alternative
  plans; the best-scoring attempt (confidence × coverage) is kept, and all attempts are reported.

### Known failure cases (holdout3)

Most remaining errors come from combinations that rarely occur together. One example is a dark-mode
screenshot that also has a shadow, a dark border and heavy JPEG compression. The other is an
upside-down page with blur, shadow and tiny text, where orientation detection is still unreliable.

## Document intelligence

```powershell
python -m docintel.evaluation.intelligence_benchmark data\samples\documents [--out outputs\intel]
```

The analysis runs on the ground-truth text and on the OCR output of each image. Values found in
the ground truth are the reference: **recall** = reference values also found after OCR (same type
and normalized value), **precision** = values found after OCR that are in the reference. This
measures how well detection survives OCR errors. Whether the detector itself is right is checked
by the hand-written unit tests (`tests/unit/test_document_intelligence.py`). When
`<name>.type.txt` exists, the document type found from OCR text is compared with it.

Measured with Tesseract 5.4 and the smart preset:

| Suite | Documents | Type correct | Value recall | Value precision |
| --- | --- | --- | --- | --- |
| documents (12 typed documents: invoice ×2, receipt, payslip, bank statement, letter, incident report, resume, utility bill, delivery challan, PAN card, prescription; with photo/scan effects) | 12 | 12/12 | 0.895 | 0.944 |
| holdout3 (OCR suite; texts not written for this feature, no expected types) | 12 | – | 0.744 | 0.879 |

Caveats: the `documents` texts were written together with the rules, so 12/12 shows that each
type is recognisable, not that real documents will be classified as well. On the first run the
same suite gave recall 0.80; merged OCR columns lost labels, which led to the label fallback and
inner-label splitting. holdout3 was not used for tuning. Its misses come from images where OCR
itself failed (for example a dark, shadowed, compressed screenshot). Remaining misses are OCR
misreads ("0" read as "O" in an IFSC or vehicle number, a dropped letter in an e-mail address).
For IFSC and vehicle numbers they are flagged as `invalid`, not silently corrected.

## Adding samples

1. Put an image or PDF in a folder, for example `my_dataset\invoice_01.png`.
2. Next to it, create `invoice_01.gt.txt` (UTF-8) containing the **manually verified** text.
   Line breaks don't matter under the default normalization.
3. For multi-page PDFs, put the text of all pages in order in the one `.gt.txt` file.
4. Run `python -m docintel.evaluation.benchmark my_dataset --out outputs\my_eval`, and again with
   `--preset auto` or `--preset original` to compare.

You can also check single pages in the app: OCR results > Text > "Check accuracy against a
reference". Compare presets, PSM modes and engines on your own dataset before changing defaults,
and don't publish accuracy figures that weren't measured on a representative dataset.
