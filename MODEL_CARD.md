# Model Card: FieldEye Crop Disease CNN

A **model card** is a standardized way of documenting a trained AI model —
what it is, how it was built, how well it performs, and where it breaks —
so anyone (technical or not) can judge whether it's trustworthy for a given
use. This one is written to be understandable without prior machine
learning knowledge; unfamiliar terms are explained inline and again in the
[Glossary](#glossary) at the end.

---

## Table of Contents

- [Summary](#summary)
- [Model Details](#model-details)
- [Intended Use](#intended-use)
- [Training Data](#training-data)
- [How Training Worked](#how-training-worked)
- [Evaluation Results](#evaluation-results)
- [Reading the Per-Class Table](#reading-the-per-class-table)
- [Known Limitations & Failure Modes](#known-limitations--failure-modes)
- [Future Work / Upgrade Path](#future-work--upgrade-path)
- [Ethical Considerations](#ethical-considerations)
- [Glossary](#glossary)

---

## Summary

This model looks at a photo of a single tomato leaf and predicts which of 5
conditions it shows: healthy, or one of 4 diseases (early blight, late
blight, leaf mold, septoria leaf spot). On a held-out test set of 1,084
images it had never been trained or tuned on, it correctly identified the
condition **96.22%** of the time, with a balanced-across-classes score
(**macro F1**, explained below) of **96.03%**.

## Model Details

| Property | Value | What this means |
|---|---|---|
| Architecture | Custom CNN, 4 convolutional blocks | A **Convolutional Neural Network (CNN)** is a deep learning model built specifically to recognize visual patterns in images — it learns simple features first (edges, colors) and combines them into complex ones (leaf spot shapes, discoloration patterns). |
| Parameters | ~590,000 | The internal adjustable numbers the model learns during training. This is a small model — deliberately so, see [How Training Worked](#how-training-worked). |
| Framework | PyTorch | The open-source deep learning library used to build and train the model. |
| Input | RGB image, resized to 128×128 pixels, color-normalized | Every photo is resized to a consistent size and its colors adjusted to a standard numeric range before the model sees it — this is required so the model receives consistent input regardless of the original photo's size or camera. |
| Output | A probability for each of the 5 classes (they add up to 100%) | E.g. "99% late blight, 0.8% early blight, 0.2% other..." — not just a single guess, but a full confidence breakdown. |
| Training hardware | Standard CPU, no GPU | A deliberate design choice — see [How Training Worked](#how-training-worked) for why. |

## Intended Use

This model is intended as a **portfolio and research demonstration** of an
offline-capable, low-resource crop diagnosis approach, aimed conceptually at
supporting smallholder farmers who lack easy access to agricultural
advisory services.

**It is not validated for real field or commercial deployment.** See
[Known Limitations](#known-limitations--failure-modes) for exactly why, and
[Future Work](#future-work--upgrade-path) for what would need to change
before it could be.

## Training Data

- **Source:** [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease),
  a public, widely-used dataset of plant leaf photographs.
- **Scope used here:** a 5-class subset covering only tomato leaves (the
  full PlantVillage dataset covers 38 classes across multiple crops — this
  project intentionally scoped down to 5 for a focused, fully reproducible
  build; see [Future Work](#future-work--upgrade-path)).
- **The 5 classes:**
  - `tomato_healthy`
  - `tomato_early_blight`
  - `tomato_late_blight`
  - `tomato_leaf_mold`
  - `tomato_septoria_leaf_spot`
- **Total images:** 7,223
- **Images per class:**

  | Class | Images |
  |---|---|
  | tomato_late_blight | 1,909 |
  | tomato_septoria_leaf_spot | 1,771 |
  | tomato_healthy | 1,591 |
  | tomato_early_blight | 1,000 |
  | tomato_leaf_mold | 952 |

  The largest class has about 2x the images of the smallest — a **mild
  class imbalance**. This is common in real-world data and one reason
  **macro F1** (explained below) was used as the primary evaluation metric
  rather than raw accuracy alone.

- **Data split:** 70% train (5,055 images) / 15% validation (1,084 images) /
  15% test (1,084 images). The split is **stratified**, meaning each of the
  three groups contains roughly the same proportion of each disease class
  as the full dataset — this prevents, for example, the test set ending up
  with too few examples of any one disease by chance.

- **Augmentation (applied only to training images, not validation or
  test):** random horizontal flips, small rotations (±15°), and random
  adjustments to brightness/contrast/saturation. This artificially varies
  the training images so the model learns to recognize a disease under
  different lighting and angles, rather than memorizing exact pixel
  patterns from a fixed set of photos.

## How Training Worked

The model was trained for 20 **epochs** (an epoch = one full pass through
all 5,055 training images). After each epoch, it was checked against the
1,084 validation images — never used to directly update the model's
learned parameters, only to monitor how well it was generalizing.

A deliberate design choice: **the model uses a compact custom architecture
(4 convolutional blocks) instead of a larger pretrained backbone** (such as
ResNet18, a much bigger, industry-standard architecture pretrained on
millions of general images). A pretrained backbone would likely reach
comparable or higher accuracy with less training data, but is significantly
larger and slower to train without a GPU. This project prioritized being
**fully reproducible on an ordinary CPU** — anyone can clone the repository
and retrain this exact model in roughly 1-2 hours with no special hardware.
The trade-off is documented, not hidden: see
[Future Work](#future-work--upgrade-path) for the pretrained-backbone
upgrade path.

Validation performance fluctuated noticeably between epochs rather than
improving smoothly (see `reports/figures/training_curves.png` in the main
repository) — this is normal given the validation set's size, but it means
simply using "whatever the model looked like after the last epoch" would
have been a worse choice than checking every epoch. So: **the checkpoint
(saved version) used for all reported results is whichever epoch scored
best on validation macro F1 — epoch 15, with a validation macro F1 of
0.9645 — not epoch 20**, the final one. Epoch 20's own validation score
(0.8902) was actually one of the worst in the run, which is exactly the
scenario this checkpointing strategy protects against.

## Evaluation Results

All results below come from a **single evaluation run on the test set** —
1,084 images that were never used in training and never used to pick the
best checkpoint. This is what makes these numbers an honest estimate of how
the model would perform on genuinely new photos, rather than a number it
was indirectly tuned to hit.

| Metric | Score | Plain-language meaning |
|---|---|---|
| Accuracy | 96.22% | Out of 100 test photos, about 96 were correctly diagnosed. |
| Macro F1 | 96.03% | The average of the F1 score (a precision/recall balance, explained below) across all 5 classes, weighted equally — confirms the model isn't just strong on the biggest classes. |

### Per-class breakdown

| Class | Precision | Recall | F1 | Support (test images) |
|---|---|---|---|---|
| tomato_healthy | 1.00 | 1.00 | 1.00 | 239 |
| tomato_early_blight | 0.86 | 0.99 | 0.92 | 150 |
| tomato_late_blight | 0.98 | 0.90 | 0.94 | 286 |
| tomato_leaf_mold | 0.95 | 0.99 | 0.97 | 143 |
| tomato_septoria_leaf_spot | 0.98 | 0.97 | 0.98 | 266 |

## Reading the Per-Class Table

- **Precision** answers: "When the model said 'this is class X,' how often
  was it actually X?" Low precision means the model over-predicts that
  class — it cries wolf.
- **Recall** answers: "Of all the images that actually were class X, how
  many did the model correctly catch?" Low recall means the model misses
  real cases of that class.
- **F1** combines both into one number per class, so neither can be
  ignored.
- **Support** is simply how many test images belonged to that class — more
  support generally means a more statistically reliable score for that row.

**The standout number:** `tomato_early_blight` has the lowest precision
(0.86) despite very high recall (0.99). In plain terms: the model almost
never *misses* an actual early blight case, but it also mislabels some
*other* diseases as early blight fairly often. Cross-referencing with the
confusion matrix (`reports/figures/confusion_matrix.png` in the main repo)
shows this is driven mostly by late blight images being called early
blight — 21 of them. This lines up with real plant pathology: **early and
late blight look visually similar to each other in their early stages**,
before the more distinctive late-stage necrosis (tissue death) patterns
develop. The model's confusion tracks a genuine, documented diagnostic
difficulty rather than a random or nonsensical error — which is a
meaningfully different (and more trustworthy) situation than a model
confusing two visually unrelated diseases.

## Known Limitations & Failure Modes

- **Early/late blight confusion (see above):** the single largest and most
  systematic error pattern. Anyone using this model should treat an "early
  blight" prediction with awareness that it has a real chance of actually
  being late blight.
- **Single-crop, controlled-photo scope:** trained and tested only on
  tomato leaves, photographed individually against a plain, uniform
  background under relatively consistent lighting (PlantVillage's
  studio-style images). Real field photos — cluttered backgrounds, multiple
  overlapping leaves, harsh or uneven sunlight, motion blur — were never
  part of training or testing, so performance in those conditions is
  **unknown and likely lower** than the 96%+ reported here.
- **Confidence scores are not calibrated:** the percentages this model
  outputs (e.g. "99.97% confident") are raw model outputs, not independently
  verified to match real-world correctness rates. A prediction at 99%
  confidence is not guaranteed to be right 99% of the time — it simply
  means the model's internal signal was strong.
- **Architecture size trade-off:** the compact custom CNN was chosen for
  full CPU-only reproducibility (see [How Training Worked](#how-training-worked)),
  at the likely cost of some accuracy compared to a larger, pretrained
  architecture.
- **No deployment hardening:** this model has not been converted for
  on-device (mobile/offline) use, load-tested, or checked for adversarial
  robustness (deliberately crafted images designed to fool it).

## Future Work / Upgrade Path

- **On-device deployment:** convert the model with **quantization** (a
  process that shrinks a trained model's size and speeds up its
  predictions, typically for running on phones or other low-power devices,
  at a small accuracy cost) to TFLite or ONNX format, enabling fully
  offline use — this is the natural next step toward a real field tool.
- **Satellite + weather data fusion:** pairing this single-photo classifier
  with regional satellite vegetation index (NDVI) data and weather trend
  data, to forecast disease risk across a whole field before symptoms are
  visible anywhere else in it. Explicitly out of scope for this project — a
  separate systems-integration effort layered on top of this classifier,
  not a natural extension of it.
- **Broader class coverage:** extending from this 5-class tomato subset to
  PlantVillage's full 38-class, multi-crop dataset.
- **Pretrained backbone fine-tuning:** replacing the compact custom CNN
  with a fine-tuned pretrained architecture (e.g. ResNet18), trading
  CPU-only training for likely higher accuracy — reasonable once GPU
  training resources are available.
- **Real field-photo validation set:** collecting or sourcing a small set
  of genuine, uncurated phone photos to test how far performance drops
  outside the PlantVillage studio-photo distribution — critical before any
  real deployment claim.

## Ethical Considerations

This model is intended to **assist**, not **replace**, informed agricultural
decision-making. Two failure directions matter here:

- **False negatives** (predicting `tomato_healthy` when disease is actually
  present) could delay treatment and worsen crop loss.
- **Misclassification between diseases** (as documented above, especially
  early vs. late blight) could lead to an ineffective treatment being
  applied for the actual disease present.

Any use of this model beyond portfolio/research purposes would require
field validation against real-world photos and a clear disclaimer to end
users — which the accompanying Streamlit demo already includes as a
starting point.

## Glossary

- **CNN (Convolutional Neural Network):** a deep learning model architecture
  built specifically for images, learning visual patterns in layers from
  simple (edges, colors) to complex (shapes, textures, objects).
- **Epoch:** one complete pass through the entire training dataset during
  training.
- **Checkpoint:** a saved snapshot of a model's learned parameters at a
  given point in training, so it can be reloaded without retraining from
  scratch.
- **Training / Validation / Test split:** three separate, non-overlapping
  portions of the data. The model learns from training data; validation
  data checks progress during training and helps pick the best checkpoint;
  test data is used exactly once, at the very end, for an unbiased final
  score.
- **Stratified split:** splitting data so that each portion (train,
  validation, test) preserves the same proportion of each class as the
  full dataset.
- **Accuracy:** the percentage of all predictions that were correct.
- **Precision:** of everything the model labeled as a given class, the
  percentage that actually was that class.
- **Recall:** of everything that actually was a given class, the
  percentage the model correctly identified.
- **F1 score:** a single number combining precision and recall for one
  class, useful when both false positives and false negatives matter.
- **Macro F1:** the F1 score averaged equally across every class, so
  classes with fewer examples count just as much as classes with more —
  prevents a model from looking good overall while quietly underperforming
  on a smaller class.
- **Confusion matrix:** a grid showing, for every true class and every
  predicted class, how many test examples fell into that combination —
  makes it easy to see exactly which categories a model confuses with each
  other, not just how often it's wrong overall.
- **Support:** the number of real examples of a given class present in a
  dataset or evaluation set.
- **Data augmentation:** artificially varying training images (flipping,
  rotating, adjusting color, etc.) so a model learns to generalize rather
  than memorize specific images.
- **Quantization:** a technique that reduces a trained model's size and
  speeds up its predictions, usually to enable running it on phones or
  other low-power devices, at a small cost to accuracy.
- **Parameters:** the internal numeric values a model adjusts while
  learning; more parameters generally means a larger, more capable, but
  also slower and more resource-hungry model.
- **Calibration (of confidence scores):** the degree to which a model's
  stated confidence (e.g. "90% sure") actually matches its real-world
  accuracy rate at that confidence level. An uncalibrated model may be
  systematically over- or under-confident.