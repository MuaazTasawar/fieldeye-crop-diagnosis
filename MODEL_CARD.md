# Model Card: FieldEye Crop Disease CNN

## Model Details
- **Architecture:** Custom 4-block CNN (Conv-BatchNorm-ReLU-MaxPool x4 → AdaptiveAvgPool → FC 128 → FC 5), ~590K parameters
- **Framework:** PyTorch
- **Input:** RGB leaf image, resized to 128x128, ImageNet-normalized
- **Output:** Softmax probability distribution over 5 classes
- **Training hardware:** CPU (no GPU required — this is a deliberate design choice, see Limitations)

## Intended Use
Diagnose tomato leaf disease from a single photograph, as a portfolio/research
demonstration of an offline-capable, low-resource crop diagnosis system aimed
at smallholder-farmer contexts. **Not validated for field or commercial
deployment.**

## Training Data
- **Source:** PlantVillage dataset (public), tomato subset
- **Classes (5):** `tomato_healthy`, `tomato_early_blight`, `tomato_late_blight`, `tomato_leaf_mold`, `tomato_septoria_leaf_spot`
- **Total images:** 7,223 (952–1,909 per class — mild imbalance, not severe enough to require resampling)
- **Split:** 70% train (5,055) / 15% val (1,084) / 15% test (1,084), stratified by class
- **Augmentation (train only):** random horizontal flip, ±15° rotation, color jitter (brightness/contrast/saturation)

## Evaluation Results (held-out test set, 1,084 images, never seen during training or model selection)

| Metric | Score |
|---|---|
| Accuracy | 96.22% |
| Macro F1 | 96.03% |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| tomato_healthy | 1.00 | 1.00 | 1.00 | 239 |
| tomato_early_blight | 0.86 | 0.99 | 0.92 | 150 |
| tomato_late_blight | 0.98 | 0.90 | 0.94 | 286 |
| tomato_leaf_mold | 0.95 | 0.99 | 0.97 | 143 |
| tomato_septoria_leaf_spot | 0.98 | 0.97 | 0.98 | 266 |

Model selection used the checkpoint with the **best validation macro F1**
across 20 epochs (epoch 15, val F1 = 0.9645) rather than the final epoch's
weights, since validation performance was noisy epoch-to-epoch (see
`reports/figures/training_curves.png`) and the last epoch was not the best.

## Known Limitations & Failure Modes
- **Early/late blight confusion:** the largest error source (21 of 286
  late-blight test images misclassified as early blight — see
  `reports/figures/confusion_matrix.png` and `misclassified_examples.png`).
  This tracks a real plant-pathology difficulty: early- and late-stage lesions
  from these two diseases visually overlap, particularly before necrosis
  fully develops.
- **Single-crop, single-condition scope:** trained only on tomato leaves
  against a controlled/uniform background (PlantVillage's studio-style
  photos). Performance on real field photos — variable lighting, cluttered
  backgrounds, multiple leaves, partial occlusion — is untested and likely
  lower than the reported test metrics.
- **No calibration validation:** confidence scores are raw softmax outputs,
  not calibrated against real-world accuracy; a 99% confidence prediction is
  not guaranteed to be correct 99% of the time.
- **Architecture choice trade-off:** a compact from-scratch CNN was used
  instead of a fine-tuned pretrained backbone (e.g. ResNet18) specifically so
  the full pipeline trains in ~1-2 hours on CPU, keeping this reproducible
  without GPU access. A pretrained-backbone fine-tune would very likely
  reach comparable or better accuracy with less data, at the cost of a larger
  model and slower CPU training.

## Future Work / Upgrade Path
- **Full 38-class PlantVillage scope** (multiple crops, more diseases),
  rather than this 5-class tomato subset.
- **On-device deployment:** convert to TFLite/ONNX with quantization for
  offline mobile inference — this is the missing piece toward the original
  "FieldEye" vision of a fully offline field-diagnosis app.
- **Satellite/weather fusion:** combine this point-in-time image classifier
  with regional NDVI and weather time-series data to forecast field-wide
  disease risk before symptoms are visible elsewhere in a field — this was
  scoped out of the MVP as a separate systems-integration project on top of
  this core classifier.
- **Pretrained backbone fine-tune** (ResNet18/EfficientNet) as an
  accuracy-upgrade path once GPU training is available.
- **Real field-photo validation set** to test generalization beyond the
  PlantVillage studio-photo distribution.

## Ethical Considerations
This tool is intended to assist, not replace, informed agricultural
decision-making. Misdiagnosis (e.g. predicting `tomato_healthy` when disease
is present) could delay treatment; misclassifying between similar diseases
could lead to an ineffective treatment being applied. Any deployment beyond
portfolio/research use would need field validation and a clear disclaimer
to end users, which the Streamlit demo includes.