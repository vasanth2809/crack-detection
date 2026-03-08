# Research Report: Automated Crack Detection in UTM Testing
### Deep Analysis, Dataset Characterization, and ML Strategy

---

## 1. Project Overview

This project builds a computer vision pipeline to detect crack initiation on material specimens during Universal Testing Machine (UTM) tests. The system performs **binary image classification** — labeling each frame as either **Damaged** (crack present) or **Undamaged** (intact) — with the long-term goal of integrating the detection signal into UTM control logic to halt loading automatically when a crack is detected.

The primary focus of this report is **Phase 1**: building and validating a CNN-based classifier that will run in Google Colab, using the labeled image dataset extracted from a UTM test video.

---

## 2. Workspace Structure

```
crack-detection/
├── code.ipynb              ← Empty (0 bytes) — no code written yet
├── problem_statement.md    ← Project methodology and objectives
├── research.md             ← This file
└── Dataset/
    ├── Damaged/            ← 359 JPEG frames (frame_01416 to frame_01774)
    └── Undamaged/          ← 209 JPEG frames (frame_00617 to frame_00825)
```

The notebook is a blank slate. All implementation starts from scratch.

---

## 3. Dataset Analysis

### 3.1 Core Statistics

| Property | Damaged | Undamaged | Total |
|----------|---------|-----------|-------|
| Image count | 359 | 209 | **568** |
| Class share | 63.2% | 36.8% | 100% |
| Frame range | 1416 – 1774 | 617 – 825 | — |
| Avg file size | 473.7 KB | 465.6 KB | ~470 KB |
| Total size | 166.1 MB | 95.0 MB | **261.1 MB** |
| Resolution | 3840 × 2176 px | 3840 × 2176 px | 4K RGB JPEG |
| Class ratio | 1.72 : 1 (Damaged : Undamaged) | | |

### 3.2 Frame Sequence Analysis

The filenames encode chronological order from a single UTM test video. The full sequence is:

```
Frame 617 ─────────────────────────── Frame 825      (Undamaged, 209 frames)
                ← 591 unlabeled frames (gap) →
Frame 1416 ─────────────────────────── Frame 1774    (Damaged, 359 frames)
```

**Critical observation:** There is a **591-frame gap** between the end of the undamaged set (frame 825) and the start of the damaged set (frame 1416). This gap represents approximately 19.7 seconds of footage at 30 fps. These frames were deliberately excluded, likely because they represent the ambiguous transition zone — early elastic/plastic deformation before a visible crack initiates. The Damaged set begins only once crack initiation was clearly observable on the specimen.

This means:
- The dataset does **not** include frames right at the moment of crack initiation — the hardest and most valuable frames to detect.
- The "Damaged" class represents **established crack propagation**, not necessarily the very earliest stage.
- The "Undamaged" class represents **early-to-mid loading phase** before any visible damage.

### 3.3 Image Characteristics

- **Resolution**: 3840 × 2176 pixels (4K, 16:9 ultra-wide frame)
- **Color mode**: RGB
- **Format**: JPEG (lossy compression)
- **Source**: Single UTM test session, fixed camera angle, consistent lighting

All images have identical dimensions, meaning no resizing inconsistencies exist before model input.

### 3.4 Class Imbalance Assessment

The 1.72:1 ratio (Damaged:Undamaged) represents a **moderate class imbalance**. It is not extreme (unlike 10:1 ratios seen in anomaly detection), but combined with the project's priority on **recall for the Damaged class**, it requires deliberate mitigation:

- **Class weights** during training
- **Lower decision threshold** (e.g., 0.35–0.4 instead of 0.5)
- **Stratified splits** to maintain the ratio in all splits

---

## 4. Critical Technical Observations

### 4.1 Temporal Correlation — The Most Important Methodological Issue

Since all 568 images are **sequential frames from a single continuous video**, adjacent frames (e.g., frame 1416 and frame 1417) are nearly pixel-identical. If we perform a **standard random 80/20 split**, temporally adjacent frames will appear in both the training and test sets. The model will then achieve artificially inflated accuracy by recognizing near-duplicate frames rather than learning genuine visual crack features.

**This is temporal data leakage** and it is the number one pitfall for this dataset.

**Mitigation**: Use a **temporal (chronological) split** per class:
- First 70% of each class's frames → Train
- Next 15% → Validation
- Last 15% → Test

This ensures the model is evaluated on frames it has never seen AND that are temporally separated from training frames.

### 4.2 High Resolution vs. Useful Signal

At 3840 × 2176 pixels, the raw images are enormous. Deep learning models (e.g., EfficientNetB0) expect 224 × 224 inputs. Direct resizing to 224 × 224 **discards 99.7% of pixels**, and a hairline crack that spans 10 pixels in the original image may vanish or become sub-pixel after resizing.

**Options (ranked by recommended approach):**

1. **ROI crop + resize** *(Recommended)*: Identify the specimen region (likely the vertical center strip of the image, away from fixtures and background), crop to it, then resize to 224 × 224. This preserves crack detail at the cost of requiring manual ROI definition.

2. **Multi-scale / patch-based approach**: Divide the 4K image into overlapping patches, run the model per-patch, and aggregate predictions (logical-OR for damage detection).

3. **Direct resize** *(Baseline approach)*: Simple but lossy. Still viable for a quick Phase 1 baseline since the global appearance difference between Damaged and Undamaged frames (surface texture, edge roughness, line patterns from crack) may still be visible at 224 × 224.

### 4.3 Very Small Dataset

568 images from a single video session is a **small dataset** for deep learning. Practical implications:
- Transfer learning is not just preferred — it is **mandatory**
- A model trained from scratch will overfit almost immediately
- Aggressive data augmentation is critical
- Cross-validation should be considered if time permits

### 4.4 Single-Source Domain Bias

All data comes from one UTM test of one material specimen. The model has zero exposure to:
- Different materials (steel, aluminum, polymer, composite)
- Different lighting conditions
- Different camera angles or zoom levels
- Different UTM machine setups
- Different specimen geometries (dogbone, cylindrical, rectangular)

A model trained only on this data will generalize poorly to new UTM setups. This is expected for Phase 1 (feasibility study), but **must be flagged** as a limitation and addressed in later phases through additional data collection.

### 4.5 Label Quality at the Boundaries

The frames immediately following the unlabeled gap (damaged frames 1416–1430) may show **very early, subtle cracks** that are borderline damaged. Similarly, undamaged frames 815–825 may show **surface deformation or micro-texture changes** that precede cracking. These boundary frames carry the highest labeling uncertainty and could introduce noise.

**Mitigation options:**
- Remove the first 20 frames of each class from training (add to a "boundary" set for manual review)
- Use label smoothing during training to reduce overconfidence on noisy labels

---

## 5. Recommended ML Architecture and Strategy

### 5.1 Architecture Choice: EfficientNetB0 with Transfer Learning

**Why EfficientNetB0:**
- Pre-trained on ImageNet (1.2M diverse images, 1000 classes)
- Compound scaling (depth + width + resolution) makes it efficient
- Excellent accuracy-to-parameter ratio on small datasets
- ~5.3M parameters — small enough to avoid overfitting on 568 images with proper regularization
- Native input: 224 × 224 × 3 (matches our resized images)
- Handles internal normalization (no manual rescaling needed)

**Alternative architectures considered:**

| Model | Parameters | Reason |
|-------|-----------|--------|
| **EfficientNetB0** *(Recommended)* | ~5.3M | Best for small datasets, efficient |
| ResNet50V2 | ~23.5M | Good baseline, proven on classification |
| MobileNetV2 | ~3.4M | Best for deployment/real-time inference |
| DenseNet121 | ~7.0M | Good feature reuse, decent for small data |
| Custom CNN | <1M | Baseline only, expected to underperform |

### 5.2 Two-Phase Training Strategy

**Phase 1 — Feature Extraction (20–30 epochs):**
- Freeze all EfficientNetB0 base layers (ImageNet weights preserved)
- Train only the custom classification head
- Learning rate: 1e-3
- Purpose: Adapt the new classification head without disrupting pretrained features

**Phase 2 — Fine-Tuning (20–30 epochs):**
- Unfreeze the top ~40 layers of EfficientNetB0 (last two blocks)
- Use a very low learning rate: 1e-5
- Purpose: Gently adapt deep feature detectors to crack-specific visual patterns
- Risk: Too many unfrozen layers + high LR = catastrophic forgetting

### 5.3 Data Augmentation Strategy

With only 398 training images (70% of 568), augmentation is essential. Recommended augmentations:

| Augmentation | Setting | Justification |
|---|---|---|
| RandomFlip (horizontal) | Always | Specimen is symmetric |
| RandomFlip (vertical) | Always | Camera could be inverted; adds robustness |
| RandomRotation | ±8° | Small rotations from camera tilt |
| RandomZoom | ±10% | Slight camera distance variation |
| RandomContrast | 20% | Lighting variation |
| RandomBrightness | 20% | Lighting variation |

**Do NOT use:**
- Large rotations (>15°) — UTM specimens are loaded vertically; large rotations are unphysical
- RandomHue/Saturation — The crack signal is primarily texture/edge-based, not color-based
- CutOut/MixUp — Can obscure small crack regions; may hurt recall

### 5.4 Loss Function and Class Weighting

Since **missing a crack (False Negative) is far worse than a false alarm (False Positive)**:

```
class_weight = {
    0 (Undamaged): 1.0,
    1 (Damaged):   2.0   ← 2× heavier penalty for missing a crack
}
```

This is applied in `model.fit()` and forces the optimizer to prioritize Damaged recall.

Additionally, use a **lower decision threshold** (0.35–0.4 vs default 0.5) at inference time. This further suppresses False Negatives. The optimal threshold should be derived from the Precision-Recall curve on the validation set.

### 5.5 Prioritized Evaluation Metrics

| Metric | Priority | Reason |
|--------|----------|--------|
| **Recall (Damaged)** | #1 — Primary | Missed crack = specimen fails unexpectedly |
| **F1 Score (Damaged)** | #2 | Balance between recall and precision |
| **AUC-ROC** | #3 | Threshold-independent; overall model quality |
| **Confusion Matrix** | Always | Count FN, FP, TP, TN explicitly |
| Overall Accuracy | Secondary | Misleading with imbalanced classes |

---

## 6. Data Pipeline Design

### 6.1 Temporal-Aware Split (Class-Wise)

```
Damaged (359 frames, frames 1416–1774):
  Train: frames 1416–1665  (251 frames, ~70%)
  Val:   frames 1666–1719  ( 54 frames, ~15%)
  Test:  frames 1720–1774  ( 54 frames, ~15%)

Undamaged (209 frames, frames 617–825):
  Train: frames 617–763    (147 frames, ~70%)
  Val:   frames 764–794    ( 31 frames, ~15%)
  Test:  frames 795–825    ( 31 frames, ~15%)

Combined Totals:
  Train: 398 images (251D + 147U)
  Val:    85 images  (54D +  31U)
  Test:   85 images  (54D +  31U)
```

### 6.2 Input Pipeline (tf.data)

```
File Path → tf.io.read_file → JPEG decode → Resize 224×224 → float32 [0,255]
                                           ↓ (train only)
                                      Augmentation layers
                                           ↓
                              EfficientNetB0 internal normalization
                                           ↓
                                    Prediction (sigmoid)
```

### 6.3 Batch Size Recommendation

- Google Colab Free (T4 GPU, ~15GB VRAM): Batch size 32 is comfortable
- If CUDA OOM errors occur: reduce to 16
- Smaller batch size = noisier gradients, but can act as implicit regularization

---

## 7. Implementation Checklist for Phase 1 Notebook

The `code.ipynb` notebook implements the following in order:

1. **GPU verification + all imports**
2. **Google Drive mount** (dataset assumed at `/content/drive/MyDrive/crack-detection/Dataset`)
3. **Configuration cell** (all hyperparameters in one place)
4. **Dataset loading with temporal split** (per-class chronological split)
5. **Exploratory Data Analysis** (sample images, class distribution bar chart)
6. **Augmentation visualization** (show augmented samples)
7. **tf.data pipeline** (train with augmentation, val/test without)
8. **EfficientNetB0 model definition** (Phase 1: frozen, Phase 2: partially unfrozen)
9. **Phase 1 training** (head only, with class weights, EarlyStopping)
10. **Phase 2 fine-tuning** (low LR, top layers unfrozen)
11. **Training history plots** (loss, accuracy, recall, AUC for both phases)
12. **Threshold optimization** (Precision-Recall curve, F1 vs threshold)
13. **Test set evaluation** (confusion matrix, classification report, ROC curve)
14. **Grad-CAM visualization** (model attention overlay on test images)
15. **Model saving** (SavedModel + H5 to Google Drive)

---

## 8. Known Risks and Mitigation Plan

| Risk | Severity | Mitigation |
|------|----------|------------|
| Temporal data leakage | Critical | Use chronological (temporal) split, NOT random split |
| Overfitting (small dataset) | High | Transfer learning, augmentation, Dropout, EarlyStopping |
| Crack features lost in resize | High | Consider ROI crop before resize in Phase 2 |
| Single-source domain bias | High | Accepted limitation for Phase 1; collect more test sessions later |
| Boundary label noise | Medium | Label smoothing, or exclude first/last 20 frames per class |
| Class imbalance | Low-Medium | Class weights {0:1.0, 1:2.0} + lower threshold |
| Grad-CAM saliency on wrong regions | Medium | Inspect visually; if model attends to fixtures/background, ROI crop is needed |

---

## 9. Phase 2 Research Directions (After Phase 1 Completion)

### 9.1 ROI-Based Feature Enhancement
Once the baseline classifier is validated, the most impactful improvement is proper Region of Interest (ROI) extraction. The 4K images contain ~80% irrelevant content (UTM fixtures, background, measurement scales). Two approaches:
- **Manual ROI**: Define a fixed crop polygon per camera setup
- **Saliency-guided ROI**: Use Phase 1's Grad-CAM maps to empirically identify where the model is actually looking; define ROI around those regions

### 9.2 Multi-Scale Analysis
Implement a sliding window or patch pyramid approach. Divide the image into overlapping patches at multiple scales, classify each patch, and use patch-level predictions to generate a spatial damage heatmap. This is more interpretable and better suited for real-time deployment.

### 9.3 Temporal Sequence Models
Since the data is sequential video, a temporal model could exploit inter-frame continuity:
- **CNN + LSTM**: Frame-level features from CNN fed into LSTM for temporal reasoning
- **3D CNN (C3D/SlowFast)**: Treats short video clips as 3D inputs
- **Advantage**: Could detect the *trend* of increasing damage severity rather than just a single-frame snapshot

### 9.4 Expand Dataset with More UTM Tests
This is the highest-priority long-term action. Collecting frames from 3–5 more UTM tests on different specimens and materials would dramatically improve generalization.

**Data collection protocol for future tests:**
- Capture frames at consistent intervals (e.g., every 2 seconds)
- Label the crack initiation frame precisely by consulting load-displacement curve data
- Capture both early hairline cracks (most valuable) and complete fracture

### 9.5 Semi-Supervised Learning for the 591 Unlabeled Frames
The 591-frame gap between classes contains the most important frames — the crack initiation transition. These frames are currently discarded. Options:
- **Manual labeling** (expensive but ideal)
- **Pseudo-labeling**: Use Phase 1 model to generate soft labels on these frames, then retrain
- **Temporal smoothing**: Model the frame sequence as a change-point detection problem

### 9.6 Object Detection for Crack Localization (Phase 4)
Binary classification tells us *whether* there is a crack. Object detection can tell us *where* it is. YOLO-based models (YOLOv8/YOLOv9) can provide bounding boxes around crack regions. This requires additional annotation (bounding box labels), but enables:
- Precise crack location tracking
- Crack propagation velocity measurement
- Integration with FEM (Finite Element Method) models

---

## 10. Architecture Decision: Why NOT a Custom CNN for Phase 1

A custom CNN trained from ImageNet-sized weights would require thousands of labeled images to learn useful crack detectors. With 568 images:
- A 4-block custom CNN would reach ~70–75% accuracy before overfitting severely
- EfficientNetB0 with transfer learning should reach 85–95% with proper tuning
- The gap is too large to justify using a custom CNN as the primary model

Custom CNN is included in the notebook as a **baseline benchmark** only, not as the main model.

---

## 11. Google Colab Deployment Notes

**Dataset upload strategy**: Upload the entire `Dataset/` folder to Google Drive under `/MyDrive/crack-detection/Dataset/`. The total size is 261 MB, which Google Drive handles without issue.

**Runtime requirements**:
- Runtime: GPU (T4 via Colab Free, or A100 via Colab Pro)
- RAM: ~12 GB is sufficient (images are loaded in batches, not all at once)
- Training time estimate: Phase 1 + Phase 2 together should complete in 20–40 minutes on T4 GPU

**Image loading**: Using `tf.data` with `num_parallel_calls=AUTOTUNE` and `.prefetch(AUTOTUNE)` ensures the GPU is never starved for data during training.

---

## 12. Summary of Findings

| Aspect | Finding |
|--------|---------|
| Dataset size | 568 images — small, demands transfer learning |
| Dataset source | 1 UTM test video — single-source, limited generalization |
| Image resolution | 4K (3840×2176) — must resize, ROI crop recommended |
| Frame naming | Sequential — temporal split is MANDATORY to avoid leakage |
| Class balance | 1.72:1 (Damaged:Undamaged) — moderate imbalance |
| Unlabeled gap | 591 frames of transition zone — omitted from dataset |
| Existing code | None — notebook is empty (0 bytes) |
| Primary challenge | Making 568 images generalize; also avoiding temporal leakage |
| Recommended model | EfficientNetB0 with 2-phase transfer learning |
| Primary metric | Recall for Damaged class (missing a crack is unacceptable) |
| Phase 1 goal | Prove feasibility; establish a working pipeline |
