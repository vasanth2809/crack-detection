# Automated Crack Detection in UTM Testing — Project Bible
### A Complete Guide: Problem → Dataset → Design → Implementation → Results → Future

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Why This Matters](#2-why-this-matters)
3. [The Dataset](#3-the-dataset)
4. [Key Challenges](#4-key-challenges)
5. [Solution Design](#5-solution-design)
6. [Architecture Deep Dive](#6-architecture-deep-dive)
7. [Implementation Walkthrough](#7-implementation-walkthrough)
8. [Results](#8-results)
9. [Limitations and Cons](#9-limitations-and-cons)
10. [Future Improvements](#10-future-improvements)
11. [Quick Reference](#11-quick-reference)

---

## 1. The Problem

### What is a UTM?

A **Universal Testing Machine (UTM)** is a laboratory instrument used in materials science and engineering to measure how a material behaves under mechanical stress. A specimen (a precisely shaped piece of material — usually metal, polymer, or composite) is clamped into the machine, and the machine slowly pulls or pushes it until it deforms or breaks. Engineers study this process to understand how strong a material is, how it fails, and when it starts to crack.

### What is the problem being solved?

During a UTM test, the moment a crack first appears on the specimen is called **crack initiation**. This is a critical event because:

- It marks the beginning of material damage
- It signals that the specimen is about to fail
- If the machine is not stopped in time, the specimen can fracture completely, destroying potentially valuable intermediate data

Currently, a human operator **watches the specimen by eye** and manually stops the machine when they see a crack. This is:

- **Subjective** — different operators have different visual acuity and reaction times
- **Slow** — human reaction time is hundreds of milliseconds; a crack can propagate in milliseconds
- **Error-prone** — early hairline cracks are nearly invisible to the naked eye
- **Unscalable** — you cannot run high-throughput testing without continuous human presence

**The core problem**: There is no automated system that can watch the specimen in real-time, detect the very first moment a crack appears, and signal the UTM to stop loading.

### What does this project do?

This project builds a **computer vision pipeline** that:

1. Takes image frames extracted from a UTM test video
2. Analyzes each frame using a deep learning model
3. Outputs a binary decision: **Damaged** (crack detected) or **Undamaged** (no crack)
4. Can eventually be integrated with UTM control hardware to stop the machine automatically

This is **Phase 1** of a multi-phase project — proving that the concept works and building a validated baseline.

---

## 2. Why This Matters

### Why use computer vision instead of sensors?

Sensors (strain gauges, acoustic emission detectors) are the traditional route for crack detection. Computer vision was chosen because:

- A camera is **non-contact** — nothing is attached to the specimen that could affect how it deforms
- A camera captures the **full surface** simultaneously — a strain gauge only measures at one point
- Computer vision results are **visually interpretable** — you can see exactly where the crack is
- A trained model can detect **spatial patterns** (crack geometry, edge sharpness) that point sensors cannot

### Why deep learning instead of classical image processing?

Classical approaches (edge detection, thresholding, morphological operations) fail here because:

- Cracks are hairline-thin and low-contrast — simple edge detectors produce too much noise
- Lighting and focus variations change pixel intensities unpredictably between frames
- The UTM fixtures and background clutter dominate the image — classical methods cannot focus on the specimen

Deep learning, specifically **Convolutional Neural Networks (CNNs)**, can learn spatial feature hierarchies directly from data, automatically extracting crack-relevant patterns (texture gradients, edge roughness, surface discontinuities) while suppressing irrelevant background.

---

## 3. The Dataset

### Where did the data come from?

All images were extracted from video recordings of a single UTM test. A camera was mounted facing the specimen and recorded the entire test. Individual frames were extracted and manually classified into two categories.

### Dataset statistics

| Property | Damaged | Undamaged | Total |
|----------|---------|-----------|-------|
| Image count | 359 | 209 | **568** |
| Class share | 63.2% | 36.8% | — |
| Frame numbers | 1416 – 1774 | 617 – 825 | — |
| Resolution | 3840 × 2176 px | 3840 × 2176 px | 4K RGB JPEG |
| Avg file size | ~474 KB | ~466 KB | ~470 KB |
| Total disk size | 166 MB | 95 MB | **261 MB** |

### The frame timeline

```
Frame 617 ——[Undamaged: 209 frames]—— Frame 825
                                             ↓
                      [GAP: 591 unlabeled frames — transition zone, frames 826–1415]
                                             ↓
Frame 1416 ——[Damaged: 359 frames]—— Frame 1774
```

The **591-frame gap** (~20 seconds at 30 fps) was deliberately excluded because it contains the ambiguous transition zone — frames where the specimen is deforming but no clear crack has formed yet. The "Damaged" class starts only once crack propagation was visibly established.

This has an important implication: the dataset does **not** contain the very earliest hairline crack frames. Those are the hardest to detect and also the most valuable.

### What does each class look like?

**Undamaged frames**: The specimen surface is smooth and intact. Edges are clean. The material shows no visible ruptures or line artifacts that deviate from the background texture.

**Damaged frames**: The specimen surface shows visible crack lines — typically dark, thin, irregular linear features running across the specimen's cross-section. As the crack propagates, the separation between the two halves of the specimen becomes increasingly visible.

### Class imbalance

The 1.72:1 (Damaged:Undamaged) ratio is a **moderate imbalance**. It is not severe enough to require oversampling or synthetic data generation (SMOTE), but it is large enough to require:
- Class-weighted loss during training
- A lower decision threshold at inference time than the default 0.5

---

## 4. Key Challenges

### Challenge 1: Temporal Data Leakage (Most Critical)

Since all 568 images come from a **single continuous video**, frame 1416 and frame 1417 are nearly pixel-identical — they differ by perhaps a single frame of motion blur. If you do a standard random 80/20 split, near-duplicate frames will end up in both the training set and the test set.

The model will then "memorize" specific frames during training and achieve near-perfect test accuracy by recognizing memorized frames — **not** by learning to detect cracks. This is called **temporal data leakage**.

**The fix**: Split each class's frames **chronologically**. The first 70% of each class's frames go to training, the next 15% to validation, and the final 15% to testing. This ensures the test set contains only frames the model has never seen, and those frames are temporally separated from the training frames.

### Challenge 2: Resolution Loss

The original images are 3840 × 2176 pixels. EfficientNetB0 (and most CNN models) expect 224 × 224 pixels as input. Resizing from 4K to 224×224 **discards 99.7% of pixels**. A hairline crack that spans 40 pixels in the original may be reduced to a single pixel or disappear entirely after resizing.

For Phase 1, direct resize is used as a baseline (global appearance differences — overall texture changes, dark lines across the frame — are still visible at 224×224). Phase 2 should address this with ROI cropping.

### Challenge 3: Small Dataset

568 images from one video session is **tiny** for deep learning. A model trained from scratch on this data would overfit in the first few epochs and generalize to nothing. Transfer learning (using weights pre-trained on ImageNet's 1.2 million images) is not optional here — it is the only viable approach.

### Challenge 4: Single-Source Domain

Every single image comes from one UTM test, one specimen, one camera angle, one lighting setup. The model has no exposure to different materials, different camera setups, or different crack morphologies. It will likely fail when applied to a different UTM setup "out of the box."

### Challenge 5: Signal-to-Noise Imbalance in the Image

In the raw 4K frames, the specimen occupies only a fraction of the image area. The majority of each frame shows UTM grips/fixtures, background structures, and measurement annotations. The crack signal is buried in a sea of irrelevant content. When the image is downscaled to 224×224, this irrelevant content can dominate the learned features.

---

## 5. Solution Design

### Why EfficientNetB0 as the primary model?

| Model | Parameters | Reason |
|-------|-----------|--------|
| **EfficientNetB0** (chosen) | ~5.3M | Best accuracy-to-parameter ratio; efficient; pre-trained |
| ResNet50V2 | ~23.5M | Proven but 4× larger — overkill for 568 images |
| MobileNetV2 | ~3.4M | Good for deployment but lower accuracy ceiling |
| DenseNet121 | ~7.0M | Good feature reuse but slower than EfficientNet |
| Custom CNN | <1M | Included as a baseline benchmark only |

EfficientNetB0 was chosen because:
- ImageNet pre-training gives it powerful general visual features (edges, textures, shapes) that transfer well to crack detection
- Its 5.3M parameters are small enough to avoid severe overfitting on 568 images with proper regularization
- Compound scaling makes it efficient on Colab's free T4 GPU
- It handles its own internal normalization — no manual pixel rescaling needed

### Two-phase training strategy

Directly fine-tuning all layers of a pre-trained model on a small dataset causes **catastrophic forgetting** — the ImageNet weights get overwritten before the new classification head has stabilized. The two-phase approach solves this:

**Phase 1 — Feature Extraction (head only, backbone frozen)**
- All EfficientNetB0 layers are frozen (weights fixed, not updated)
- Only the new classification head (Dense → Dropout → Dense → sigmoid) is trained
- Learning rate: 1e-3 (higher, since only the head needs to learn)
- The backbone acts as a fixed feature extractor
- Runs for up to 30 epochs with early stopping

**Phase 2 — Fine-Tuning (top backbone layers unfrozen)**
- The last 40 layers of EfficientNetB0 are unfrozen and allowed to update
- Very low learning rate: 1e-5 (tiny steps to avoid destroying pre-trained weights)
- The model gently adapts deep features to crack-specific patterns
- Runs for up to 30 more epochs with early stopping

### Why lower the decision threshold?

In a binary classifier with sigmoid output, the default threshold is 0.5 — if the model outputs a probability > 0.5, it predicts "Damaged." But in this application, **missing a crack (False Negative) is far worse than a false alarm (False Positive)**.

A false positive means: "we thought there was a crack, but there wasn't." The UTM stops unnecessarily — annoying, but recoverable.  
A false negative means: "there was a crack and we missed it." The specimen continues loading and may fracture completely — irreversible.

By lowering the threshold (e.g., 0.40–0.55), we make the model more sensitive to cracks at the cost of more false alarms. The optimal threshold is found empirically by sweeping 0.05 → 0.95 on the validation set and choosing the value that maximizes F1 score for the Damaged class.

### Class weighting

The loss function applies a 2× penalty for misclassifying a Damaged sample. This pushes the optimizer to prioritize Damaged recall over everything else.

```python
CLASS_WEIGHTS = {0: 1.0, 1: 2.0}   # Undamaged normal, Damaged penalized 2×
```

### Data augmentation

With only ~397 training images, augmentation is essential to prevent overfitting. The augmentations chosen are physically plausible for a fixed camera setup:

| Augmentation | Setting | Why |
|---|---|---|
| RandomFlip (H + V) | Always | Specimen symmetry; camera could be inverted |
| RandomRotation | ±8° | Small tilts from camera mounting |
| RandomZoom | ±10% | Slight depth of field variation |
| RandomContrast | ±20% | Lighting intensity variation |
| RandomBrightness | ±20% | Ambient light variation |

Augmentations **not used** (and why):
- Large rotations > 15°: UTM loading is vertical; large rotations are unphysical
- Hue/Saturation shifts: Cracks are texture/edge features, not color features
- CutOut / MixUp: Can erase the crack regions, hurting recall

The augmentation layer is applied **after** caching in the tf.data pipeline, so each epoch receives fresh random transforms on the same cached images — the best of both worlds (no repeated Drive I/O, still random augmentation).

---

## 6. Architecture Deep Dive

### Full model stack

```
Input: (224, 224, 3) float32 [0–255]
        ↓
EfficientNetB0 backbone (ImageNet weights)
  — applies internal normalization (divides by 127.5, subtracts 1)
  — 236 layers total, produces (7, 7, 1280) feature maps
  — pooling="avg" → global average pooling → (1280,) vector
        ↓
BatchNormalization
        ↓
Dense(256, activation="relu", L2 regularization 1e-4)
        ↓
Dropout(0.40)
        ↓
Dense(64, activation="relu", L2 regularization 1e-4)
        ↓
Dropout(0.30)
        ↓
Dense(1, activation="sigmoid")    ← Output: P(Damaged) ∈ [0, 1]
```

**Total parameters**: ~5.5M  
**Trainable in Phase 1**: ~330K (head only)  
**Trainable in Phase 2**: ~1.8M (head + last 40 backbone layers)

### Baseline CNN (benchmark only)

A simple 4-block CNN built from scratch with no pre-trained weights. Its purpose is to establish a lower bound — to quantify how much value transfer learning adds.

```
Input (224, 224, 3)
  → Rescaling (÷255)
  → Conv2D(32) → BN → MaxPool
  → Conv2D(64) → BN → MaxPool
  → Conv2D(128) → BN → MaxPool
  → Conv2D(256) → BN → GlobalAvgPool
  → Dense(256) → Dropout(0.5)
  → Dense(1, sigmoid)
```

The baseline CNN is trained for a maximum of 10 epochs (it overfits quickly) and evaluated at threshold=0.5. Its results tell us directly: "this is what you get without ImageNet."

### Grad-CAM — Visual Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) produces a heatmap showing **which pixels in the image most influenced the model's decision**. It works by:

1. Running a forward pass through the model
2. Computing the gradient of the output probability with respect to the last convolutional layer's feature maps
3. Weighting each feature map channel by its mean gradient
4. Averaging the weighted feature maps into a single spatial map
5. Overlaying this map (colorized with jet colormap) on the original image

If the heatmap highlights the **specimen surface** → the model is learning real crack features.  
If the heatmap highlights **fixtures or background** → the model is cheating, and ROI cropping is needed.

The target layer in EfficientNetB0 is `"top_activation"` — the last activation before global average pooling.

---

## 7. Implementation Walkthrough

### Step-by-step notebook structure

The notebook (`code.ipynb`) is organized into 14 steps, designed to run sequentially on Google Colab with a T4 GPU.

---

**Step 0 — Environment Setup (Cells 2–3)**

Verifies GPU availability and sets memory growth to avoid TensorFlow claiming all VRAM at startup. Imports all required libraries — no pip installs needed (all packages are Colab pre-installed).

---

**Step 1 — Mount Google Drive (Cell 5)**

Mounts Google Drive and sets up paths. Dataset is expected at:
```
MyDrive/Dataset/Damaged/      ← 359 JPEG images
MyDrive/Dataset/Undamaged/    ← 209 JPEG images
```
A sanity-check prints the image count from each folder.

---

**Step 2 — Configuration (Cell 7)**

All hyperparameters are centralized in one cell so nothing needs to be hunted through the notebook:

```python
IMG_SIZE      = 224        # EfficientNetB0 native input size
BATCH_SIZE    = 32         # Comfortable for T4 GPU
P1_EPOCHS     = 30         # Phase 1 max epochs
P1_LR         = 1e-3       # Phase 1 learning rate
P2_EPOCHS     = 30         # Phase 2 max epochs
P2_LR         = 1e-5       # Phase 2 fine-tuning LR
UNFREEZE_FROM = -40        # Unfreeze last 40 backbone layers
CLASS_WEIGHTS = {0:1.0, 1:2.0}
THRESHOLD     = 0.40       # Updated after threshold search
```

---

**Step 3 — Temporal Split (Cell 9)**

The `temporal_split()` function:
1. Globs all `.jpg` files in the class directory
2. Sorts them by the numeric frame number in the filename (e.g., `frame_01416.jpg` → 1416)
3. Splits first 70% to train, next 15% to val, last 15% to test
4. Returns lists of `(path, label)` tuples

This is called separately for Damaged (label=1) and Undamaged (label=0), then merged. Only the training set is shuffled — val and test retain temporal order.

---

**Step 4 — EDA (Cells 11–12)**

Two visualizations:
- Bar chart showing class distribution across the full dataset
- Stacked bar chart showing Damaged vs Undamaged counts per split
- Sample image grid (2×4) displaying evenly-spaced frames from each class with a centre-strip crop to highlight the specimen area

---

**Step 5 — tf.data Pipeline (Cell 14)**

```
Path list → tf.data.Dataset.from_tensor_slices
         → .map(load_and_preprocess)        [reads JPEG, decodes, resizes, casts]
         → .cache()                          [stores decoded images in RAM]
         → (train only) .map(augmentation)  [fresh random transforms per epoch]
         → .batch(32)
         → .prefetch(AUTOTUNE)
```

The `.cache()` call is critical for performance. Without it, TensorFlow reads and decodes every image from Google Drive on every epoch. With it, images are read once and stored in Colab's ~12GB RAM (~342 MB total). Subsequent epochs hit RAM instead of Drive, making each epoch ~10× faster.

---

**Step 6 — Baseline CNN Training (Cells 17–18)**

Builds and trains the benchmark CNN for up to 10 epochs. EarlyStopping monitors `val_recall` with patience=5. After training, plots loss/accuracy/recall/AUC curves.

---

**Step 7 — EfficientNetB0 Training (Cells 20–24)**

Phase 1: Builds the model with frozen backbone, compiles with Adam(1e-3), trains with ModelCheckpoint saving the best `val_recall` epoch to Drive.

Phase 2: Rebuilds the model with `unfreeze_from=-40`, loads Phase 1 best weights, compiles with Adam(1e-5), trains again. Plots individual phase curves and a combined Phase 1 + Phase 2 history with a purple boundary line.

---

**Step 8 — Threshold Optimisation (Cell 26)**

Runs the fine-tuned model over the validation set, collecting predicted probabilities for all 84 images. Sweeps thresholds from 0.05 to 0.95 in 0.01 steps and computes F1, Precision, and Recall at each threshold for the Damaged class. The threshold with the highest F1 score is selected and used for all subsequent evaluation. Plots the three curves vs. threshold with a red vertical line at the optimal value.

---

**Step 9 — Full Evaluation (Cells 28–29)**

`evaluate_model()` runs on the 87 held-out test images for both models:
- Prints `sklearn` classification report (per-class precision, recall, F1, support)
- Plots confusion matrix as a seaborn heatmap
- Plots ROC curve with AUC annotation
- Returns probabilities and predictions for downstream use

A head-to-head comparison table is printed with Accuracy, Precision, Recall, F1, AUC-ROC for both models.

---

**Step 10 — Grad-CAM (Cells 31–32)**

Implements Grad-CAM from scratch using `tf.GradientTape`. Produces a 4-row × 3-column figure showing Original / Heatmap / Overlay for 2 Damaged + 2 Undamaged test images. This is used to visually verify the model is attending to the specimen surface and not the fixtures.

---

**Step 11 — Single Image Inference (Cell 34)**

`predict_image()` is a utility function that takes any JPEG path, runs the model, prints the prediction and probability, and optionally shows a Grad-CAM overlay. Useful for ad-hoc testing of new frames.

---

**Step 12 — Save Model (Cell 36)**

Saves three artifacts to Google Drive:
- `crack_detector_efficientnet/` — TensorFlow SavedModel format (for TF Serving / TFLite)
- `crack_detector_efficientnet.h5` — Keras H5 format (easy reload)
- `crack_detector_cnn_baseline.h5` — Baseline CNN H5
- `model_metadata.json` — Stores optimal threshold, class names, img_size

---

**Step 13 — Reload Check (Cell 38)**

A standalone cell that loads the saved H5 model and metadata from scratch (no variables from earlier cells needed) and runs a smoke test on one Damaged image. Verifies the saved model works in isolation.

---

**Step 14 — Final Results Summary (Cells 39–40)**

Prints a consolidated one-pass summary of everything: dataset splits, training epochs per phase, optimal threshold, per-model metrics table, confusion matrix breakdown, and an automated verdict.

---

### Performance optimization decisions

| Decision | Reason |
|---|---|
| `.cache()` in tf.data pipeline | Eliminates Google Drive I/O every epoch — ~10× speedup |
| Augmentation after cache | Raw images cached; augmentation still random per epoch |
| Baseline CNN limited to 10 epochs | It's a benchmark; no need to spend 30 epochs on it |
| EarlyStopping monitors `val_recall` | Prioritizes crack detection sensitivity over accuracy |
| `mode="max"` for EarlyStopping | Recall and AUC should be maximized, not minimized |

---

## 8. Results

The model was trained and evaluated on Google Colab Free with a T4 GPU.

### Training run summary

| Phase | Epochs Run | Early Stop Triggered? |
|---|---|---|
| Baseline CNN | 6 / 10 | Yes (val_recall plateaued) |
| EfficientNetB0 Phase 1 | 9 / 30 | Yes |
| EfficientNetB0 Phase 2 | 11 / 30 | Yes |

### Optimal threshold

After the F1 sweep on the validation set: **THRESHOLD = 0.55**

### Test set metrics (87 images)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Baseline CNN (t=0.50) | 0.6322 | 0.6322 | **1.0000** | 0.7746 | **1.0000** |
| EfficientNetB0 FT (t=0.55) | **0.9310** | **0.9016** | **1.0000** | **0.9483** | **1.0000** |

### Confusion matrix — EfficientNetB0 (87 test images)

| | Predicted Undamaged | Predicted Damaged |
|---|---|---|
| **True Undamaged** (32) | 26 (TN) | 6 (FP — false alarm) |
| **True Damaged** (55) | 0 (FN — **zero missed cracks**) | 55 (TP) |

### Interpretation

**EfficientNetB0 is performing excellently for this task:**

- **Recall = 1.00** — Zero cracks were missed on the entire test set. This is the most important metric. A false negative (missed crack) means the experiment continues undetected, risking specimen fracture. Zero FN means perfect safety coverage.
- **AUC = 1.00** — The model's probability scores cleanly separate Damaged and Undamaged images with no overlap. This indicates extremely confident, well-separated predictions.
- **Precision = 90.16%** — 6 out of 32 undamaged images were flagged as cracked (false alarms). In practice this means the UTM would have been stopped unnecessarily 6 times out of 87 events — acceptable in a safety-critical system.
- **F1 = 0.9483** — Strong balance between precision and recall.

**Baseline CNN interpretation:**

The baseline CNN also shows Recall=1.00 and AUC=1.00, which sounds impressive but is misleading. Its accuracy of 63.22% equals the proportion of Damaged images in the test set (55/87 = 63.2%). This means the baseline CNN is likely predicting **everything** as "Damaged" — a trivially biased classifier. This actually **validates EfficientNetB0**: it achieves 93.1% accuracy with the same perfect recall, which means it genuinely learned to distinguish cracks rather than just defaulting to one class.

---

## 9. Limitations and Cons

### 9.1 Single-video generalization problem

**This is the most significant limitation.** Every single image in the dataset comes from one UTM test, one specimen, one camera, one lighting setup. The model has never seen:
- A different material (aluminum, polymer, ceramic, composite)
- A different specimen geometry
- Different camera angle or zoom level
- Different UTM fixture design
- Different lab lighting conditions

Deploying this model on a new UTM setup will very likely produce poor results. The "excellent" test metrics measure performance on held-out frames from the **same video** — this is internal validation, not external generalization.

### 9.2 No early crack detection

The "Damaged" class begins at frame 1416 — well into visible crack propagation. The 591-frame transition zone (frames 826–1415), which contains the most valuable and hardest-to-detect frames (first visible hairline cracks), was excluded from the dataset. The model was never trained on these frames. It may fail to detect a crack at its earliest, most subtle stage.

### 9.3 Resolution bottleneck

Downscaling 4K images (3840×2176) to 224×224 discards 99.7% of pixels. A thin crack that spans 10 pixels in the original image (covering a crack of ~0.5mm if the camera is ~1m from the specimen) is reduced to sub-pixel size after resizing. The model works here because the cracks in the "Damaged" class are already well-developed and visible at low resolution. Early hairline cracks may not survive the resize.

### 9.4 No spatial localization

The model outputs a single probability — it tells you **whether** there is a crack but not **where** the crack is. You cannot track crack propagation path, measure crack length, or measure crack opening displacement from this output alone.

### 9.5 Fixed-camera assumption

The pipeline assumes the camera is in a fixed position relative to the specimen. Any camera movement, zoom change, or re-mounting between test sessions changes the apparent position, scale, and appearance of the specimen in the frame. The model is not robust to such changes.

### 9.6 No temporal reasoning

Each frame is classified independently. The model does not know whether the previous 10 frames were undamaged. A real crack detection system should exploit temporal continuity — if the model sees undamaged → undamaged → undamaged → one anomalous frame → undamaged, that anomalous frame is probably noise, not a crack. The current system has no mechanism for this.

### 9.7 Baseline CNN is degenerate

The baseline CNN achieved perfect recall by predicting everything as Damaged. While this was identified and explained, it means the baseline provides no useful lower-bound information for recall. A more informative baseline design would fix the threshold to prevent this degenerate behavior.

### 9.8 AUC = 1.00 raises a flag

Perfect AUC on a test set this small (87 images) from a single video source should be treated with caution. It likely reflects the relatively easy discrimination between well-developed cracks and clearly undamaged frames in this specific video — not necessarily a model that will generalize. It is not a reason to stop improving; it is a reason to test on a second video before declaring success.

---

## 10. Future Improvements

### 10.1 ROI Cropping (Highest Priority, Phase 2)

**What**: Instead of resizing the full 4K frame to 224×224, first crop a Region of Interest around the specimen area (typically a vertical center strip where the specimen gauge section sits), then resize that crop to 224×224.

**Why**: The crack signal is concentrated in a small area (~20–30% of the frame width). Cropping before resize preserves crack detail that is otherwise lost. The Grad-CAM outputs from Phase 1 can guide the ROI definition empirically — wherever the model's attention map is most active is where the ROI should be.

**How to implement**:
```python
# In load_and_preprocess():
img = tf.image.crop_to_bounding_box(img, offset_height=0, offset_width=960, 
                                         target_height=2176, target_width=1920)
img = tf.image.resize(img, [224, 224])
```
The exact crop coordinates depend on the camera setup and can be tuned visually.

### 10.2 More Data from More UTM Tests (Most Important for Generalization)

**What**: Collect labeled frames from 3–5 additional UTM tests on different specimens and/or materials.

**Why**: This directly solves the single-source generalization problem. Even 200–300 frames per new test session, spread across different materials and camera setups, would substantially improve deployment robustness.

**Data collection protocol for future tests**:
- Capture frames every 2 seconds throughout the test
- Precisely identify the crack initiation frame by cross-referencing the load-displacement curve (the point where load drops indicates crack)
- Capture the transition zone (frames 10–20 before and after crack initiation) — these are the most valuable
- Store frame-level metadata: load at time of capture, displacement, material type

### 10.3 Label the 591-Frame Transition Zone

**What**: Manually review the 591 frames between the current Undamaged and Damaged sets and label them (Undamaged / Transitional / Damaged).

**Why**: These frames contain the hardest-to-detect early cracks — exactly the ones that matter most for early intervention. Adding even 100 labeled transition frames to training could dramatically improve the model's sensitivity to subtle cracks.

**Alternative**: Use the Phase 1 model to pseudo-label these frames, then retrain with the pseudo-labels weighted differently from hard labels. This is a form of semi-supervised learning.

### 10.4 Temporal Models (CNN + LSTM)

**What**: Instead of classifying each frame independently, process sequences of N consecutive frames. Extract per-frame features with a CNN, then feed the sequence into an LSTM or Transformer to reason about temporal patterns.

**Why**: Crack initiation shows a temporal signature — the load-displacement curve changes, the surface texture changes progressively. A temporal model can detect the *trend* toward damage rather than reacting to a single frame.

**Architecture sketch**:
```
[Frame t-4] → CNN → feature vector
[Frame t-3] → CNN → feature vector        → LSTM → P(Damaged at t)
[Frame t-2] → CNN → feature vector
[Frame t-1] → CNN → feature vector
[Frame t  ] → CNN → feature vector
```

### 10.5 YOLO-Based Crack Localization (Phase 4)

**What**: Replace the binary classifier with a YOLO object detection model that outputs bounding boxes around detected crack regions.

**Why**: Knowing **where** the crack is, not just **whether** it exists, enables:
- Crack propagation velocity measurement
- Spatial correlation with applied stress field
- Integration with Digital Image Correlation (DIC) measurements
- More interpretable output for materials scientists

**Requirement**: Bounding box annotations for crack regions — this requires manual labeling of at least 200–300 images with polygon annotations around the crack.

### 10.6 TFLite / Edge Deployment

**What**: Convert the saved EfficientNetB0 model to TensorFlow Lite and deploy it on an edge device (Raspberry Pi, NVIDIA Jetson Nano) mounted beside the UTM.

**Why**: Real-time crack detection during a live UTM test requires either a local inference device or a very low-latency connection to Colab. An edge device provides sub-100ms inference per frame at 30fps, enabling real-time monitoring without cloud dependency.

**Estimated performance**: TFLite INT8 quantized EfficientNetB0 runs at ~15ms per frame on a Raspberry Pi 4 — fast enough for 30fps video.

### 10.7 UTM Control Integration

**What**: Connect the model's output directly to the UTM's control interface (typically an analog I/O port or serial command interface) so that when the model predicts "Damaged" with confidence above threshold, it sends a stop signal to the machine.

**Why**: This is the ultimate goal — a fully autonomous crack detection and intervention system with zero human reaction time.

**Architecture**:
```
Camera → Frame capture → Preprocessing → Model inference
                                             ↓
                                    P(Damaged) >= THRESHOLD?
                                         Yes → UTM Stop Signal
                                         No  → Continue
```

### 10.8 Handling the Resolution Problem with Patch-Based Classification

**What**: Divide the original 4K image into a grid of overlapping patches (e.g., 128×128 patches extracted from the specimen ROI), run the model on each patch, and combine patch predictions.

**Why**: Patch-level processing preserves local crack detail that is lost in global resize. If any patch contains a crack, the specimen-level prediction is "Damaged."

**Trade-off**: Inference time scales linearly with number of patches. A 2×4 grid of 512×512 crops from the specimen area would require 8 forward passes per frame instead of 1, but each individual patch has 4× more crack detail than the globally resized image.

---

## 11. Quick Reference

### File structure

```
crack-detection/
├── code.ipynb              ← Complete 44-cell Google Colab notebook
├── problem_statement.md    ← Original project brief and multi-phase roadmap
├── research.md             ← Deep pre-implementation analysis and ML strategy
└── Dataset/
    ├── Damaged/            ← 359 JPEG frames (frame_01416 to frame_01774)
    └── Undamaged/          ← 209 JPEG frames (frame_00617 to frame_00825)
```

### Key hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `IMG_SIZE` | 224 | EfficientNetB0 native size |
| `BATCH_SIZE` | 32 | Fits T4 GPU comfortably |
| `P1_LR` | 1e-3 | Head-only training |
| `P2_LR` | 1e-5 | Fine-tuning (very conservative) |
| `UNFREEZE_FROM` | -40 | Last 40 backbone layers unfrozen in P2 |
| `CLASS_WEIGHTS` | {0:1.0, 1:2.0} | 2× penalty for missing crack |
| `THRESHOLD` | 0.55 (tuned) | From val F1 sweep |

### Metrics priority

1. **Recall (Damaged)** — Primary. Zero missed cracks is the goal.
2. **F1 Score (Damaged)** — Balance recall vs false alarms.
3. **AUC-ROC** — Threshold-independent overall quality.
4. **Confusion Matrix** — Always inspect FN count explicitly.
5. **Accuracy** — Misleading with imbalanced classes; low priority.

### Results achieved (Phase 1)

| Metric | Baseline CNN | EfficientNetB0 FT |
|---|---|---|
| Accuracy | 63.22% | **93.10%** |
| Precision (Damaged) | 63.22% | **90.16%** |
| Recall (Damaged) | **100%** | **100%** |
| F1 (Damaged) | 77.46% | **94.83%** |
| AUC-ROC | 100% | **100%** |
| Missed cracks (FN) | 0 | **0** |
| False alarms (FP) | 32 | 6 |

### Google Drive folder layout (for Colab)

```
MyDrive/
├── Dataset/
│   ├── Damaged/       ← Place 359 JPEG images here
│   └── Undamaged/     ← Place 209 JPEG images here
└── saved_models/      ← Created automatically by notebook
    ├── crack_detector_efficientnet/    ← SavedModel
    ├── crack_detector_efficientnet.h5  ← Keras H5
    ├── crack_detector_cnn_baseline.h5
    ├── model_metadata.json             ← threshold + class names
    ├── class_distribution.png
    ├── sample_images.png
    ├── augmentation_samples.png
    ├── phase1_history.png
    ├── phase2_history.png
    ├── combined_history.png
    ├── threshold_optimisation.png
    ├── baseline_customcnn_evaluation.png
    ├── efficientnetb0_fine-tuned_evaluation.png
    └── gradcam_visualisation.png
```

### Phase roadmap

| Phase | Status | Description |
|---|---|---|
| Phase 1 | ✅ Complete | CNN-based binary classifier — feasibility proven |
| Phase 2 | 🔜 Next | ROI cropping, improved feature extraction |
| Phase 3 | 🔜 Planned | Roboflow deployment, real-world inference validation |
| Phase 4 | 🔜 Future | YOLO localization, UTM control integration |
