# Automated Crack Detection in UTM Testing using Computer Vision

## Problem Statement

In conventional mechanical testing using a Universal Testing Machine (UTM), material specimens are subjected to controlled loading until failure in order to study their mechanical properties. During such tests, crack initiation and crack propagation are critical events that indicate the onset of material damage and imminent failure.

Currently, the detection of cracks during UTM testing is largely manual and visual, where an operator continuously monitors the specimen and manually stops the machine upon observing visible cracking. This approach is subjective, prone to human error, and limited by reaction time, especially in cases where crack initiation is subtle and occurs rapidly.

The absence of an automated real time crack monitoring mechanism leads to several limitations:

- Early stage cracks may go unnoticed.
- Complete fracture of the specimen may occur.
- Valuable intermediate damage data may be lost.
- Experimental control becomes limited.
- Continuous human supervision is inefficient and not scalable for high throughput testing environments.

The core problem addressed in this project is the lack of an automated vision based system capable of detecting early stage crack initiation on material specimens during UTM testing and enabling timely intervention.

---

# Objective

The primary objective of this work is to design and validate an automated computer vision based framework that can detect the presence of cracks or crack initiation on a material specimen using image data captured during UTM loading.

The system is intended to operate on image frames extracted from recorded or live video streams and produce a **binary decision** indicating whether the specimen is:

- **Damaged**
- **Undamaged**

In the long term, this framework aims to support **real time deployment**, where the detection signal can be integrated with the UTM control system to automatically halt loading when crack initiation is detected, thereby preventing catastrophic failure.

---

# Key Challenges

The problem presents several non trivial challenges:

### 1. Subtle Visual Characteristics
Early stage cracks are visually subtle and often appear as:

- Hairline fractures
- Slight edge roughness
- Micro level texture changes

These patterns are difficult to capture using simple threshold based or classical image processing techniques.

### 2. Signal to Noise Imbalance

The crack region typically occupies only a small portion of the image, while most of the frame contains irrelevant elements such as:

- UTM fixtures
- Background structures
- Measurement scale markings

This creates a severe **signal to noise imbalance** for learning based models.

### 3. Visual Similarity Between Classes

Damaged and undamaged images captured under similar lighting and camera conditions can appear almost identical at a global image level. This makes naive image classification approaches unreliable.

---

# Proposed Approach

To address these challenges, a phased and systematic methodology is adopted.

---

## Phase 1. Dataset Creation and Baseline Modeling

Video recordings of UTM tests are processed to extract individual image frames at different stages of loading.

These frames are categorized into two classes:

**Undamaged**
- Frames captured before crack initiation.

**Damaged**
- Frames captured during early crack initiation or visible damage.

A **binary image classification model based on Convolutional Neural Networks (CNNs)** is developed as a feasibility study to distinguish between damaged and undamaged frames.

Model performance is evaluated using standard metrics:

- Accuracy
- Precision
- Recall
- Confusion Matrix

Special emphasis is placed on **recall for the damaged class**, because missing a crack detection is more critical than triggering a false alarm.

---

## Phase 2. Region of Interest Focus and Feature Enhancement (Tenatative can change if anything new discovered)

Based on empirical observations, the model is refined by focusing on the **Region of Interest (ROI)**. Specifically the specimen surface and edges where cracks typically initiate.

Key improvements include:

- Removal of irrelevant background regions
- Improved signal to noise ratio
- Transfer learning using pretrained convolutional backbones
- Enhanced feature extraction for edge and texture patterns

These techniques help the model detect subtle crack structures commonly observed in metallic materials.

---

## Phase 3. Deployment Oriented Validation (not now)

The trained classification model is deployed using **Roboflow** as a lightweight inference platform to simulate real world usage scenarios.

For each input image, the deployed system outputs:

- Binary classification (Damaged or Undamaged)
- Confidence score

This phase ensures consistency between **offline training performance and real world inference behavior**.

---

## Phase 4. Transition to Localization and Real Time Monitoring (not now)

Recognizing the limitations of global image classification for localized defects, the framework is extended toward **object detection models** such as:

- YOLO

Object detection enables:

- Localization of crack regions
- Spatial awareness of defect position
- Improved monitoring accuracy

The long term goal is to integrate the detection pipeline with **UTM control logic**, enabling automatic interruption of loading when crack initiation is detected.

---

# Expected Outcome

The expected outcome of this project is a **validated computer vision pipeline** capable of detecting early stage crack initiation from image data captured during UTM testing.

The system demonstrates the feasibility of automated damage monitoring in mechanical testing systems and provides a foundation for:

- Real time failure detection
- Intelligent experimental control
- Predictive failure analysis
- Smart mechanical testing infrastructure