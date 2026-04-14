# Automated Crack Detection in UTM Testing using Computer Vision

## Scope and Goal

In Universal Testing Machine (UTM) experiments, identifying crack initiation at the right moment is critical for safety, data quality, and preventing complete specimen fracture. Manual visual monitoring is slow, subjective, and hard to scale.

This project addresses that gap by building an automated vision pipeline that classifies each frame as:
- Damaged
- Undamaged

The system is designed as a deployment-ready foundation for automated intervention (stop trigger) in future real-time UTM workflows.

## Core Objective

Build and validate a robust crack-detection model that prioritizes:
1. Zero or near-zero missed cracks (false negatives)
2. High reliability under realistic UTM imaging conditions
3. Reproducible workflow from data loading to model export

## What Was Implemented

### Phase 1 (Completed)
- Temporal-aware train/val/test split (to avoid leakage from sequential frames)
- Baseline CNN for reference
- EfficientNetB0 with two-stage transfer learning:
  - Phase 1: frozen backbone
  - Phase 2: fine-tuning top layers
- Threshold optimization on validation set
- Full test evaluation (classification report, confusion matrix, ROC)
- Grad-CAM visual explanation
- SavedModel + H5 + metadata export

### Phase 2 (Completed)
- ROI-aware preprocessing integrated into `tf.data` pipeline
- Controlled A/B experiment:
  - A: Full-frame
  - B: ROI enabled
- Safety-first ranking based on FN, Recall, F1, AUC

## Final Outcomes (from notebook outputs)

### Primary model performance (final summary)
- EfficientNetB0 FT:
  - Accuracy: 1.0000
  - Precision: 1.0000
  - Recall: 1.0000
  - F1: 1.0000
  - AUC: 1.0000
  - Confusion matrix: TP=55, TN=32, FP=0, FN=0

### Phase 2 A/B result (quick run)
- Full-frame:
  - Accuracy: 0.9770, FP=2, FN=0
- ROI:
  - Accuracy: 1.0000, FP=0, FN=0
- Winner: ROI

## Current Project Status

The project objective for this dataset has been achieved:
- No missed cracks on test set
- ROI variant selected as best operating mode
- End-to-end pipeline documented and reproducible

## Remaining Constraints

Even with strong internal results, the dataset is from a limited source. Before real-world deployment claims, external validation is required on unseen UTM sessions (different specimen/material/lighting setups).

## Next Practical Step (Optional Future Work)

1. Run the same pipeline on at least one new UTM session.
2. Keep ROI as default path.
3. Freeze threshold + model metadata only after external verification.
4. Integrate inference output with a controlled stop-signal layer.
