# Research Report: Automated Crack Detection in UTM Testing
### Final Consolidated Findings (Phase 1 + Phase 2)

## Scope and Goal

This project develops a computer vision pipeline for crack detection in Universal Testing Machine (UTM) experiments using binary image classification:
- Undamaged
- Damaged

Primary optimization target: minimize missed cracks (false negatives), even at the cost of occasional false alarms.

## Dataset Notes

- Total images: 568
- Damaged: 359
- Undamaged: 209
- Resolution: 3840 x 2176 (RGB JPEG)
- Source: sequential frames from UTM testing

Key methodological choice: temporal split to reduce leakage risk from near-duplicate neighboring frames.

## What Was Implemented

Implemented in `crack_detection.ipynb`:
1. Dataset mount and sanity checks
2. Temporal train/val/test split
3. EDA and sample inspection
4. `tf.data` pipeline with caching and augmentation
5. Baseline CNN benchmark
6. EfficientNetB0 transfer learning
   - Phase 1: frozen backbone
   - Phase 2: fine-tune top layers
7. Validation threshold sweep
8. Test evaluation + confusion matrix + ROC
9. Grad-CAM explainability
10. Model export/reload checks
11. Final summary
12. Phase 2 A/B experiment (Full-frame vs ROI)

## Final Outcomes (from notebook outputs)

From final summary outputs:

- EfficientNetB0 FT
  - Accuracy: 1.0000
  - Precision: 1.0000
  - Recall: 1.0000
  - F1: 1.0000
  - AUC: 1.0000
  - TP=55, TN=32, FP=0, FN=0

- Baseline CNN
  - Lower reliability and weaker ranking behavior

Interpretation: transfer learning with threshold tuning substantially outperforms the baseline and achieves zero false negatives on the available test split.

## Phase 2 A/B Result (quick run)

A controlled A/B experiment was run with shared training/evaluation logic.

### Variant A: Full-frame
- Threshold: 0.8000
- Accuracy: 0.9770
- Precision: 0.9649
- Recall: 1.0000
- F1: 0.9821
- AUC: 1.0000
- TP=55, TN=30, FP=2, FN=0

### Variant B: ROI-enabled
- Threshold: 0.2500
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1: 1.0000
- AUC: 1.0000
- TP=55, TN=32, FP=0, FN=0

### A/B Conclusion
ROI variant wins under safety-first ranking (FN first, then Recall/F1/AUC). It preserves zero misses while also removing residual false positives seen in full-frame mode.

## Current Project Status

1. The current project goals for this dataset were met.
2. ROI preprocessing is now the preferred default mode.
3. EfficientNetB0 + temporal split + threshold calibration is the validated stack.
4. The workflow is reproducible end-to-end and exportable to deployment formats.

## Remaining Constraints

Despite near-perfect internal metrics, the following constraints remain:
- Data source is limited in diversity (single workflow family)
- Sequential frame similarity can still inflate confidence if external tests are absent
- Performance on unseen UTM sessions/materials is not yet formally established

## Next Practical Step (Optional Future Work)

Project can be closed as complete for current scope.

Optional future extension:
1. External validation on new UTM sessions
2. Keep ROI path as deployment default
3. Final threshold freeze after external verification
4. Integrate inference output into a controlled stop-trigger interface

## Final Statement

This project successfully progressed from an initial unstable baseline to a robust transfer-learning solution, then further improved via Phase 2 ROI A/B experimentation. The final selected configuration achieved zero missed cracks and zero false alarms on the available test split, with full documentation and reproducible implementation.
