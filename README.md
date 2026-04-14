# Automated Crack Detection in UTM Testing

Computer vision pipeline for binary crack detection (`Damaged` vs `Undamaged`) from UTM test frames, implemented in Google Colab.

## Scope and Goal

Build a reproducible crack-detection workflow that prioritizes safety-first performance (minimizing false negatives) and supports future UTM stop-trigger integration.

## Current Project Status

- Phase 1: Complete
- Phase 2 (ROI A/B): Complete
- Best current operating variant: ROI-enabled pipeline

## Final Outcomes (from notebook outputs)

### Final summary run
- EfficientNetB0 FT
  - Accuracy: 1.0000
  - Precision: 1.0000
  - Recall: 1.0000
  - F1: 1.0000
  - AUC: 1.0000
  - Confusion matrix: TP=55, TN=32, FP=0, FN=0

### Phase 2 A/B (quick run)
- Full-frame: Accuracy 0.9770, FP=2, FN=0
- ROI: Accuracy 1.0000, FP=0, FN=0
- Winner: ROI

## Repository Structure

```
crack-detection/
├── crack_detection.ipynb   # Main end-to-end notebook
├── problem_statement.md    # Finalized project statement and scope
├── research.md             # Final research/analysis summary
└── Dataset/
    ├── Damaged/
    └── Undamaged/
```

## Notebook Pipeline

`crack_detection.ipynb` includes:
1. Data mounting + checks
2. Temporal split (train/val/test)
3. EDA
4. `tf.data` pipeline with optional ROI crop
5. Baseline CNN
6. EfficientNetB0 two-stage training
7. Threshold optimization
8. Test-set evaluation
9. Grad-CAM
10. Model export + reload check
11. Final summary
12. Phase 2 A/B experiment (Full-frame vs ROI)

## Dataset Notes

- Total images: 568
- Damaged: 359
- Undamaged: 209
- Sequential frame data from UTM video
- Temporal split is used to reduce leakage risk

## Reproducibility Notes

- ROI controlled by `PHASE2_USE_ROI` and `ROI_OFFSETS`
- A/B experiment cell runs both variants with shared logic
- Threshold is calibrated on validation set for each run

## Remaining Constraints

Current results are excellent for this dataset split, but external generalization must still be validated on new UTM sessions before deployment claims.

## Next Practical Step (Optional Future Work)

- Run the same ROI pipeline on an unseen UTM session and report the same metrics (FN, FP, Recall, F1, AUC).
