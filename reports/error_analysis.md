# AquaFresh Bottle Defect Classifier

## Task 3 - Error Analysis and Reporting

This report evaluates the best tuned ResNet18 model on the held-out test set and analyzes classification errors using a confusion matrix and misclassified sample review.

## 1. Model Information

- Model: ResNet18
- Checkpoint: `best_tuned_resnet18.pt`
- Learning rate: `0.0003`
- Batch size: `4`
- Best epoch: `10`
- Classes: `sealstable, unsealed`

## 2. Test Set Results

- Total test images: **8**
- Correct predictions: **8**
- Incorrect predictions: **0**
- Test accuracy: **100.00%**

## 3. Confusion Matrix

The confusion matrix is saved as `confusion_matrix.png` and `confusion_matrix.csv`.

| Actual / Predicted | sealstable | unsealed |
|---|---|---|
| sealstable | 5 | 0 |
| unsealed | 0 | 3 |

## 4. Classification Report

```text
              precision    recall  f1-score   support

  sealstable       1.00      1.00      1.00         5
    unsealed       1.00      1.00      1.00         3

    accuracy                           1.00         8
   macro avg       1.00      1.00      1.00         8
weighted avg       1.00      1.00      1.00         8
```

## 5. Misclassified Samples

**No misclassified samples were found in the current test set.**

Therefore, there are no misclassified sample images to display for this test run.

## 6. Findings

The current test run produced zero classification errors. Both bottle classes were correctly classified on all available test crops.

The confusion matrix therefore contains only diagonal values and no false-positive or false-negative predictions.

## 7. Recommendations

1. Expand the independent test set because the current test set is small.
2. Collect more bottle images under different lighting conditions, camera angles, backgrounds, and bottle orientations.
3. Maintain balanced examples for both `sealstable` and `unsealed` classes.
4. If future misclassifications appear, review those samples and add targeted training data or augmentation for the problematic cases.
5. Re-run this error-analysis script whenever the model or test dataset changes.

## 8. Limitations

The current test set contains only a small number of image crops. A 100% test accuracy on this small set should not be interpreted as proof of robust real-world generalization. A larger independent test dataset is recommended for stronger evaluation.
