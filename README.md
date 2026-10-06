# AquaFresh Bottle Defect Classifier

## Task 1 — Project Setup and ResNet18 Baseline

A baseline image classification system for AquaFresh bottle-condition inspection using transfer learning with ResNet18.

## Dataset

The selected bottle-condition dataset contains Pascal VOC-style annotations with two verified classes:

- `sealstable`
- `unsealed`

The original dataset contains **53 annotated images** and **54 annotated objects**. One image (`IMG_1822`) contains two objects, both belonging to `sealstable`.

For classification, the dataset was prepared by splitting the **original images first** and then generating crops from the annotated bounding boxes. This prevents crops from the same original image from being distributed across different splits.

## Dataset Split

The split is reproducible using random seed `42`.

| Split | Original Images |
|---|---:|
| Train | 37 |
| Validation | 8 |
| Test | 8 |
| **Total** | **53** |

Generated classification crops:

| Split | `sealstable` | `unsealed` | Total Crops |
|---|---:|---:|---:|
| Train | 22 | 16 | 38 |
| Validation | 4 | 4 | 8 |
| Test | 5 | 3 | 8 |
| **Total** | **31** | **23** | **54** |

The training set contains 38 crops because `IMG_1822` contains two `sealstable` objects.

## Baseline Model

Task 1 uses a pretrained **ResNet18** model with the final classification layer adapted for the two dataset classes.

### Training Configuration

- Model: ResNet18
- Framework: PyTorch
- Torchvision: 0.29.0
- Device: CPU
- Random seed: 42
- Batch size: 8
- Epochs: 10
- Learning rate: 0.0001
- Optimizer: Adam
- Loss: Cross Entropy Loss
- Input size: 224 × 224
- Image normalization: ImageNet mean/std

## Training Result

The baseline training completed successfully for 10 epochs.

The best validation accuracy reached:

**100.00%**

## Test Result

The saved baseline model was independently evaluated on the held-out test set.

- Test images: 8
- Correct predictions: 8/8
- Test accuracy: **100.00%**

Because the test set contains only 8 images, this result represents the measured performance on this test split and should not be interpreted as a general real-world accuracy estimate.

## Saved Model

The trained baseline checkpoint is:

```text
models/baseline_resnet18.pt
---

## Task 2 — Data Augmentation and Hyperparameter Tuning

### Data Augmentation

The training pipeline applies:

- Resize to 224 × 224 pixels.
- Random rotation up to 10 degrees.
- Random horizontal flip with probability 0.5.
- Random brightness adjustment using ColorJitter.
- Tensor conversion and ImageNet normalization.

Validation and test images use resizing and normalization without random augmentation.

### Hyperparameter Sweep

Four configurations were evaluated for 10 epochs each.

| Learning Rate | Batch Size | Best Epoch | Validation Accuracy | Validation Loss |
|---:|---:|---:|---:|---:|
| 0.0001 | 4 | 6 | 100.00% | 0.01021 |
| 0.0001 | 8 | 6 | 100.00% | 0.00307 |
| 0.0003 | 4 | 10 | 100.00% | 0.00162 |
| 0.0003 | 8 | 10 | 100.00% | 0.04891 |

Selected configuration: learning rate 0.0003, batch size 4.

### Final Test Results

- Test images: 8
- Correct predictions: 8/8
- Test accuracy: 100.00%
- Test loss: 0.004667

The baseline and tuned models both achieved 100% accuracy on this eight-image test set. Therefore, a 2-percentage-point improvement over the baseline has not been demonstrated. A larger independent test set is needed to evaluate generalization reliably.

### Reports

- `reports/training_history.csv`
- `reports/experiment_results.csv`
- `reports/training_curves.png`
- `reports/hyperparameter_comparison.png`
- `reports/final_test_result.txt`

### Saved Models

- `models/augmented_resnet18.pt`
- `models/best_tuned_resnet18.pt`
- Four hyperparameter-sweep checkpoints in `models/`

### Reproducing Task 2

Run these commands from the project root with the virtual environment activated:

```bash
python src/train_task2.py
python src/hyperparameter_sweep.py

## Task 3 — Error Analysis and Reporting

Task 3 evaluates the best tuned ResNet18 model on the held-out test set and performs error analysis using a confusion matrix, prediction records, and misclassified-sample review.

### Error Analysis Implementation

The error-analysis pipeline is implemented in:

```text
src/error_analysis.py