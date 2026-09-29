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