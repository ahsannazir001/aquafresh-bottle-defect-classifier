# Model Card — AquaFresh Bottle Defect Classifier

## 1. Model Overview

- **Project:** AquaFresh Bottle Defect Classifier
- **Model architecture:** ResNet18
- **Framework:** PyTorch
- **Task:** Image classification
- **Checkpoint:** `models/best_tuned_resnet18.pt`
- **Input size:** 224 × 224 pixels
- **Inference API:** FastAPI
- **Deployment platform:** Render

## 2. Intended Purpose

The model classifies bottle images into one of two trained classes to support bottle-condition inspection:

- `sealstable`
- `unsealed`

It is intended as a learning and demonstration project for image classification and API deployment. Its predictions should support, not replace, human quality inspection.

## 3. Dataset

The dataset uses Pascal VOC-style annotations.

- Original annotated images: 53
- Annotated objects: 54
- Training crops: 38
- Validation crops: 8
- Test crops: 8

The original images were split before object crops were generated to reduce the risk of data leakage between training, validation, and test sets.

## 4. Training Configuration

- Architecture: ResNet18
- Input resolution: 224 × 224
- Optimizer: Adam
- Loss function: Cross Entropy Loss
- Selected learning rate: 0.0003
- Selected batch size: 4
- Training epochs: 10
- Random seed: 42
- Compute device: CPU

Training augmentation included random rotation, horizontal flipping, and brightness adjustment.

## 5. Evaluation Results

The selected model achieved the following results on the held-out test split:

- Test images: 8
- Correct predictions: 8
- Measured test accuracy: 100%
- Test loss: 0.004667

The test set is very small. These results describe only this particular split and do not establish expected performance on new or real-world images.

## 6. Limitations

- The model only predicts the two classes represented in its training data.
- It has not been validated for all bottle defects, including cracks, incorrect labels, or empty bottles.
- Performance may vary with lighting, image quality, camera angle, bottle type, and backgrounds.
- Model confidence is not a guarantee of correctness.
- Additional independent test images and real-world validation are required before operational use.

## 7. Ethical and Safety Considerations

- Do not use predictions as the sole basis for rejecting bottles or making consequential quality-control decisions.
- Have a human inspector review uncertain or potentially defective bottles.
- Evaluate performance across different bottle types, lighting conditions, and image sources.
- Avoid claiming detection capabilities that have not been trained and evaluated.

## 8. Inference and Deployment

The model is served through a FastAPI application and deployed on Render.

- Live API: https://aquafresh-bottle-defect-classifier.onrender.com
- Health endpoint: `/health`
- Prediction endpoint: `POST /predict`
- Interactive API documentation: `/docs`

The prediction endpoint accepts an image upload and returns the predicted class, confidence, and class probabilities.

## 9. Maintenance Recommendations

Periodically evaluate the model on newly collected, correctly labelled images. Review incorrect predictions, update the dataset where appropriate, retrain and evaluate candidate models, and validate changes before replacing the deployed checkpoint.
