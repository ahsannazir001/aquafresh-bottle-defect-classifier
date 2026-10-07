from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import models, transforms


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_tuned_resnet18.pt"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CLASS_NAMES = ["sealstable", "unsealed"]
IMAGE_SIZE = 224
DEVICE = torch.device("cpu")


# --------------------------------------------------
# Image preprocessing
# Same preprocessing used during model evaluation
# --------------------------------------------------

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# --------------------------------------------------
# Load trained ResNet18 model
# --------------------------------------------------

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = models.resnet18(weights=None)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()
# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="AquaFresh Bottle Defect Classifier API",
    description=(
        "FastAPI inference service for classifying "
        "AquaFresh bottle defects."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "AquaFresh Bottle Defect Classifier API",
        "status": "running",
        "endpoint": "/predict",
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "best_tuned_resnet18.pt",
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate uploaded file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    try:
        # Read uploaded image
        image_bytes = await file.read()

        # Convert image to RGB
        image = Image.open(
            __import__("io").BytesIO(image_bytes)
        ).convert("RGB")

        # Apply model preprocessing
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)

        # Get predicted class
        confidence, predicted_index = torch.max(
            probabilities,
            dim=1,
        )

        predicted_class = CLASS_NAMES[predicted_index.item()]

        # Return probabilities for both classes
        class_probabilities = {
            CLASS_NAMES[i]: round(
                probabilities[0][i].item(),
                4,
            )
            for i in range(len(CLASS_NAMES))
        }

        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": round(confidence.item(), 4),
            "probabilities": class_probabilities,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process image: {exc}",
        ) from exc