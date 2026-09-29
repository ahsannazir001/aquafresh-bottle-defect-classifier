from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "baseline_resnet18.pt"


# ============================================================
# Configuration
# ============================================================

SEED = 42

BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

NUM_WORKERS = 0

DEVICE = torch.device("cpu")


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("AquaFresh ResNet18 Baseline Training")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()

    # --------------------------------------------------------
    # Image transformations
    # --------------------------------------------------------

    weights = ResNet18_Weights.DEFAULT

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    test_dir = DATA_DIR / "test"

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=eval_transform
    )

    print("Dataset:")
    print(f"  Train: {len(train_dataset)} crops")
    print(f"  Val:   {len(val_dataset)} crops")
    print(f"  Test:  {len(test_dataset)} crops")
    print()

    print("Classes:")
    print(train_dataset.class_to_idx)
    print()

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # --------------------------------------------------------
    # Load pretrained ResNet18
    # --------------------------------------------------------

    print("Loading pretrained ResNet18...")

    model = models.resnet18(
        weights=weights
    )

    # Replace original ImageNet classifier.
    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        len(train_dataset.classes)
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_accuracy = 0.0

    print()
    print("=" * 60)
    print("Training")
    print("=" * 60)

    for epoch in range(NUM_EPOCHS):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                predictions = outputs.argmax(
                    dim=1
                )

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_accuracy = (
            val_correct / val_total
        )

        print(
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_to_idx": train_dataset.class_to_idx,
                "classes": train_dataset.classes,
                "seed": SEED,
                "epoch": epoch + 1,
                "val_accuracy": val_accuracy,
            }

            torch.save(
                checkpoint,
                MODEL_PATH
            )

            print(
                f"  Saved best model -> {MODEL_PATH}"
            )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Loading Best Model")
    print("=" * 60)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    model.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            test_correct += (
                predictions == labels
            ).sum().item()

            test_total += labels.size(0)

    test_accuracy = (
        test_correct / test_total
    )

    print()
    print("=" * 60)
    print("FINAL TEST RESULT")
    print("=" * 60)

    print(
        f"Test correct:  {test_correct}/{test_total}"
    )

    print(
        f"Test accuracy: {test_accuracy:.4f}"
    )

    print(
        f"Test accuracy: {test_accuracy * 100:.2f}%"
    )

    print()
    print(
        f"Best model saved at:"
    )

    print(
        MODEL_PATH
    )

    print("=" * 60)


if __name__ == "__main__":
    main()