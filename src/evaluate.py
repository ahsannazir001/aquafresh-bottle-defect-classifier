from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_resnet18.pt"

BATCH_SIZE = 8
DEVICE = torch.device("cpu")


def main():
    print("=" * 60)
    print("AquaFresh ResNet18 Baseline Evaluation")
    print("=" * 60)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total else 0.0

    print()
    print("Classes:", classes)
    print("Test images:", total)
    print(f"Correct predictions: {correct}/{total}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print()
    print(f"Model: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
