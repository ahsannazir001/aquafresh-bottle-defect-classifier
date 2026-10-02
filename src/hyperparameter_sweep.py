
from pathlib import Path
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights


# ============================================================
# Paths and reproducibility
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 10
NUM_WORKERS = 0
DEVICE = torch.device("cpu")

# Four configurations for the hyperparameter sweep.
CONFIGS = [
    {"lr": 0.0001, "batch_size": 4},
    {"lr": 0.0001, "batch_size": 8},
    {"lr": 0.0003, "batch_size": 4},
    {"lr": 0.0003, "batch_size": 8},
]

RESULTS_CSV = REPORT_DIR / "experiment_results.csv"
HISTORY_CSV = REPORT_DIR / "training_history.csv"
BEST_MODEL_PATH = MODEL_DIR / "best_tuned_resnet18.pt"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(num_classes):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("Evaluation dataset is empty.")

    return total_loss / total, correct / total


def main():
    set_seed(SEED)

    print("=" * 65)
    print("AquaFresh - Hyperparameter Sweep")
    print("=" * 65)
    print("Device:", DEVICE)
    print("Epochs per experiment:", EPOCHS)
    print("Number of configurations:", len(CONFIGS))

    # Augmentation is applied ONLY to training images.
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(
        DATA_DIR / "train", transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        DATA_DIR / "val", transform=eval_transform
    )
    test_dataset = datasets.ImageFolder(
        DATA_DIR / "test", transform=eval_transform
    )

    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Train and validation classes do not match.")
    if train_dataset.classes != test_dataset.classes:
        raise ValueError("Train and test classes do not match.")
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Train, validation, or test dataset is empty.")

    classes = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx

    print("Classes:", class_to_idx)
    print(
        f"Images/crops: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )

    criterion = nn.CrossEntropyLoss()
    experiment_results = []
    epoch_history = []

    # --------------------------------------------------------
    # Run every hyperparameter configuration
    # --------------------------------------------------------

    for config_number, config in enumerate(CONFIGS, start=1):
        lr = config["lr"]
        batch_size = config["batch_size"]

        print("\n" + "=" * 65)
        print(
            f"Experiment {config_number}/{len(CONFIGS)} | "
            f"LR={lr} | Batch size={batch_size}"
        )
        print("=" * 65)

        set_seed(SEED)

        generator = torch.Generator().manual_seed(SEED)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            generator=generator,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        model = build_model(len(classes))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_accuracy = -1.0
        best_val_loss = float("inf")
        best_epoch = 0

        checkpoint_path = MODEL_DIR / (
            f"sweep_lr_{lr}_bs_{batch_size}.pt"
        )

        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_accuracy = correct / total

            val_loss, val_accuracy = evaluate(
                model, val_loader, criterion
            )

            epoch_history.append({
                "experiment": config_number,
                "learning_rate": lr,
                "batch_size": batch_size,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            })

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Train Loss={train_loss:.4f} | "
                f"Train Acc={train_accuracy:.4f} | "
                f"Val Loss={val_loss:.4f} | "
                f"Val Acc={val_accuracy:.4f}"
            )

            # Select the best epoch using validation metrics only.
            if (
                val_accuracy > best_val_accuracy
                or (
                    val_accuracy == best_val_accuracy
                    and val_loss < best_val_loss
                )
            ):
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss
                best_epoch = epoch

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "class_to_idx": class_to_idx,
                    "seed": SEED,
                    "epoch": epoch,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                }, checkpoint_path)

        experiment_results.append({
            "experiment": config_number,
            "learning_rate": lr,
            "batch_size": batch_size,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        })

        print(
            f"Experiment complete: best validation accuracy="
            f"{best_val_accuracy:.4f} at epoch {best_epoch}"
        )

    # --------------------------------------------------------
    # Save experiment metrics to CSV files
    # --------------------------------------------------------

    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(experiment_results[0].keys())
        )
        writer.writeheader()
        writer.writerows(experiment_results)

    with HISTORY_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(epoch_history[0].keys())
        )
        writer.writeheader()
        writer.writerows(epoch_history)

    # --------------------------------------------------------
    # Select the configuration by validation accuracy,
    # breaking ties with validation loss.
    # --------------------------------------------------------

    best_result = sorted(
        experiment_results,
        key=lambda row: (
            -row["best_val_accuracy"],
            row["best_val_loss"],
        ),
    )[0]

    best_checkpoint = torch.load(
        ROOT / best_result["checkpoint"],
        map_location=DEVICE,
        weights_only=True,
    )

    best_model = build_model(len(classes))
    best_model.load_state_dict(best_checkpoint["model_state_dict"])

    # Save a separate copy of the selected model.
    torch.save(best_checkpoint, BEST_MODEL_PATH)

    # Test set is evaluated only after configuration selection.
    test_loader = DataLoader(
        test_dataset,
        batch_size=best_result["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loss, test_accuracy = evaluate(
        best_model, test_loader, criterion
    )

    best_result["test_loss"] = test_loss
    best_result["test_accuracy"] = test_accuracy
    # Save the final test metrics to the experiment results CSV.
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "experiment",
            "learning_rate",
            "batch_size",
            "best_epoch",
            "best_val_accuracy",
            "best_val_loss",
            "checkpoint",
            "test_loss",
            "test_accuracy",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(experiment_results)

    # --------------------------------------------------------
    # Plot training and validation curves for each experiment
    # --------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for result in experiment_results:
        exp_id = result["experiment"]
        rows = [
            row for row in epoch_history
            if row["experiment"] == exp_id
        ]

        label = (
            f"LR={result['learning_rate']}, "
            f"BS={result['batch_size']}"
        )

        epochs = [row["epoch"] for row in rows]

        axes[0].plot(
            epochs, [row["train_loss"] for row in rows],
            marker="o", label=f"{label} train"
        )
        axes[0].plot(
            epochs, [row["val_loss"] for row in rows],
            marker="x", linestyle="--", label=f"{label} val"
        )
        axes[1].plot(
            epochs, [row["val_accuracy"] for row in rows],
            marker="o", label=label
        )

    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7)
    axes[0].grid(True)

    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    fig.tight_layout()
    curves_path = REPORT_DIR / "training_curves.png"
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)

    # Validation accuracy comparison chart.
    labels = [
        f"LR={row['learning_rate']}\nBS={row['batch_size']}"
        for row in experiment_results
    ]
    values = [
        row["best_val_accuracy"] for row in experiment_results
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1.05)
    plt.ylabel("Best validation accuracy")
    plt.title("Hyperparameter Comparison")
    plt.grid(axis="y")
    plt.tight_layout()

    comparison_path = REPORT_DIR / "hyperparameter_comparison.png"
    plt.savefig(comparison_path, dpi=150)
    plt.close()

    # Separate final test result report.
    with (REPORT_DIR / "final_test_result.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write("AquaFresh Task 2 - Selected Model Test Result\n")
        file.write(f"Selected experiment: {best_result['experiment']}\n")
        file.write(f"Learning rate: {best_result['learning_rate']}\n")
        file.write(f"Batch size: {best_result['batch_size']}\n")
        file.write(f"Best epoch: {best_result['best_epoch']}\n")
        file.write(f"Validation accuracy: {best_result['best_val_accuracy']:.4f}\n")
        file.write(f"Test loss: {test_loss:.4f}\n")
        file.write(f"Test accuracy: {test_accuracy:.4f}\n")
        file.write(f"Test accuracy (%): {test_accuracy * 100:.2f}%\n")

    print("\n" + "=" * 65)
    print("HYPERPARAMETER SWEEP COMPLETE")
    print("=" * 65)

    for result in experiment_results:
        print(
            f"Experiment {result['experiment']}: "
            f"LR={result['learning_rate']}, "
            f"BS={result['batch_size']}, "
            f"Best Val Acc={result['best_val_accuracy']:.4f}, "
            f"Best Val Loss={result['best_val_loss']:.4f}"
        )

    print("\nSelected experiment:", best_result["experiment"])
    print("Selected LR:", best_result["learning_rate"])
    print("Selected batch size:", best_result["batch_size"])
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print("Selected model:", BEST_MODEL_PATH)
    print("Results CSV:", RESULTS_CSV)
    print("History CSV:", HISTORY_CSV)
    print("Training curves:", curves_path)
    print("Comparison chart:", comparison_path)
    print("Test report:", REPORT_DIR / "final_test_result.txt")


if __name__ == "__main__":
    main()
