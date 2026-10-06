from pathlib import Path
from collections import Counter
import csv
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# 1. Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "best_tuned_resnet18.pt"

REPORT_DIR = ROOT / "reports"
MISCLASSIFIED_DIR = REPORT_DIR / "misclassified_samples"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

if MISCLASSIFIED_DIR.exists():
    shutil.rmtree(MISCLASSIFIED_DIR)

MISCLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Configuration
# ============================================================

DEVICE = torch.device("cpu")
BATCH_SIZE = 4

IMAGE_SIZE = 224

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


# ============================================================
# 3. Test transformation
# ============================================================

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=NORMALIZE_MEAN,
        std=NORMALIZE_STD
    ),
])


# ============================================================
# 4. Load test dataset
# ============================================================

test_dir = DATA_DIR / "test"

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=eval_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

classes = test_dataset.classes

print("=" * 60)
print("TASK 3 - ERROR ANALYSIS")
print("=" * 60)

print(f"Test directory: {test_dir}")
print(f"Classes: {classes}")
print(f"Test images: {len(test_dataset)}")


# ============================================================
# 5. Load best tuned model
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

checkpoint_classes = checkpoint["classes"]

if classes != checkpoint_classes:
    raise ValueError(
        f"Dataset classes {classes} do not match "
        f"model classes {checkpoint_classes}"
    )


# ============================================================
# 6. Build ResNet18 architecture
# ============================================================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


print(f"Model: ResNet18")
print(f"Learning rate: {checkpoint['learning_rate']}")
print(f"Batch size: {checkpoint['batch_size']}")
print(f"Best epoch: {checkpoint['epoch']}")


# ============================================================
# 7. Run predictions
# ============================================================

all_true_labels = []
all_predicted_labels = []
prediction_records = []

with torch.no_grad():

    sample_index = 0

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        for i in range(len(labels)):

            true_index = labels[i].item()
            predicted_index = predictions[i].item()

            image_path = test_dataset.samples[
                sample_index
            ][0]

            true_label = classes[true_index]
            predicted_label = classes[predicted_index]

            all_true_labels.append(true_index)
            all_predicted_labels.append(predicted_index)

            prediction_records.append({
                "image": image_path,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": true_label == predicted_label
            })

            sample_index += 1


# ============================================================
# 8. Calculate basic results
# ============================================================

total_images = len(all_true_labels)

correct_predictions = sum(
    true == pred
    for true, pred in zip(
        all_true_labels,
        all_predicted_labels
    )
)

incorrect_predictions = (
    total_images - correct_predictions
)

accuracy = (
    correct_predictions / total_images
    if total_images > 0
    else 0
)

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Total images: {total_images}")
print(f"Correct predictions: {correct_predictions}")
print(f"Incorrect predictions: {incorrect_predictions}")
print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")


# ============================================================
# 9. Confusion matrix
# ============================================================

cm = confusion_matrix(
    all_true_labels,
    all_predicted_labels,
    labels=list(range(len(classes)))
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 10. Save confusion matrix CSV
# ============================================================

cm_csv_path = REPORT_DIR / "confusion_matrix.csv"

with open(
    cm_csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(
        ["Actual / Predicted"] + classes
    )

    for class_name, row in zip(classes, cm):

        writer.writerow(
            [class_name] + row.tolist()
        )


# ============================================================
# 11. Plot confusion matrix
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title("AquaFresh Bottle Defect Classifier - Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(classes))

plt.xticks(
    tick_marks,
    classes,
    rotation=45,
    ha="right"
)

plt.yticks(
    tick_marks,
    classes
)

threshold = cm.max() / 2 if cm.max() > 0 else 0

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
            fontsize=14,
            fontweight="bold"
        )


plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")

plt.tight_layout()

confusion_matrix_path = (
    REPORT_DIR / "confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. Save prediction results
# ============================================================

predictions_csv_path = (
    REPORT_DIR / "test_predictions.csv"
)

with open(
    predictions_csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "image",
        "true_label",
        "predicted_label",
        "correct"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        prediction_records
    )


# ============================================================
# 13. Extract misclassified samples
# ============================================================

misclassified = [
    record
    for record in prediction_records
    if not record["correct"]
]


print("\n" + "=" * 60)
print("MISCLASSIFIED SAMPLES")
print("=" * 60)

print(
    f"Total misclassified samples: "
    f"{len(misclassified)}"
)


for index, record in enumerate(
    misclassified,
    start=1
):

    source_path = Path(record["image"])

    destination_name = (
        f"{index:03d}_"
        f"actual-{record['true_label']}_"
        f"predicted-{record['predicted_label']}_"
        f"{source_path.name}"
    )

    destination_path = (
        MISCLASSIFIED_DIR / destination_name
    )

    shutil.copy2(
        source_path,
        destination_path
    )

    print(
        f"{index}. "
        f"Actual={record['true_label']} | "
        f"Predicted={record['predicted_label']} | "
        f"{source_path.name}"
    )


# ============================================================
# 14. Top misclassified classes
# ============================================================

misclassified_by_class = Counter(
    record["true_label"]
    for record in misclassified
)

top_misclassified = (
    misclassified_by_class
    .most_common(3)
)


# ============================================================
# 15. Classification report
# ============================================================

classification_report_dict = classification_report(
    all_true_labels,
    all_predicted_labels,
    labels=list(range(len(classes))),
    target_names=classes,
    output_dict=True,
    zero_division=0
)

classification_report_text = classification_report(
    all_true_labels,
    all_predicted_labels,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0
)


# ============================================================
# 16. Generate Markdown report
# ============================================================

report_path = REPORT_DIR / "error_analysis.md"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write("# AquaFresh Bottle Defect Classifier\n\n")
    report.write("## Task 3 - Error Analysis and Reporting\n\n")

    report.write(
        "This report evaluates the best tuned ResNet18 model "
        "on the held-out test set and analyzes classification "
        "errors using a confusion matrix and misclassified "
        "sample review.\n\n"
    )

    report.write("## 1. Model Information\n\n")

    report.write(
        f"- Model: ResNet18\n"
        f"- Checkpoint: `{MODEL_PATH.name}`\n"
        f"- Learning rate: `{checkpoint['learning_rate']}`\n"
        f"- Batch size: `{checkpoint['batch_size']}`\n"
        f"- Best epoch: `{checkpoint['epoch']}`\n"
        f"- Classes: `{', '.join(classes)}`\n\n"
    )

    report.write("## 2. Test Set Results\n\n")

    report.write(
        f"- Total test images: **{total_images}**\n"
        f"- Correct predictions: **{correct_predictions}**\n"
        f"- Incorrect predictions: **{incorrect_predictions}**\n"
        f"- Test accuracy: **{accuracy * 100:.2f}%**\n\n"
    )

    report.write("## 3. Confusion Matrix\n\n")

    report.write(
        "The confusion matrix is saved as "
        "`confusion_matrix.png` and "
        "`confusion_matrix.csv`.\n\n"
    )

    report.write("| Actual / Predicted | " +
                 " | ".join(classes) +
                 " |\n")

    report.write("|---|" +
                 "---|" * len(classes) +
                 "\n")

    for class_name, row in zip(classes, cm):

        report.write(
            "| " +
            class_name +
            " | " +
            " | ".join(
                str(value)
                for value in row
            ) +
            " |\n"
        )

    report.write("\n")

    report.write("## 4. Classification Report\n\n")

    report.write("```text\n")
    report.write(classification_report_text)
    report.write("```\n\n")

    report.write("## 5. Misclassified Samples\n\n")

    if not misclassified:

        report.write(
            "**No misclassified samples were found in "
            "the current test set.**\n\n"
        )

        report.write(
            "Therefore, there are no misclassified sample "
            "images to display for this test run.\n\n"
        )

    else:

        report.write(
            f"Total misclassified samples: "
            f"**{len(misclassified)}**\n\n"
        )

        report.write(
            "The misclassified images are available in "
            "`reports/misclassified_samples/`.\n\n"
        )

        report.write(
            "### Top Misclassified Classes\n\n"
        )

        for class_name, count in top_misclassified:

            report.write(
                f"- **{class_name}**: {count} error(s)\n"
            )

        report.write("\n")

        report.write(
            "### Misclassified Examples\n\n"
        )

        for index, record in enumerate(
            misclassified,
            start=1
        ):

            report.write(
                f"{index}. "
                f"Actual: `{record['true_label']}` | "
                f"Predicted: `{record['predicted_label']}` | "
                f"Image: `{Path(record['image']).name}`\n"
            )

        report.write("\n")

    report.write("## 6. Findings\n\n")

    if not misclassified:

        report.write(
            "The current test run produced zero classification "
            "errors. Both bottle classes were correctly "
            "classified on all available test crops.\n\n"
        )

        report.write(
            "The confusion matrix therefore contains only "
            "diagonal values and no false-positive or "
            "false-negative predictions.\n\n"
        )

    else:

        report.write(
            "The error analysis identified "
            f"{len(misclassified)} misclassified sample(s). "
            "These examples should be reviewed to determine "
            "whether the issue is related to image quality, "
            "lighting, orientation, labeling, or insufficient "
            "training examples.\n\n"
        )

    report.write("## 7. Recommendations\n\n")

    report.write(
        "1. Expand the independent test set because the "
        "current test set is small.\n"
    )

    report.write(
        "2. Collect more bottle images under different "
        "lighting conditions, camera angles, backgrounds, "
        "and bottle orientations.\n"
    )

    report.write(
        "3. Maintain balanced examples for both "
        "`sealstable` and `unsealed` classes.\n"
    )

    report.write(
        "4. If future misclassifications appear, review "
        "those samples and add targeted training data or "
        "augmentation for the problematic cases.\n"
    )

    report.write(
        "5. Re-run this error-analysis script whenever the "
        "model or test dataset changes.\n\n"
    )

    report.write("## 8. Limitations\n\n")

    report.write(
        "The current test set contains only a small number "
        "of image crops. A 100% test accuracy on this small "
        "set should not be interpreted as proof of robust "
        "real-world generalization. A larger independent "
        "test dataset is recommended for stronger evaluation.\n"
    )


# ============================================================
# 17. Final output
# ============================================================

print("\n" + "=" * 60)
print("TASK 3 REPORT GENERATED")
print("=" * 60)

print(f"Confusion matrix: {confusion_matrix_path}")
print(f"Confusion matrix CSV: {cm_csv_path}")
print(f"Predictions CSV: {predictions_csv_path}")
print(f"Misclassified samples: {MISCLASSIFIED_DIR}")
print(f"Markdown report: {report_path}")

print("\nTask 3 error analysis completed.")