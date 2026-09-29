from pathlib import Path
import json
import random
import shutil
import xml.etree.ElementTree as ET

from PIL import Image


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "VOC2007"
ANNOTATIONS_DIR = DATASET_DIR / "Annotations"
IMAGES_DIR = DATASET_DIR / "JPEGImages"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_FILE = PROCESSED_DIR / "splits.json"


# ============================================================
# Configuration
# ============================================================

SEED = 42

TRAIN_COUNT = 37
VAL_COUNT = 8
TEST_COUNT = 8


# ============================================================
# Parse Pascal VOC annotation
# ============================================================

def parse_annotation(xml_path):

    root = ET.parse(xml_path).getroot()

    filename = root.findtext("filename")

    if not filename:
        raise ValueError(
            f"Missing filename in {xml_path.name}"
        )

    objects = []

    for obj in root.findall("object"):

        class_name = obj.findtext("name")

        if not class_name:
            raise ValueError(
                f"Missing class name in {xml_path.name}"
            )

        bbox = obj.find("bndbox")

        if bbox is None:
            raise ValueError(
                f"Missing bounding box in {xml_path.name}"
            )

        xmin = int(float(bbox.findtext("xmin")))
        ymin = int(float(bbox.findtext("ymin")))
        xmax = int(float(bbox.findtext("xmax")))
        ymax = int(float(bbox.findtext("ymax")))

        objects.append(
            {
                "class": class_name.strip(),
                "bbox": [xmin, ymin, xmax, ymax],
            }
        )

    if not objects:
        raise ValueError(
            f"No objects found in {xml_path.name}"
        )

    return filename, objects


# ============================================================
# Collect dataset records
# ============================================================

def collect_records():

    records = []

    xml_files = sorted(
        ANNOTATIONS_DIR.glob("*.xml")
    )

    if len(xml_files) != 53:
        print(
            f"WARNING: Expected 53 XML files, "
            f"found {len(xml_files)}."
        )

    for xml_path in xml_files:

        filename, objects = parse_annotation(
            xml_path
        )

        unique_classes = sorted(
            {obj["class"] for obj in objects}
        )

        if len(unique_classes) != 1:
            raise ValueError(
                f"{xml_path.name} contains multiple "
                f"classes: {unique_classes}"
            )

        image_path = IMAGES_DIR / filename

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image: {image_path}"
            )

        records.append(
            {
                "image": filename,
                "image_path": image_path,
                "xml": xml_path.name,
                "class": unique_classes[0],
                "objects": objects,
            }
        )

    return records


# ============================================================
# Exact stratified split: 37 / 8 / 8
# ============================================================

def create_split(records):

    rng = random.Random(SEED)

    class_groups = {}

    for record in records:

        class_groups.setdefault(
            record["class"], []
        ).append(record)

    # We have:
    #
    # sealstable = 30 images
    # unsealed   = 23 images
    #
    # Exact split:
    #
    # TRAIN = 37
    # VAL   = 8
    # TEST  = 8
    #
    # The following allocation keeps the class
    # proportions close to the original dataset.

    target_counts = {
        "sealstable": {
            "train": 21,
            "val": 4,
            "test": 5,
        },
        "unsealed": {
            "train": 16,
            "val": 4,
            "test": 3,
        },
    }

    split_records = {
        "train": [],
        "val": [],
        "test": [],
    }

    for class_name, class_records in sorted(
        class_groups.items()
    ):

        rng.shuffle(class_records)

        expected = target_counts.get(class_name)

        if expected is None:
            raise ValueError(
                f"Unexpected class: {class_name}"
            )

        total_expected = (
            expected["train"]
            + expected["val"]
            + expected["test"]
        )

        if total_expected != len(class_records):
            raise ValueError(
                f"Class {class_name}: expected "
                f"{total_expected} images but found "
                f"{len(class_records)}."
            )

        train_end = expected["train"]

        val_end = (
            train_end
            + expected["val"]
        )

        split_records["train"].extend(
            class_records[:train_end]
        )

        split_records["val"].extend(
            class_records[train_end:val_end]
        )

        split_records["test"].extend(
            class_records[val_end:]
        )

    # Shuffle each final split.
    for split_name in split_records:

        rng.shuffle(
            split_records[split_name]
        )

    # Final safety checks.
    assert len(split_records["train"]) == TRAIN_COUNT
    assert len(split_records["val"]) == VAL_COUNT
    assert len(split_records["test"]) == TEST_COUNT

    return split_records


# ============================================================
# Save split metadata
# ============================================================

def save_splits(split_records):

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "seed": SEED,
        "split_counts": {
            "train": TRAIN_COUNT,
            "val": VAL_COUNT,
            "test": TEST_COUNT,
        },
        "splits": {},
    }

    for split_name, records in split_records.items():

        output["splits"][split_name] = []

        for record in records:

            output["splits"][split_name].append(
                {
                    "image": record["image"],
                    "xml": record["xml"],
                    "class": record["class"],
                }
            )

    with open(
        SPLITS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )


# ============================================================
# Create object crops
# ============================================================

def create_crops(split_records):

    # Remove previously generated split folders.
    for split_name in [
        "train",
        "val",
        "test",
    ]:

        split_dir = PROCESSED_DIR / split_name

        if split_dir.exists():
            shutil.rmtree(split_dir)

    for split_name, records in split_records.items():

        for record in records:

            image_path = record["image_path"]

            with Image.open(image_path) as image:

                image = image.convert("RGB")

                for object_index, obj in enumerate(
                    record["objects"],
                    start=1
                ):

                    class_name = obj["class"]

                    xmin, ymin, xmax, ymax = (
                        obj["bbox"]
                    )

                    # Keep coordinates inside image.
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(image.width, xmax)
                    ymax = min(image.height, ymax)

                    if (
                        xmax <= xmin
                        or ymax <= ymin
                    ):
                        raise ValueError(
                            f"Invalid bounding box "
                            f"in {record['xml']}: "
                            f"{obj['bbox']}"
                        )

                    crop = image.crop(
                        (
                            xmin,
                            ymin,
                            xmax,
                            ymax,
                        )
                    )

                    output_dir = (
                        PROCESSED_DIR
                        / split_name
                        / class_name
                    )

                    output_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    output_name = (
                        f"{image_path.stem}"
                        f"_object{object_index}.jpg"
                    )

                    output_path = (
                        output_dir / output_name
                    )

                    crop.save(
                        output_path,
                        format="JPEG",
                        quality=95
                    )


# ============================================================
# Print final summary
# ============================================================

def print_summary(split_records):

    print()
    print("=" * 60)
    print("FINAL DATASET PREPARATION")
    print("=" * 60)

    print(f"Random seed: {SEED}")
    print()

    total = 0

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        records = split_records[split_name]

        total += len(records)

        class_counts = {}

        for record in records:

            class_name = record["class"]

            class_counts[class_name] = (
                class_counts.get(
                    class_name,
                    0
                )
                + 1
            )

        print(
            f"{split_name.upper():5} images: "
            f"{len(records)}"
        )

        for class_name, count in sorted(
            class_counts.items()
        ):

            print(
                f"      {class_name}: {count}"
            )

    print()
    print(
        f"Total original images: {total}"
    )

    print()
    print(
        f"Splits saved to: {SPLITS_FILE}"
    )

    print(
        f"Crops saved to: "
        f"{PROCESSED_DIR}"
    )

    print()
    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("AquaFresh Dataset Preparation")
    print("=" * 60)

    records = collect_records()

    print(
        f"Found {len(records)} annotated images."
    )

    split_records = create_split(
        records
    )

    save_splits(
        split_records
    )

    create_crops(
        split_records
    )

    print_summary(
        split_records
    )


if __name__ == "__main__":
    main()