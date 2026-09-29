from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_FILE = PROCESSED_DIR / "splits.json"

SPLITS = ["train", "val", "test"]
CLASSES = ["sealstable", "unsealed"]


def main():
    print("=" * 60)
    print("Processed Dataset Verification")
    print("=" * 60)

    with open(SPLITS_FILE, "r", encoding="utf-8") as file:
        split_data = json.load(file)

    print("\nOriginal-image split:")
    print("-" * 60)

    for split in SPLITS:
        records = split_data["splits"][split]
        print(f"{split:5}: {len(records)} original images")

    print("\nGenerated crop counts:")
    print("-" * 60)

    total_crops = 0

    for split in SPLITS:

        split_total = 0

        print(f"\n{split.upper()}")

        for class_name in CLASSES:

            class_dir = (
                PROCESSED_DIR
                / split
                / class_name
            )

            images = list(
                class_dir.glob("*.jpg")
            )

            count = len(images)
            split_total += count

            print(
                f"  {class_name:12}: {count}"
            )

        print(
            f"  Total crops : {split_total}"
        )

        total_crops += split_total

    print()
    print(f"Total generated crops: {total_crops}")

    print()
    print("=" * 60)
    print("Verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()