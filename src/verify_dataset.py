from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "VOC2007"

ANNOTATIONS_DIR = DATASET_DIR / "Annotations"
IMAGES_DIR = DATASET_DIR / "JPEGImages"


def main():
    xml_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))
    image_files = sorted(IMAGES_DIR.glob("*.jpg"))

    print("=" * 60)
    print("AquaFresh Dataset Verification")
    print("=" * 60)

    print(f"Dataset:      {DATASET_DIR}")
    print(f"XML files:    {len(xml_files)}")
    print(f"JPG images:   {len(image_files)}")
    print()

    class_counts = Counter()
    annotation_images = set()
    xml_errors = []

    for xml_file in xml_files:
        try:
            root = ET.parse(xml_file).getroot()

            filename = root.findtext("filename")
            if filename:
                annotation_images.add(Path(filename).stem)

            objects = root.findall("object")

            for obj in objects:
                class_name = obj.findtext("name")

                if class_name:
                    class_counts[class_name.strip()] += 1

        except ET.ParseError as exc:
            xml_errors.append((xml_file.name, str(exc)))

    image_stems = {image.stem for image in image_files}

    missing_images = sorted(annotation_images - image_stems)
    missing_annotations = sorted(image_stems - annotation_images)

    print("Classes found:")
    print("-" * 60)

    for class_name, count in class_counts.most_common():
        print(f"{class_name:30} {count}")

    print()
    print(f"Unique classes: {len(class_counts)}")
    print()

    print("Dataset consistency:")
    print("-" * 60)
    print(f"Images referenced by XML but missing: {len(missing_images)}")
    print(f"Images without XML annotation:       {len(missing_annotations)}")
    print(f"XML parsing errors:                   {len(xml_errors)}")

    print()

    if xml_errors:
        print("XML parsing errors:")
        for filename, error in xml_errors[:10]:
            print(f"  {filename}: {error}")

    if missing_images:
        print("\nMissing image files:")
        for filename in missing_images[:20]:
            print(f"  {filename}")

    if missing_annotations:
        print("\nImages without annotations:")
        for filename in missing_annotations[:20]:
            print(f"  {filename}")

    print()
    print("=" * 60)
    print("Verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()