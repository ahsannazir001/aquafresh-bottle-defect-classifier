from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "raw" / "VOC2007" / "Annotations"


def main():
    xml_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))

    object_count_distribution = Counter()
    images_with_multiple_objects = []
    images_with_both_classes = []

    for xml_file in xml_files:
        root = ET.parse(xml_file).getroot()

        objects = root.findall("object")
        labels = []

        for obj in objects:
            name = obj.findtext("name")
            if name:
                labels.append(name.strip())

        object_count_distribution[len(labels)] += 1

        if len(labels) > 1:
            images_with_multiple_objects.append(
                (xml_file.name, labels)
            )

        if len(set(labels)) > 1:
            images_with_both_classes.append(
                (xml_file.name, labels)
            )

    print("=" * 60)
    print("Annotation Structure Inspection")
    print("=" * 60)

    print(f"Total XML files: {len(xml_files)}")
    print()

    print("Objects per image:")
    print("-" * 60)

    for count, image_count in sorted(object_count_distribution.items()):
        print(f"{count} object(s): {image_count} image(s)")

    print()

    print(f"Images with multiple objects: {len(images_with_multiple_objects)}")
    print(f"Images containing both classes: {len(images_with_both_classes)}")

    if images_with_multiple_objects:
        print()
        print("Multiple-object images:")
        for filename, labels in images_with_multiple_objects:
            print(f"  {filename}: {labels}")

    if images_with_both_classes:
        print()
        print("Images containing both classes:")
        for filename, labels in images_with_both_classes:
            print(f"  {filename}: {labels}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()