import sys
from pathlib import Path

import requests


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Test image
IMAGE_PATH = PROJECT_ROOT / "data" / "processed" / "test"

# API endpoint
API_URL = "http://127.0.0.1:8000/predict"


def find_test_image():
    """Find the first JPG/PNG image in the processed test dataset."""
    for extension in ("*.jpg", "*.jpeg", "*.png"):
        images = list(IMAGE_PATH.rglob(extension))
        if images:
            return images[0]

    raise FileNotFoundError(
        f"No test image found in: {IMAGE_PATH}"
    )


def test_predict_endpoint():
    image_path = find_test_image()

    print(f"Testing image: {image_path}")

    with open(image_path, "rb") as image_file:
        response = requests.post(
            API_URL,
            files={
                "file": (
                    image_path.name,
                    image_file,
                    "image/jpeg",
                )
            },
            timeout=30,
        )

    print(f"HTTP status: {response.status_code}")
    print(f"Response: {response.json()}")

    # API should return HTTP 200
    assert response.status_code == 200

    result = response.json()

    # Required response fields
    assert "filename" in result
    assert "prediction" in result
    assert "confidence" in result
    assert "probabilities" in result

    # Prediction must be one of our two classes
    assert result["prediction"] in {
        "sealstable",
        "unsealed",
    }

    # Confidence must be between 0 and 1
    assert 0 <= result["confidence"] <= 1

    print("API test PASSED.")


if __name__ == "__main__":
    try:
        test_predict_endpoint()
    except Exception as exc:
        print(f"API test FAILED: {exc}")
        sys.exit(1)