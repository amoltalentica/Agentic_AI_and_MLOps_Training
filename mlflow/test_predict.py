#!/usr/bin/env python3
"""
test_predict.py — End-to-end test for the CNN Kidney Disease Classifier predict API.

Two modes:
  1. Synthetic CT image  : python test_predict.py
  2. Real image file     : python test_predict.py --image /path/to/image.jpg
  3. Save test image only: python test_predict.py --save-only

The script generates a grayscale kidney CT-like image, encodes it as base64
and POSTs it to http://localhost:8080/predict.

Requirements (already in requirements.txt): numpy, Pillow -> pip install pillow numpy requests
"""

import argparse
import base64
import io
import json
import sys

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter

API_URL = "http://localhost:8080/predict"
IMAGE_SIZE = (224, 224)


# ── Synthetic kidney CT–like image generator ─────────────────────────────────

def _generate_ct_image(add_tumor: bool = False) -> Image.Image:
    """Create a grayscale image that broadly mimics a kidney CT slice."""
    rng = np.random.default_rng(42)
    # Base: noisy grey background (tissue)
    arr = rng.integers(30, 60, size=IMAGE_SIZE, dtype=np.uint8)

    img = Image.fromarray(arr, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)

    # Outer body contour (dark oval)
    draw.ellipse([10, 10, 214, 214], fill=(20, 20, 20))

    # Kidney shape (bright bean)
    draw.ellipse([55, 70, 170, 155], fill=(160, 130, 110))
    draw.ellipse([80, 85, 145, 140], fill=(80, 60, 55))   # renal pelvis

    if add_tumor:
        # Add a distinctive bright mass to one pole
        draw.ellipse([130, 75, 168, 110], fill=(220, 200, 180))
        draw.ellipse([138, 82, 162, 105], fill=(240, 230, 210))

    # Blur slightly to look more like a real scan
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    return img


# ── Helpers ──────────────────────────────────────────────────────────────────

def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB").resize(IMAGE_SIZE)


def call_predict(b64: str) -> dict:
    resp = requests.post(
        API_URL,
        json={"image": b64},
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test the /predict endpoint")
    parser.add_argument("--image", help="Path to a real CT scan image (JPEG/PNG)")
    parser.add_argument(
        "--tumor", action="store_true", default=False,
        help="Add synthetic tumor to the generated test image"
    )
    parser.add_argument(
        "--save-only", action="store_true",
        help="Just save the test images to disk without calling the API"
    )
    args = parser.parse_args()

    # ── Build test image ──
    if args.image:
        print(f"Loading image: {args.image}")
        img = load_image(args.image)
        label = "custom"
    else:
        print(f"Generating synthetic CT image (tumor={'yes' if args.tumor else 'no'}) ...")
        img = _generate_ct_image(add_tumor=args.tumor)
        label = "tumor_sample" if args.tumor else "normal_sample"

    # Always save PNG so user can visually inspect what is being sent
    out_png = f"test_image_{label}.png"
    img.save(out_png)
    print(f"Test image saved → {out_png}  (open this in your browser/image viewer)")

    if args.save_only:
        # Also save both synthetic variants for convenience
        for tumor in (False, True):
            name = "tumor_sample" if tumor else "normal_sample"
            _generate_ct_image(add_tumor=tumor).save(f"test_image_{name}.png")
            print(f"  Saved: test_image_{name}.png")
        print("\nDone. Upload either file via the Upload button at http://localhost:8080")
        return

    # ── Encode and POST ──
    b64 = image_to_base64(img)
    print(f"\nPOSTing to {API_URL} ...")
    try:
        result = call_predict(b64)
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot reach the API.")
        print("  Make sure the tunnel is running:  ./k8s/tunnel.sh")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\nHTTP ERROR {e.response.status_code}: {e.response.text}")
        sys.exit(1)

    print("\n" + "=" * 40)
    print("  PREDICTION RESULT")
    print("=" * 40)
    print(json.dumps(result, indent=2))
    print("=" * 40)
    print(f"\nExpected values: 'Tumor' or 'Normal'")


if __name__ == "__main__":
    main()
