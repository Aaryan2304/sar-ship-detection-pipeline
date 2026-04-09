#!/usr/bin/env python3
"""Per-chip percentile stretch normalization for SAR imagery.

Applies independent 1st-99th percentile stretch to each chip, mapping
the stretched range to [0, 255]. This normalizes ship-to-background
contrast across tiles that may have different incidence angles and
ocean brightness levels.

Variant A (baseline): raw pixels / 255 (Ultralytics default)
Variant B (this script): per-chip percentile stretch → train on normalized images

Usage:
    python tools/normalize_chips.py                              # Default: SSDD dataset
    python tools/normalize_chips.py --input datasets/SSDD --output datasets/SSDD_norm
    python tools/normalize_chips.py --p-lo 2 --p-hi 98          # Custom percentiles
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Per-chip percentile stretch for SAR imagery")
    parser.add_argument("--input", type=str, default="datasets/SSDD",
                        help="Input dataset directory (contains images/ and labels/)")
    parser.add_argument("--output", type=str, default="datasets/SSDD_norm",
                        help="Output directory for normalized images")
    parser.add_argument("--p-lo", type=float, default=1.0,
                        help="Lower percentile for stretch (default: 1.0)")
    parser.add_argument("--p-hi", type=float, default=99.0,
                        help="Upper percentile for stretch (default: 99.0)")
    parser.add_argument("--preview", type=int, default=5,
                        help="Number of before/after preview images to generate")
    return parser.parse_args()


def percentile_stretch(img, p_lo=1.0, p_hi=99.0):
    """Apply per-chip percentile stretch to a SAR image.

    Args:
        img: BGR or grayscale uint8 image.
        p_lo: Lower percentile (default 1st).
        p_hi: Upper percentile (default 99th).

    Returns:
        Stretched uint8 image (same shape as input).
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Non-zero pixels (ignore black background/padding)
    nz = gray[gray > 0]
    if len(nz) < 10:
        # Not enough data to stretch — return as-is
        return img

    lo = np.percentile(nz, p_lo)
    hi = np.percentile(nz, p_hi)

    if hi <= lo:
        return img

    # Stretch: map [lo, hi] → [0, 255]
    stretched = np.clip((gray.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255)
    stretched = stretched.astype(np.uint8)

    # Zero pixels stay zero
    stretched[gray == 0] = 0

    # Convert back to 3-channel if input was 3-channel
    if len(img.shape) == 3:
        stretched = cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)

    return stretched


def make_preview(original, normalized, output_path, fname):
    """Create a side-by-side before/after preview image."""
    h, w = original.shape[:2]
    # Ensure normalized is same size
    norm_resized = cv2.resize(normalized, (w, h)) if normalized.shape[:2] != (h, w) else normalized

    # Side by side
    canvas = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
    canvas[:, :w] = original
    canvas[:, w + 10:] = norm_resized

    # Labels
    cv2.putText(canvas, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, "P-stretch", (w + 20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(str(output_path / fname), canvas)


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    img_input = input_dir / "images"
    lbl_input = input_dir / "labels"
    img_output = output_dir / "images"
    lbl_output = output_dir / "labels"

    img_output.mkdir(parents=True, exist_ok=True)
    lbl_output.mkdir(parents=True, exist_ok=True)

    # Process all images
    image_files = sorted(img_input.glob("*.jpg"))
    print(f"Processing {len(image_files)} images from {img_input}")
    print(f"Percentile stretch: {args.p_lo}th - {args.p_hi}th percentile")
    print(f"Output: {output_dir}")

    stats = {"processed": 0, "skipped": 0, "intensity_shifts": []}

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] Could not read: {img_path.name}")
            stats["skipped"] += 1
            continue

        # Compute stats before
        gray_before = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_before = gray_before[gray_before > 0].mean() if (gray_before > 0).any() else 0

        # Apply stretch
        normalized = percentile_stretch(img, p_lo=args.p_lo, p_hi=args.p_hi)

        # Compute stats after
        gray_after = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
        mean_after = gray_after[gray_after > 0].mean() if (gray_after > 0).any() else 0

        stats["intensity_shifts"].append(mean_after - mean_before)

        # Save normalized image
        cv2.imwrite(str(img_output / img_path.name), normalized)

        # Copy label (unchanged)
        lbl_path = lbl_input / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy2(str(lbl_path), str(lbl_output / lbl_path.name))

        stats["processed"] += 1

    # Copy split files if they exist
    for split_file in ["train.txt", "val.txt"]:
        src = input_dir / split_file
        if src.exists():
            # Update paths from ./images/ to ./images/ (same structure)
            shutil.copy2(str(src), str(output_dir / split_file))

    print(f"\nDone: {stats['processed']} processed, {stats['skipped']} skipped")
    shifts = np.array(stats["intensity_shifts"])
    print(f"Mean intensity shift: {shifts.mean():+.1f} (std={shifts.std():.1f})")
    print(f"Shift range: [{shifts.min():+.1f}, {shifts.max():+.1f}]")

    # Generate preview images
    if args.preview > 0:
        preview_dir = output_dir / "preview"
        preview_dir.mkdir(exist_ok=True)
        sample_files = sorted(img_input.glob("*.jpg"))[:args.preview]
        for img_path in sample_files:
            original = cv2.imread(str(img_path))
            normalized = cv2.imread(str(img_output / img_path.name))
            if original is not None and normalized is not None:
                make_preview(original, normalized, preview_dir, img_path.name)
        print(f"Previews saved to: {preview_dir}")

    # Create dataset.yaml
    yaml_content = f"""# SSDD with per-chip percentile stretch normalization
# Variant B of the normalization ablation (Phase 3.3)
# Stretch: {args.p_lo}th - {args.p_hi}th percentile of non-zero pixels → [0, 255]

path: {output_dir.resolve()}
train: train.txt
val: val.txt
test: val.txt

names:
  0: ship
"""
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"Dataset config: {yaml_path}")


if __name__ == "__main__":
    main()
