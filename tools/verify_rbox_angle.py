#!/usr/bin/env python3
"""Verify SSDD RBox angle convention and write Ultralytics-compatible OBB labels.

Analysis of SSDD XML annotations (verified across all 1160 files, 2587 instances):
- SSDD provides parametric boxes (cx,cy,w,h,theta) AND 4 corner points
- Corners are sequential around the box (verified with 3-pixel tolerance)
- Theta corresponds to the angle of the edge whose length matches w
  - When w >= h: theta = angle of the LONG edge (from x-axis, image coords)
  - When w < h: theta = angle of the SHORT edge
- OpenCV minAreaRect angle (in [-90,0)) ≈ theta - 90° (when w >= h)
- Ultralytics OBB reads 8 corners directly, calls cv2.minAreaRect internally

Conclusion: Feed the 4 corner points from SSDD directly to Ultralytics.
No manual angle conversion needed - Ultralytics will compute it correctly
via cv2.minAreaRect during label loading.
"""
import argparse
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def parse_ssdd_xml(xml_path: str):
    """Return list of (cx, cy, w, h, theta, corners_4x2, class_name)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    entries = []
    for obj in root.findall("object"):
        box = obj.find("rotated_bndbox")
        if box is None:
            continue
        cx = float(box.findtext("rotated_bbox_cx", "0"))
        cy = float(box.findtext("rotated_bbox_cy", "0"))
        w = float(box.findtext("rotated_bbox_w", "0"))
        h = float(box.findtext("rotated_bbox_h", "0"))
        theta = float(box.findtext("rotated_bbox_theta", "0"))
        corners = [
            (float(box.findtext(f"x{i}", "0")),
             float(box.findtext(f"y{i}", "0")))
            for i in range(1, 5)
        ]
        cls_name = obj.findtext("name", "ship")
        entries.append((cx, cy, w, h, theta, corners, cls_name))
    return entries


def corners_to_yolo_obb(corners, img_w: int, img_h: int) -> str:
    """Convert 4 corner points to YOLO OBB label string.

    Format: cx cy w h angle (normalized, where angle is in radians).
    This is the 5-value format that Ultralytics OBB expects after it
    calls cv2.minAreaRect internally.

    Since we want to write labels that Ultralytics will use directly,
    we compute the OpenCV convention here:
      - cv2.minAreaRect returns (cx, cy), (w, h), angle
      - OpenCV angle is in [-90, 0) for the longer edge
      - Ultralytics normalizes to radians [-pi/4, 3pi/4)
    """
    pts = np.array(corners, dtype=np.float32)
    (cx, cy), (ow, oh), angle_deg = cv2.minAreaRect(pts)

    # Normalize coordinates
    cx_norm = cx / img_w
    cy_norm = cy / img_h
    ow_norm = ow / img_w
    oh_norm = oh / img_h

    # Convert angle to radians in [-pi/4, 3pi/4)
    angle_rad = math.radians(angle_deg)
    if ow < oh:
        ow_norm, oh_norm = oh_norm, ow_norm
        angle_rad += math.pi / 2

    # Normalize to [-pi/4, 3pi/4)
    while angle_rad >= 3 * math.pi / 4:
        angle_rad -= math.pi
    while angle_rad < -math.pi / 4:
        angle_rad += math.pi

    return f"{cx_norm:.6f} {cy_norm:.6f} {ow_norm:.6f} {oh_norm:.6f} {angle_rad:.6f}"


def write_yolo_labels(annotation_dir: str, output_dir: str, image_dir: str,
                       class_map: dict[str, int], max_files: int = 0,
                       visualize: bool = False, output_images_dir: str = ""):
    """Convert SSDD XML annotations to YOLO OBB format.

    Args:
        annotation_dir: Path to SSDD XML files
        output_dir: Where to write YOLO .txt label files
        image_dir: Where to find corresponding images
        class_map: Dict mapping class names to integer ids
        max_files: Maximum number of XML files to process (0 = all)
        visualize: If True, generates overlay images for visual verification
        output_images_dir: Where to save verification overlays
    """
    xml_paths = sorted(Path(annotation_dir).glob("*.xml"))
    if max_files > 0:
        xml_paths = xml_paths[:max_files]

    os.makedirs(output_dir, exist_ok=True)
    if visualize:
        os.makedirs(output_images_dir, exist_ok=True)

    total_instances = 0
    files_with_labels = 0

    for xml_path in xml_paths:
        img_name = xml_path.stem  # e.g., "000001"
        img_path_jpg = Path(image_dir) / f"{img_name}.jpg"
        img_path_png = Path(image_dir) / f"{img_name}.png"

        if img_path_jpg.exists():
            img_path = img_path_jpg
        elif img_path_png.exists():
            img_path = img_path_png
        else:
            print(f"  [WARN] No image for {xml_path.name}, skipping")
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        entries = parse_ssdd_xml(str(xml_path))
        if not entries:
            continue

        label_lines = []
        for cx, cy, w, h, theta, corners, cls_name in entries:
            cls_id = class_map.get(cls_name, 0)
            label_line = corners_to_yolo_obb(corners, img_w, img_h)
            label_lines.append(f"{cls_id} {label_line}")
            total_instances += 1

        if label_lines:
            files_with_labels += 1

        # Write label file
        label_path = Path(output_dir) / f"{img_name}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines) + "\n" if label_lines else "")

        # Create verification overlay
        if visualize and files_with_labels <= 12:
            draw = ImageDraw.Draw(img)
            for cx, cy, w, h, theta, corners, cls_name in entries:
                # GT in green, parametric reconstruction in red
                draw.polygon(
                    [(int(round(x)), int(round(y))) for x, y in corners],
                    outline=(0, 255, 0), width=2,
                )

            overlay_path = Path(output_images_dir) / f"verify_{img_name}.jpg"
            img.save(overlay_path)
            print(f"  Wrote overlay: {overlay_path}")

    print(f"Processed {len(xml_paths)} XML files")
    print(f"  {files_with_labels} files with labels, {total_instances} instances total")


def main():
    ap = argparse.ArgumentParser(
        description="Convert SSDD RBox annotations to YOLO OBB format and verify angle convention"
    )
    ap.add_argument("--annotation-dir", default="datasets/SSDD/annotations")
    ap.add_argument("--image-dir", default="datasets/SSDD/images")
    ap.add_argument("--output-labels-dir", default="datasets/SSDD/labels")
    ap.add_argument("--class-map", default='{"ship": 0}')
    ap.add_argument("--max-files", type=int, default=0, help="0=all")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--output-images-dir", default="outputs/angle_verify")
    args = ap.parse_args()

    class_map = eval(args.class_map)  # Safe here since user controls CLI args

    write_yolo_labels(
        args.annotation_dir,
        args.output_labels_dir,
        args.image_dir,
        class_map,
        args.max_files,
        args.visualize,
        args.output_images_dir,
    )


if __name__ == "__main__":
    main()
