#!/usr/bin/env python3
"""Convert SSDD RBox annotations to YOLO OBB format.

Reads SSDD XML annotations and writes YOLO-style OBB labels.
Two output modes are available:
  1. 8-corner normalized format: class x1 y1 x2 y2 x3 y3 x4 y4
     (direct from SSDD corners, no angle ambiguity)
  2. 5-value xywhr format: class cx cy w h angle
     (computed from corners via cv2.minAreaRect)

Usage:
    python tools/convert_ssdd_to_yolo_obb.py --mode corners
    python tools/convert_ssdd_to_yolo_obb.py --mode xywhr  (default)
"""

import argparse
import math
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def extract_annotations(xml_path: str):
    """Extract all objects from an SSDD XML annotation file.

    Returns:
        tuple: (filename, image_w, image_h, list_of_annotations)
        Each annotation is a dict with keys:
            cls: class name (usually 'ship')
            cx, cy, w, h, theta: parametric box values
            corners: list of (x, y) tuples for the 4 corners
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename", Path(xml_path).stem)
    size = root.find("size")
    image_w = int(size.findtext("width", "0"))
    image_h = int(size.findtext("height", "0"))

    annotations = []
    for obj in root.findall("object"):
        box = obj.find("rotated_bndbox")
        if box is None:
            continue

        cls_name = obj.findtext("name", "ship")
        cx = float(box.findtext("rotated_bbox_cx", "0"))
        cy = float(box.findtext("rotated_bbox_cy", "0"))
        w = float(box.findtext("rotated_bbox_w", "0"))
        h = float(box.findtext("rotated_bbox_h", "0"))
        theta = float(box.findtext("rotated_bbox_theta", "0"))

        corners = []
        for i in range(1, 5):
            x = float(box.findtext(f"x{i}", "0"))
            y = float(box.findtext(f"y{i}", "0"))
            corners.append((x, y))

        annotations.append({
            "cls": cls_name,
            "cx": cx, "cy": cy, "w": w, "h": h, "theta": theta,
            "corners": corners,
        })

    return filename, image_w, image_h, annotations


def corners_to_yolo_corners(corners, img_w: int, img_h: int) -> str:
    """Convert 4 corners to YOLO OBB 8-value format.

    Each coordinate is normalized to [0, 1] by image dimensions.
    Format: x1 y1 x2 y2 x3 y3 x4 y4
    """
    coords = []
    for x, y in corners:
        coords.append(f"{x / img_w:.6g}")
        coords.append(f"{y / img_h:.6g}")
    return " ".join(coords)


def corners_to_yolo_xywhr(corners, img_w: int, img_h: int) -> str:
    """Convert 4 corners to YOLO OBB 5-value format (cx cy w h angle).

    Normalizes corners to [0, 1] first, then computes minAreaRect in
    normalized space. This prevents dimension-aspect-ratio confusion when
    the box's longer side maps to the image's shorter axis.

    Angle is converted to Ultralytics' internal range [-pi/4, 3pi/4).
    """
    pts = np.array(corners, dtype=np.float32)
    pts[:, 0] /= img_w
    pts[:, 1] /= img_h
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(pts)

    angle_rad = math.radians(angle_deg)
    if w < h:
        w, h = h, w
        angle_rad += math.pi / 2

    while angle_rad >= 3 * math.pi / 4:
        angle_rad -= math.pi
    while angle_rad < -math.pi / 4:
        angle_rad += math.pi

    return f"{cx:.6g} {cy:.6g} {w:.6g} {h:.6g} {angle_rad:.6g}"


def verify_angle_convention(xml_dir: str, max_files: int = 100):
    """Verify SSDD angle convention against OpenCV minAreaRect.

    Tests four reconstruction methods:
    1. Standard rotation (no swap, no negation)
    2. Swap w/h before rotation
    3. Negate theta before rotation
    4. Swap w/h and negate theta

    Returns the best matching convention.
    """
    xml_paths = sorted(Path(xml_dir).glob("*.xml"))
    if max_files > 0:
        xml_paths = xml_paths[:max_files]

    configs = {
        "standard": (False, False),
        "swap_wh": (True, False),
        "negate_theta": (False, True),
        "swap_and_negate": (True, True),
    }
    errors = {k: [] for k in configs}

    for xml_path in xml_paths:
        _, _, _, annotations = extract_annotations(str(xml_path))
        for ann in annotations:
            cx, cy, w, h, theta = ann["cx"], ann["cy"], ann["w"], ann["h"], ann["theta"]
            gt_corners = ann["corners"]

            for label, (swap, negate) in configs.items():
                predicted = _reconstruct_corners(cx, cy, w, h, theta, swap, negate)
                err = _max_corner_error(predicted, gt_corners)
                errors[label].append(err)

    print("\n=== Angle Convention Verification ===")
    best_label = None
    best_median = float("inf")
    for label in configs:
        errs = errors[label]
        if not errs:
            continue
        med = statistics.median(errs)
        avg = statistics.mean(errs)
        mx = max(errs)
        p1px = sum(1 for e in errs if e <= 1.0) / len(errs) * 100
        p2px = sum(1 for e in errs if e <= 2.0) / len(errs) * 100
        status = "MATCH" if med < 1.0 else ("CLOSE" if med < 2.0 else "MISMATCH")
        print(
            f"  {label:>18}: median={med:.2f}px, mean={avg:.2f}px, "
            f"max={mx:.2f}px, <=1px={p1px:.0f}%, <=2px={p2px:.0f}%  [{status}]"
        )
        if med < best_median:
            best_median = med
            best_label = label

    print(f"\nBest convention: {best_label} (median error: {best_median:.2f} px)")
    return best_label


def _reconstruct_corners(cx, cy, w, h, theta_deg, swap=False, negate=False):
    """Reconstruct 4 corners from parametric OBB."""
    if swap:
        w, h = h, w
    if negate:
        theta_deg = -theta_deg

    angle = math.radians(theta_deg)
    cs = math.cos(angle)
    sn = math.sin(angle)
    hw, hh = w / 2.0, h / 2.0

    rel = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [(cx + (dx * cs - dy * sn), cy + (dx * sn + dy * cs)) for dx, dy in rel]


def _max_corner_error(pred, gt):
    """Max L2 corner distance, testing all cyclic permutations and reversal."""
    best = float("inf")
    # Same direction
    for shift in range(4):
        mx = max(
            math.hypot(pred[i][0] - gt[(i + shift) % 4][0],
                       pred[i][1] - gt[(i + shift) % 4][1])
            for i in range(4)
        )
        if mx < best:
            best = mx

    # Reversed direction
    gt_rev = list(reversed(gt))
    for shift in range(4):
        mx = max(
            math.hypot(pred[i][0] - gt_rev[(i + shift) % 4][0],
                       pred[i][1] - gt_rev[(i + shift) % 4][1])
            for i in range(4)
        )
        if mx < best:
            best = mx
    return best


def convert_all(xml_dir: str, output_dir: str, class_map: dict[str, int],
                mode: str = "xywhr", visualize: bool = False,
                image_dir: str = "", output_dir_vis: str = "", max_annotations: int = 0):
    """Convert all SSDD XML annotations to YOLO OBB format."""
    xml_paths = sorted(Path(xml_dir).glob("*.xml"))
    if max_annotations > 0:
        xml_paths = xml_paths[:max_annotations]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    total_instances = 0
    total_files = 0
    skipped_files = []

    if visualize:
        Path(output_dir_vis).mkdir(parents=True, exist_ok=True)
        vis_count = 0
        max_vis = 8

    for xml_path in xml_paths:
        filename, img_w, img_h, annotations = extract_annotations(str(xml_path))

        if not annotations:
            skipped_files.append(xml_path.name)
            continue

        img_name = Path(filename).stem
        label_path = Path(output_dir) / f"{img_name}.txt"

        label_lines = []
        for ann in annotations:
            cls_id = class_map.get(ann["cls"], 0)
            if mode == "corners":
                box_str = corners_to_yolo_corners(ann["corners"], img_w, img_h)
            else:
                box_str = corners_to_yolo_xywhr(ann["corners"], img_w, img_h)
            label_lines.append(f"{cls_id} {box_str}")
            total_instances += 1

        with open(label_path, "w") as f:
            f.write("\n".join(label_lines) + "\n")

        total_files += 1

        if visualize and vis_count < max_vis:
            img_path = _find_image(image_dir, img_name)
            if img_path:
                _draw_overlay(img_path, output_dir_vis, annotations, vis_count)
                vis_count += 1

    print(f"Converted {total_files} files ({total_instances} instances)")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files with no annotations")
        print(f"  Examples: {', '.join(skipped_files[:5])}")


def _find_image(image_dir: str, img_name: str):
    """Locate the image file for a given annotation name."""
    if not image_dir:
        return None

    for ext in (".jpg", ".png", ".jpeg"):
        path = Path(image_dir) / f"{img_name}{ext}"
        if path.is_file():
            return path
    return None


def _draw_overlay(img_path, output_dir: str, annotations: list, idx: int):
    """Draw ground truth corner boxes on an image for verification."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for ann in annotations:
        gt_corners = [(int(round(x)), int(round(y))) for x, y in ann["corners"]]
        draw.polygon(gt_corners, outline=(0, 255, 0), width=2)

        cx, cy = int(round(ann["cx"])), int(round(ann["cy"]))
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 255, 0))

    out_path = Path(output_dir) / f"verify_{img_path.name}"
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert SSDD RBox annotations to YOLO OBB format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xml-dir", default="datasets/SSDD/annotations",
                        help="Directory containing SSDD XML annotation files")
    parser.add_argument("--image-dir", default="datasets/SSDD/images",
                        help="Directory containing source images")
    parser.add_argument("--output-dir", default="datasets/SSDD/labels",
                        help="Output directory for YOLO OBB label files")
    parser.add_argument("--output-dir-vis", default="outputs/angle_verify",
                        help="Output directory for visualization overlays")
    parser.add_argument("--mode", default="xywhr", choices=["corners", "xywhr"],
                        help="YOLO OBB label format: 'corners' (8 values) or 'xywhr' (5 values)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate overlay images to verify conversion")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Process at most this many XML files (0 = all)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify angle convention, do not convert")
    args = parser.parse_args()

    class_map = {"ship": 0}

    if args.verify_only:
        verify_angle_convention(args.xml_dir, max_files=args.max_files or 100)
        return

    # Always verify first when converting
    if not args.verify_only:
        print("=== Verifying SSDD angle convention ===")
        verify_angle_convention(args.xml_dir, max_files=50)
        print()

    print(f"=== Converting SSDD to YOLO OBB ({args.mode} format) ===")
    convert_all(
        xml_dir=args.xml_dir,
        output_dir=args.output_dir,
        class_map=class_map,
        mode=args.mode,
        visualize=args.visualize,
        image_dir=args.image_dir,
        output_dir_vis=args.output_dir_vis,
        max_annotations=args.max_files,
    )


if __name__ == "__main__":
    main()
