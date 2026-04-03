#!/usr/bin/env python3
"""
Data augmentation pipeline for SAR ship detection.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data augmentation pipeline for SAR ship detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--train-annotations",
        type=str,
        required=True,
        help="Path to the training COCO annotations file"
    )
    
    parser.add_argument(
        "--train-images-dir",
        type=str,
        required=True,
        help="Directory containing training images"
    )
    
    parser.add_argument(
        "--output-images-dir",
        type=str,
        required=True,
        help="Directory to save augmented images"
    )
    
    parser.add_argument(
        "--output-coco",
        type=str,
        required=True,
        help="Output COCO file path"
    )
    
    return parser.parse_args()


# Set seeds for reproducibility
random.seed(42)
_np_rng = np.random.default_rng(42)


def _flip_h(boxes, W):
    """Flip boxes horizontally."""
    return [[W - (x + w), y, w, h] for x, y, w, h in boxes]


def _flip_v(boxes, H):
    """Flip boxes vertically."""
    return [[x, H - (y + h), w, h] for x, y, w, h in boxes]


def _rotate90_ccw(boxes, W, H):
    """90 CCW rotation."""
    out = []
    for x, y, w, h in boxes:
        x2, y2 = x + w, y + h
        out.append([y, W - x2, y2 - y, x2 - x])
    return out


def _rotate_k90(boxes, W, H, k):
    """Apply k * 90 CCW rotations."""
    cW, cH = W, H
    cur = boxes
    for _ in range(k % 4):
        cur = _rotate90_ccw(cur, cW, cH)
        cW, cH = cH, cW
    return cur, cW, cH


def _clip_boxes(boxes, W, H):
    """Clip boxes to image boundaries."""
    out = []
    for x, y, w, h in boxes:
        x = max(0.0, x)
        y = max(0.0, y)
        w = min(w, W - x)
        h = min(h, H - y)
        if w >= 2 and h >= 2:
            out.append([x, y, w, h])
    return out


def aug_flip_h(img, boxes):
    """Flip image horizontally."""
    W, H = img.size
    return img.transpose(Image.FLIP_LEFT_RIGHT), _clip_boxes(_flip_h(boxes, W), W, H)


def aug_flip_v(img, boxes):
    """Flip image vertically."""
    W, H = img.size
    return img.transpose(Image.FLIP_TOP_BOTTOM), _clip_boxes(_flip_v(boxes, H), W, H)


def aug_rot90(img, boxes):
    """Rotate image 90 degrees clockwise."""
    W, H = img.size
    nb, nW, nH = _rotate_k90(boxes, W, H, k=1)
    return img.transpose(Image.ROTATE_90), _clip_boxes(nb, nW, nH)


def aug_rot180(img, boxes):
    """Rotate image 180 degrees."""
    W, H = img.size
    flipped = [[W - (x + w), H - (y + h), w, h] for x, y, w, h in boxes]
    return img.transpose(Image.ROTATE_180), _clip_boxes(flipped, W, H)


def aug_rot270(img, boxes):
    """Rotate image 270 degrees clockwise."""
    W, H = img.size
    nb, nW, nH = _rotate_k90(boxes, W, H, k=3)
    return img.transpose(Image.ROTATE_270), _clip_boxes(nb, nW, nH)


def aug_noise(img, boxes, sigma=8.0):
    """Add noise to image."""
    arr = np.array(img, dtype=np.float32)
    arr = np.clip(arr + _np_rng.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr), boxes


def brighten(img, boxes):
    """Brighten image."""
    return ImageEnhance.Brightness(img).enhance(1.25), boxes


def darken(img, boxes):
    """Darken image."""
    return ImageEnhance.Brightness(img).enhance(0.75), boxes


def aug_contrast(img, boxes):
    """Adjust contrast."""
    return ImageEnhance.Contrast(img).enhance(1.20), boxes


AUGMENTATIONS = [
    ("fliph", aug_flip_h),
    ("flipv", aug_flip_v),
    ("rot90", aug_rot90),
    ("rot180", aug_rot180),
    ("rot270", aug_rot270),
    ("noise", aug_noise),
    ("bright_up", brighten),
    ("bright_dn", darken),
    ("contrast", aug_contrast),
]


def _fix_categories(coco):
    """Fix duplicate ship category issues."""
    coco["categories"] = [{"id": 1, "name": "ship", "supercategory": "none"}]
    for ann in coco["annotations"]:
        ann["category_id"] = 1
    return coco


def run(train_ann, train_img_dir, out_img_dir, out_coco):
    """
    Main augmentation pipeline.
    
    Args:
        train_ann: Path to training COCO annotations file
        train_img_dir: Directory containing training images
        out_img_dir: Directory to save augmented images
        out_coco: Output COCO file path
    
    Returns:
        None
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)

    with open(train_ann) as f:
        coco = json.load(f)
    coco = _fix_categories(coco)

    id_to_img = {im["id"]: im for im in coco["images"]}
    annotated_ids = {a["image_id"] for a in coco["annotations"]}

    new_images = []
    new_anns = []
    next_img_id = max(id_to_img) + 1
    next_ann_id = max(a["id"] for a in coco["annotations"]) + 1
    count = 0

    for img_id in sorted(annotated_ids):
        meta = id_to_img[img_id]
        src = train_img_dir / meta["file_name"]
        if not src.exists():
            print(f"  [WARN] missing: {src}")
            continue

        orig_stem = src.stem
        orig_dst = out_img_dir / f"{orig_stem}.png"
        pil_img = Image.open(src).convert("L")
        pil_img.save(orig_dst, format="PNG")
        count += 1

        img_anns = [a for a in coco["annotations"] if a["image_id"] == img_id]
        raw_boxes = [a["bbox"] for a in img_anns]
        cat_id = img_anns[0]["category_id"]
        meta["file_name"] = orig_dst.name

        for suffix, fn in AUGMENTATIONS:
            aug_img, aug_boxes = fn(pil_img, [list(b) for b in raw_boxes])
            if not aug_boxes:
                continue

            aug_fname = f"{orig_stem}_{suffix}.png"
            aug_img.save(out_img_dir / aug_fname, format="PNG")

            aW, aH = aug_img.size
            new_images.append({
                "id": next_img_id, "file_name": aug_fname,
                "width": aW, "height": aH,
            })
            for box in aug_boxes:
                new_anns.append({
                    "id": next_ann_id, "image_id": next_img_id,
                    "category_id": cat_id,
                    "bbox": [round(v, 2) for v in box],
                    "area": round(box[2] * box[3], 2),
                    "iscrowd": 0,
                })
                next_ann_id += 1
            next_img_id += 1

    out = {
        "info": coco.get("info", {"description": "SAR ship detection — augmented train split"}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": coco["images"] + new_images,
        "annotations": coco["annotations"] + new_anns,
    }
    
    with open(out_coco, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Copied {count} originals.")
    print(f"Augmented: {len(new_images)} images, {len(new_anns)} annotations")
    print(f"Total output: {len(out['images'])} images, {len(out['annotations'])} annotations")
    print(f"Saved to {out_img_dir} / {out_coco}")


def main():
    """Main function to run the data augmentation."""
    args = parse_args()
    
    print("Running augmentation (train split only)...")
    run(
        Path(args.train_annotations),
        Path(args.train_images_dir),
        Path(args.output_images_dir),
        Path(args.output_coco)
    )


if __name__ == "__main__":
    main()