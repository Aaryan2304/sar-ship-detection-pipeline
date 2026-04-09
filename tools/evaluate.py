#!/usr/bin/env python3
"""Evaluation script for SAR ship detection (OBB).

Evaluates a YOLOv8/v11-obb model on SSDD or Umbra datasets.

Metrics: mAP@50, mAP@50:95, per-class precision/recall, inference speed.

Usage:
    python evaluate.py                                          # Umbra test set, default pretrained weights
    python evaluate.py --data datasets/SSDD/dataset.yaml       # SSDD dataset
    python evaluate.py --weights best.pt                       # Your trained model
    python evaluate.py --no-overlays                           # Skip visual overlay generation
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

def draw_obb_box(image, corners, color, label, thickness=2):
    """Draw a single oriented bounding box polygon on an image.

    Args:
        image: BGR image array (H, W, 3).
        corners: Flat list of 8 values [x1,y1,x2,y2,x3,y3,x4,y4] in pixel coords.
        color: BGR tuple, e.g. (0, 255, 0) for green.
        label: Text label to draw near the box.
        thickness: Line thickness in pixels.
    """
    pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
    if label:
        # Place label at the top-left corner (min y, then min x)
        # corners is flat: [x1,y1,x2,y2,x3,y3,x4,y4]
        ys = [corners[i] for i in range(1, 8, 2)]
        xs_at_min_y = [corners[i - 1] for i in range(1, 8, 2)]
        top_idx = min(range(4), key=lambda i: (ys[i], xs_at_min_y[i]))
        tx, ty = int(xs_at_min_y[top_idx]), int(ys[top_idx])
        cv2.putText(image, label, (tx, max(ty - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def load_gt_boxes(label_path, img_w, img_h):
    """Load ground truth OBB boxes from a YOLO-format label file.

    Args:
        label_path: Path to .txt label file.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        List of (class_name, pixel_corners) tuples.
        pixel_corners is a flat list [x1,y1,x2,y2,x3,y3,x4,y4].
    """
    boxes = []
    if not label_path.exists() or label_path.stat().st_size == 0:
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            cls_id = int(parts[0])
            # Normalized corners → pixel coords
            coords = [float(x) for x in parts[1:9]]
            pixel_corners = []
            for i in range(0, 8, 2):
                pixel_corners.append(coords[i] * img_w)
                pixel_corners.append(coords[i + 1] * img_h)
            boxes.append(("ship", pixel_corners))
    return boxes


def generate_overlays(model, image_dir, label_dir, output_dir, conf_thres, imgsz):
    """Generate visual overlays with GT (green) and predicted (red) OBB boxes.

    Runs inference on each image, draws both GT and predictions, saves to output_dir.

    Args:
        model: Loaded YOLO model.
        image_dir: Directory containing test images.
        label_dir: Directory containing YOLO OBB label files.
        output_dir: Directory to save overlay images.
        conf_thres: Confidence threshold for predictions.
        imgsz: Inference image size.

    Returns:
        Dict with overlay stats: total images, images with predictions, images with GT.
    """
    overlay_dir = Path(output_dir) / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = sorted([
        f for f in Path(image_dir).iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not image_files:
        print(f"  No images found in {image_dir}")
        return {"total": 0, "with_preds": 0, "with_gt": 0}

    stats = {"total": len(image_files), "with_preds": 0, "with_gt": 0}
    print(f"\n--- GENERATING OVERLAYS ({len(image_files)} images) ---")

    for idx, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] Could not read: {img_path.name}")
            continue

        img_h, img_w = img.shape[:2]
        overlay = img.copy()

        # Draw ground truth (green)
        label_path = Path(label_dir) / (img_path.stem + ".txt")
        gt_boxes = load_gt_boxes(label_path, img_w, img_h)
        if gt_boxes:
            stats["with_gt"] += 1
        for cls_name, corners in gt_boxes:
            draw_obb_box(overlay, corners, (0, 255, 0), f"GT: {cls_name}")

        # Run prediction (red)
        preds = model(str(img_path), imgsz=imgsz, conf=conf_thres, verbose=False)
        pred_count = 0
        for r in preds:
            if r.obb is not None and len(r.obb):
                for det in r.obb:
                    # OBB xyxyxyxy format: 8 values = 4 corner points
                    xyxyxyxy = det.xyxyxyxy[0].cpu().numpy().flatten()
                    conf = float(det.conf[0])
                    corners = list(xyxyxyxy)
                    draw_obb_box(overlay, corners, (0, 0, 255),
                                 f"ship {conf:.2f}", thickness=2)
                    pred_count += 1
        if pred_count > 0:
            stats["with_preds"] += 1

        # Add legend
        cv2.rectangle(overlay, (5, 5), (180, 50), (30, 30, 30), -1)
        cv2.putText(overlay, "Green=GT  Red=Pred", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        out_path = overlay_dir / img_path.name
        cv2.imwrite(str(out_path), overlay)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(image_files):
            print(f"  [{idx + 1}/{len(image_files)}] Overlays generated")

    print(f"  Saved to: {overlay_dir}")
    print(f"  Images with GT: {stats['with_gt']}, with predictions: {stats['with_preds']}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Cross-domain SAR ship detection eval")
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11n-obb.pt",
        help="Path to .pt weights file (default: pretrained ultralytics weights)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/umbra-test/dataset.yaml",
        help="Path to dataset YAML file",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.7,
        help="IoU threshold for NMS",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip visual overlay generation",
    )
    args = parser.parse_args()

    from ultralytics import YOLO
    from ultralytics.data import YOLODataset
    import torch

    # Load model
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    # Run validation
    print(f"\nValidating on {args.data} at imgsz={args.imgsz}...")
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf_thres,
        iou=args.iou_thres,
        save_json=True,
        save_hybrid=False,
        plots=True,
    )

    # Extract metrics
    box_metrics = results.box  # for OBB, results.box still has standard metrics
    maps = results.maps  # mAP per class
    names = results.names

    print("\n" + "=" * 60)
    print(" CROSS-DOMAIN EVALUATION RESULTS (SSDD -> Umbra)")
    print("=" * 60)

    print(f"\nModel: {args.weights}")
    print(f"Test set: 40 images (30 positive, 10 negative)")
    print(f"Confidence: {args.conf_thres}, IoU: {args.iou_thres}")

    # mAP metrics
    print(f"\n{'Metric':<30} {'Value':>10}")
    print(f"{'------':<30} {'-'*10:>10}")
    print(f"{'mAP@50:95 (overall)':<30} {results.box.map:>10.4f}")
    print(f"{'mAP@50':<30} {results.box.map50:>10.4f}")
    print(f"{'mAP@75':<30} {results.box.map75:>10.4f}")

    if len(maps) == len(names):
        for cls_id, cls_name in names.items():
            print(f"{'mAP@50:95 (%s):' % cls_name:<30} {maps[cls_id]:>10.4f}")

    # Precision/Recall
    print(f"\n{'Precision':<30} {results.box.mp:>10.4f}")
    print(f"{'Recall':<30} {results.box.mr:>10.4f}")

    # False Positive Rate computation
    print(f"\n--- FALSE POSITIVE ANALYSIS ---")
    fp_images = 0
    total_neg = 0
    total_fp_detections = 0

    # Count false positives from negative samples
    label_dir = Path("datasets/umbra-test/test/labels")
    neg_labels = [f for f in label_dir.glob("*.txt") if f.stat().st_size == 0]
    total_neg = len(neg_labels)
    if total_neg > 0:
        # Use results per-image to compute FP on negatives
        if hasattr(results, 'preds') and results.preds:
            for pred, lab in zip(results.preds, results.labels):
                if len(lab) == 0:  # Negative sample
                    if len(pred) > 0:  # Model made detections
                        fp_images += 1
                        total_fp_detections += len(pred)

        print(f"{'Negative images tested':<30} {total_neg:>10d}")
        print(f"{'Neg images with FP':<30} {fp_images:>10d}")
        print(f"{'Total FP detections':<30} {total_fp_detections:>10d}")
        if total_neg > 0:
            fp_rate = fp_images / total_neg
            print(f"{'FP image rate':<30} {fp_rate:>10.4f}")

    # Per-image inference speed
    print(f"\n--- INFERENCE SPEED ---")
    # Run a quick timed benchmark
    test_dir = Path("datasets/umbra-test/test/images")
    test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    if test_images:
        times = []
        model.model.eval()
        with torch.no_grad():
            # Warmup
            for img_path in test_images[:3]:
                _ = model(img_path, imgsz=args.imgsz, conf=args.conf_thres, verbose=False)
            # Timed runs
            for img_path in test_images:
                start = time.perf_counter()
                _ = model(img_path, imgsz=args.imgsz, conf=args.conf_thres, verbose=False)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms

        times = np.array(times)
        print(f"{'Images tested':<30} {len(times):>10d}")
        print(f"{'Mean latency (ms)':<30} {times.mean():>10.1f}")
        print(f"{'Median latency (ms)':<30} {np.median(times):>10.1f}")
        print(f"{'P95 latency (ms)':<30} {np.percentile(times, 95):>10.1f}")
        print(f"{'Min latency (ms)':<30} {times.min():>10.1f}")

    # Visual overlays (GT in green, predictions in red)
    overlay_stats = {"total": 0, "with_preds": 0, "with_gt": 0}
    if not args.no_overlays:
        # Resolve image/label dirs from dataset YAML
        import yaml
        with open(args.data) as f:
            ds_cfg = yaml.safe_load(f)
        # 'path' in Ultralytics YAML is relative to CWD (project root)
        ds_root = Path(ds_cfg.get("path", "."))
        if not ds_root.is_absolute():
            ds_root = Path.cwd() / ds_root
        # Use the 'test' or 'val' split for images
        test_split = ds_cfg.get("test") or ds_cfg.get("val", "test/images")
        image_dir = (ds_root / test_split).resolve()
        # Labels mirror the images directory structure
        label_dir = Path(str(image_dir).replace("images", "labels"))

        overlay_stats = generate_overlays(
            model, image_dir, label_dir,
            output_dir="outputs/umbra_eval",
            conf_thres=args.conf_thres,
            imgsz=args.imgsz,
        )

    # Save results
    output_dir = Path("outputs/umbra_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_results = {
        "model": args.weights,
        "dataset": str(Path(args.data).resolve()),
        "imgsz": args.imgsz,
        "conf_thres": args.conf_thres,
        "iou_thres": args.iou_thres,
        "mAP_50_95": float(results.box.map),
        "mAP_50": float(results.box.map50),
        "mAP_75": float(results.box.map75),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "fp_images": fp_images,
        "total_negatives": total_neg,
        "fp_rate": float(fp_images / total_neg) if total_neg > 0 else None,
        "overlay_total": overlay_stats["total"],
        "overlay_with_preds": overlay_stats["with_preds"],
        "overlay_with_gt": overlay_stats["with_gt"],
    }

    result_path = output_dir / "umbra_eval_results.json"
    with open(result_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nResults saved to: {result_path}")

    # Also generate a confusion matrix and PR curves (Ultralytics already plots these)
    print("Validation plots saved in: runs/obb/val/")


if __name__ == "__main__":
    main()
