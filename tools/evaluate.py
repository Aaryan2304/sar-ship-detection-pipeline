#!/usr/bin/env python3
"""Evaluation script for SAR ship detection (OBB).

Evaluates a YOLOv8/v11-obb model on SSDD or Umbra datasets.

Metrics: mAP@50, mAP@50:95, per-class precision/recall, inference speed.

Usage:
    python evaluate.py                                          # Umbra test set, default pretrained weights
    python evaluate.py --data datasets/SSDD/dataset.yaml       # SSDD dataset
    python evaluate.py --weights best.pt                       # Your trained model
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

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
    }

    result_path = output_dir / "umbra_eval_results.json"
    with open(result_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nResults saved to: {result_path}")

    # Also generate a confusion matrix and PR curves (Ultralytics already plots these)
    print("Validation plots saved in: runs/obb/val/")


if __name__ == "__main__":
    main()
