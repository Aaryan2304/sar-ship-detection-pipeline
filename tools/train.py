#!/usr/bin/env python3
"""
Train YOLO26s-OBB on SSDD for SAR ship detection.

This script documents the complete training configuration including
SAR-specific augmentations and hyperparameters used for the baseline model.

Usage:
    python train.py [--epochs 50] [--batch 8] [--imgsz 640] [--resume]

Results saved to: runs/baseline/
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO26s-OBB on SSDD")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (4GB VRAM limit)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--weights", type=str, default="yolo26s-obb.pt", help="Pretrained weights")
    parser.add_argument("--data", type=str, default="datasets/SSDD/dataset.yaml", help="Dataset config")
    parser.add_argument("--project", type=str, default="runs", help="Project directory")
    parser.add_argument("--name", type=str, default="baseline", help="Run name")
    return parser.parse_args()


def main():
    args = parse_args()

    # Model: YOLO26s-OBB (Ultralytics 8.4.33)
    # YOLO26 is the latest YOLO variant with improved OBB head
    model = YOLO(args.weights)

    # Training configuration
    # Designed for SAR imagery (grayscale, rotated ships, port/open-ocean scenes)
    results = model.train(
        # Dataset
        data=args.data,

        # Training params
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,

        # Optimizer: AdamW (auto-selected by Ultralytics)
        # lr0=0.002, momentum=0.9 from auto-tune

        # ─── SAR-Specific Augmentations ───────────────────────────────
        # SAR is grayscale → disable all color transforms
        hsv_h=0.0,      # No hue shift (grayscale)
        hsv_s=0.0,      # No saturation shift (grayscale)
        hsv_v=0.0,      # No value/brightness shift (SAR intensity is signal)

        # Geometric augmentations (critical for ship orientation diversity)
        degrees=90.0,   # Random rotation ±90° (ships at any heading)
        flipud=0.5,     # Vertical flip 50% (SAR has no "up")
        fliplr=0.5,     # Horizontal flip 50%

        # Mosaic & mixing (moderate for small dataset)
        mosaic=0.5,     # 50% mosaic (helps context, less aggressive than 1.0)
        mixup=0.0,      # No mixup (SAR intensity matters, blending hurts)
        copy_paste=0.0, # No copy-paste (ships are context-dependent)

        # ─── Training Settings ────────────────────────────────────────
        amp=True,           # Mixed precision (FP16) for 4GB VRAM
        device=0,           # GPU 0
        verbose=True,
        close_mosaic=10,    # Disable mosaic for last 10 epochs (fine-tuning)
        patience=0,         # No early stopping (we want full 50 epochs)
        cache=False,        # Don't cache to RAM (1160 images > available RAM)
        plots=True,         # Generate training plots
        save=True,          # Save checkpoints
        exist_ok=True,      # Overwrite existing run

        # Output
        project=args.project,
        name=args.name,
        resume=args.resume,
    )

    print("\n=== TRAINING COMPLETE ===")
    print(f"Results saved to: {args.project}/{args.name}/")
    print(f"Best weights: {args.project}/{args.name}/weights/best.pt")
    print(f"mAP@50:     {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"mAP@50:95:  {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")


if __name__ == "__main__":
    main()
