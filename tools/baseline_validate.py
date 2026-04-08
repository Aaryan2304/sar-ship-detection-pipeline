#!/usr/bin/env python3
"""
Baseline validation: measure mAP, precision, recall, inference latency.

Usage:
    python tools/baseline_validate.py [--weights runs/baseline/weights/best.pt]

Outputs validation metrics and inference latency for the Phase 5 benchmark table.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline validation + latency")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(PROJECT_ROOT / "runs/baseline/weights/best.pt"),
        help="Path to model weights",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "datasets/SSDD/dataset.yaml"),
        help="Dataset config",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.weights)

    # Run validation
    results = model.val(
        data=args.data,
        imgsz=640,
        batch=8,
        device=0,
        verbose=True,
    )

    # Print metrics
    print("\n=== BASELINE VALIDATION RESULTS ===")
    print(f"mAP@50:     {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"mAP@50:95:  {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"Precision:  {results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"Recall:     {results.results_dict.get('metrics/recall(B)', 0):.4f}")

    # Measure inference latency
    dummy = torch.randn(1, 3, 640, 640).cuda()
    model.model.eval()

    # Warmup
    for _ in range(10):
        _ = model.model(dummy)

    # Benchmark (100 iterations for stable stats)
    latencies = []
    with torch.no_grad():
        for _ in range(100):
            start = time.perf_counter()
            _ = model.model(dummy)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    print(f"\nInference Latency (PyTorch): {np.mean(latencies):.1f} +/- {np.std(latencies):.1f} ms")
    print("(Ultralytics reports model-only latency in validation output above)")


if __name__ == "__main__":
    main()
