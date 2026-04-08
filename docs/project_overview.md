# SAR Ship Detection Pipeline

## Project Description

Complete pipeline for oriented bounding box (OBB) ship detection in SAR imagery, from training on the SSDD benchmark to cross-domain evaluation on custom Umbra Space chips.

## What It Does

1. **Chips GeoTIFFs** into ML-ready 640×640 tiles with configurable overlap
2. **Converts SSDD annotations** from Pascal VOC RBox → YOLO OBB format (9-value corner format, angle verified at 0.95px median error)
3. **Trains YOLO26-OBB** on 1,160 SSDD images (2,587 ship instances)
4. **Evaluates cross-domain generalization** on 40 Umbra chips (30 annotated + 10 negative)
5. **Augments COCO-labeled chips** with 9 bbox-aware transformations
6. **Visualizes** in FiftyOne with QC tagging and metadata

## Phase 2 Results (Baseline Model)

| Metric | Value |
|--------|-------|
| **mAP@50** | 0.9868 |
| **mAP@50:95** | 0.7957 |
| **Precision** | 0.9662 |
| **Recall** | 0.9469 |
| **Inference** | 18.9 ms (PyTorch, 640px) |

Model: `yolo26s-obb.pt` trained for 50 epochs on SSDD (928 train / 232 val).

## Project Structure

```
sar-ship-detection-pipeline/
├── pipeline/
│   ├── chip_tiles.py           # GeoTIFF → 640×640 tile chipper
│   ├── augment_data.py         # COCO-aware image + bbox augments
│   └── ingest_fiftyone.py      # FiftyOne dataset management + QC
├── tools/
│   ├── train.py                # Training script (SAR augmentations documented)
│   ├── evaluate.py             # Cross-domain evaluation
│   ├── baseline_validate.py    # Baseline validation + inference latency
│   ├── convert_ssdd_to_yolo_obb.py   # SSDD XML → YOLO OBB conversion
│   └── verify_rbox_angle.py          # Angle convention verification
├── datasets/
│   ├── SSDD/                   # 1,160 imgs, 2,587 ships
│   │   ├── images/, annotations/, labels/
│   │   ├── train.txt, val.txt
│   │   └── dataset.yaml
│   └── umbra-test/             # 40 chips (30 pos + 10 neg)
│       ├── test/images/, test/labels/
│       └── dataset.yaml
├── runs/
│   ├── baseline/               # Training results (plots, weights, args)
│   │   ├── weights/best.pt     # Best checkpoint
│   │   └── results.csv         # Per-epoch metrics
│   └── validation/             # Standalone validation results
├── data/
│   └── raw/                    # 7 source GeoTIFFs (~500 MB)
├── annotations/labels/         # COCO chips (8 annotated, 13 ships)
│   ├── train/ (6 chips)
│   ├── valid/ (1 chip)
│   ├── test/ (1 chip)
│   └── augmented/ (60 chips)
├── docs/
├── assets/screenshots/
├── setup.py
├── requirements.txt
├── UPGRADE_PLAN.md
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
pip install .

# Train baseline model (all augmentations + hyperparams documented in train.py)
python tools/train.py --epochs 50 --batch 8

# Validate best model
python tools/baseline_validate.py

# Convert SSDD labels to YOLO OBB format
python tools/convert_ssdd_to_yolo_obb.py --mode corners

# Verify angle convention
python tools/verify_rbox_angle.py

# Evaluate on Umbra cross-domain test set
python tools/evaluate.py --data datasets/umbra-test/dataset.yaml
```

## Datasets

### SSDD (Training)
- 1,160 SAR images from TerraSAR-X, RadarSat-2, Sentinel-1
- 2,587 annotated ships (RBox format)
- Splits: 928 train / 232 test

### Umbra (Cross-Domain Test)
- 40 chips from 7 GeoTIFFs (~0.5m resolution)
- 30 annotated, 10 negative samples

## License

MIT
