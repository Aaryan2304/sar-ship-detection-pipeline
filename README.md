# SAR Ship Detection Pipeline

End-to-end SAR ship detection pipeline with oriented bounding boxes (OBB). Trains on the SSDD benchmark (1,160 images, 2,587 ships) and evaluates cross-domain generalization on custom Umbra Space SAR chips.

## What This Is

A complete SAR ship detection pipeline that:

1. **Chips** large GeoTIFFs into ML-ready tiles via sliding windows
2. **Converts** SSDD RBox annotations to YOLO OBB format (verified angle convention)
3. **Trains** OBB detectors (YOLOv8/v11-OBB) on the SSDD benchmark
4. **Evaluates** cross-domain generalization (SSDD → Umbra open data)
5. **Augments** COCO-labeled chips with 9 transformations (bbox-aware)
6. **Loads** everything into FiftyOne for QC and visualization

## Quick Start

```bash
pip install -r requirements.txt
pip install .

# Chip GeoTIFFs into tiles
chip-tiles --input-tif data/raw/sar_image_1.tif --output-dir data/chips

# Augment COCO-labeled training images
augment-data \
    --train-annotations annotations/labels/train/_annotations.coco.json \
    --train-images-dir annotations/labels/train \
    --output-images-dir annotations/labels/augmented \
    --output-coco annotations/labels/augmented/_annotations.coco.json

# Load into FiftyOne
ingest-fiftyone \
    --aug-coco annotations/labels/augmented/_annotations.coco.json \
    --aug-img-dir annotations/labels/augmented \
    --img-root data/chips

# Evaluate a model on the cross-domain test set
python evaluate.py --data datasets/umbra-test/dataset.yaml --weights yolo11n-obb.pt
```

## Project Structure

```
sar-ship-detection-pipeline/
├── pipeline/
│   ├── chip_tiles.py         # Sliding-window chip generation (640x640, 64px overlap)
│   ├── augment_data.py       # COCO-aware image + bbox augmentations
│   └── ingest_fiftyone.py    # FiftyOne dataset management + QC
├── datasets/
│   ├── SSDD/                 # SSDD benchmark (1,160 imgs, 2,587 ships)
│   │   ├── images/           # 1,160 JPGs
│   │   ├── annotations/      # 1,160 XML (Pascal VOC with rotated_bndbox)
│   │   ├── labels/           # YOLO OBB .txt files (1160 total)
│   │   └── splits/           # train/test splits
│   └── umbra-test/           # Cross-domain test set (40 chips)
│       ├── test/
│       │   ├── images/       # 40 chips (30 positive, 10 negative)
│       │   └── labels/       # 40 label files (30 annotated, 10 empty)
│       └── dataset.yaml
├── tools/
│   ├── convert_ssdd_to_yolo_obb.py   # SSDD XML → YOLO OBB conversion
│   └── verify_rbox_angle.py          # Angle convention verification
├── evaluate.py               # Cross-domain evaluation script
├── data/
│   ├── raw/                  # Source GeoTIFFs (~500 MB, 7 scenes)
│   │   └── README.md
│   └── chips/                # Generated chips + metadata sidecars
├── annotations/
│   └── labels/               # COCO-labeled chips (manually annotated)
│       ├── train/            # 6 chips, 10 annotations
│       ├── valid/            # 1 chip, 1 annotation
│       ├── test/             # 1 chip, 1 annotation
│       └── augmented/        # 60 chips, ~90 annotations
├── docs/                     # Documentation
├── assets/
│   └── screenshots/          # FiftyOne GUI screenshots
├── outputs/
│   └── angle_verify/         # Label verification overlays
├── setup.py
├── requirements.txt
├── UPGRADE_PLAN.md
└── README.md
```

## Datasets

### SSDD Benchmark
| Metric | Value |
|--------|-------|
| Images | 1,160 (TerraSAR-X, RadarSat-2, Sentinel-1) |
| Ships | 2,587 instances |
| Splits | 928 train / 232 test (+ inshore/offshore) |
| Labels | Pascal VOC with `rotated_bndbox` (RBox variant) |

### Umbra Cross-Domain Test Set
| Metric | Value |
|--------|-------|
| Chips | 40 (30 annotated, 10 negative) |
| Source | Umbra X-band SAR (GEC, ~0.5m resolution) |
| Purpose | Cross-domain evaluation (SSDD → Umbra) |

### Original COCO-Labeled Chips
| Metric | Value |
|--------|-------|
| Chips | 8 (train/valid/test splits) |
| Ships | 13 instances |
| After augmentation | 60 images, ~90 annotations |

## Augmentations

Each training image gets 9 augmented variants with proper COCO bbox transforms:

| Transform | Effect on BBoxes |
|-----------|-----------------|
| Horizontal flip | x-coords mirrored |
| Vertical flip | y-coords mirrored |
| 90° / 180° / 270° rotation | Corner-remapped |
| Gaussian noise (σ=8) | Unchanged |
| Brightness +25% / -25% | Unchanged |
| Contrast +20% | Unchanged |

Dropped boxes with zero area after transform. All other boxes clipped to boundaries.

## Upgrade Roadmap

See `UPGRADE_PLAN.md` for the full OBB upgrade plan including training config, evaluation phases, SAHI inference, and inference optimization.

## License

MIT
