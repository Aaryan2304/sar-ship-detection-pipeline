# SAR Ship Detection Pipeline

## Project Description

Complete pipeline for oriented bounding box (OBB) ship detection in SAR imagery, from training on the SSDD benchmark to cross-domain evaluation on custom Umbra Space chips.

## What It Does

1. **Chips GeoTIFFs** into ML-ready 640×640 tiles with configurable overlap
2. **Converts SSDD annotations** from Pascal VOC RBox → YOLO OBB format (angle verified at 0.95px median error)
3. **Trains YOLOv8/v11-OBB** on 1,160 SSDD images (2,587 ship instances)
4. **Evaluates cross-domain generalization** on 40 Umbra chips (30 annotated + 10 negative)
5. **Augments COCO-labeled chips** with 9 bbox-aware transformations
6. **Visualizes** in FiftyOne with QC tagging and metadata

## Project Structure

```
sar-ship-detection-pipeline/
├── pipeline/
│   ├── chip_tiles.py         # GeoTIFF → 640×640 tile chipper
│   ├── augment_data.py       # COCO-aware image + bbox augments
│   └── ingest_fiftyone.py    # FiftyOne dataset management + QC
├── datasets/
│   ├── SSDD/                 # 1,160 imgs, 2,587 ships
│   │   ├── images/, annotations/, labels/, splits/
│   │   └── dataset.yaml
│   └── umbra-test/           # 40 chips (30 pos + 10 neg)
│       ├── test/images/, test/labels/
│       └── dataset.yaml
├── tools/
│   ├── convert_ssdd_to_yolo_obb.py
│   └── verify_rbox_angle.py
├── evaluate.py               # Cross-domain evaluation
├── data/
│   ├── raw/                  # 7 source GeoTIFFs (~500 MB)
│   └── chips/                # 459 generated chips + metadata for 3 images
├── annotations/labels/       # COCO chips (8 annotated, 13 ships)
│   ├── train/ (6 chips)
│   ├── valid/ (1 chip)
│   ├── test/ (1 chip)
│   └── augmented/ (60 chips)
├── outputs/angle_verify/     # Label verification overlays
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

chip-tiles --input-tif data/raw/sar_image_1.tif --output-dir data/chips
augment-data --train-annotations annotations/labels/train/_annotations.coco.json \
    --train-images-dir annotations/labels/train \
    --output-images-dir annotations/labels/augmented \
    --output-coco annotations/labels/augmented/_annotations.coco.json
ingest-fiftyone --aug-coco annotations/labels/augmented/_annotations.coco.json \
    --aug-img-dir annotations/labels/augmented \
    --img-root data/chips
python evaluate.py --data datasets/umbra-test/dataset.yaml
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
