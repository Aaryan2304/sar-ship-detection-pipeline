# SAR Ship Detection Pipeline Usage

This document covers the CLI tools and scripts in this repo.

## Installation

```bash
pip install -r requirements.txt
pip install .
```

## Pipeline Components

### 1. Chip Generation (`chip-tiles`)

Convert SAR GeoTIFFs into 640x640 tiles:

```bash
chip-tiles \
    --input-tif data/raw/sar_image_1.tif \
    --output-dir data/chips
```

Optional overrides:
```bash
chip-tiles \
    --input-tif data/raw/sar_image_2.tif \
    --output-dir data/chips \
    --tile-size 512 \
    --overlap 32 \
    --min-valid 0.3
```

### 2. Data Augmentation (`augment-data`)

Apply COCO bbox-aware transformations to training images:

```bash
augment-data \
    --train-annotations annotations/labels/train/_annotations.coco.json \
    --train-images-dir annotations/labels/train \
    --output-images-dir annotations/labels/augmented \
    --output-coco annotations/labels/augmented/_annotations.coco.json
```

Produces 9 augmentations per image: flips, rotations, noise, brightness, contrast.

### 3. FiftyOne Integration (`ingest-fiftyone`)

Load datasets into FiftyOne for visualization and QC:

```bash
ingest-fiftyone \
    --aug-coco annotations/labels/augmented/_annotations.coco.json \
    --aug-img-dir annotations/labels/augmented \
    --img-root data/chips \
    --dataset-name sar_ships_dataset
```

Add `--run-embeddings` for visual embeddings.

## SSDD Tools

### Label Conversion

Convert SSDD Pascal VOC XML to YOLO OBB format:

```bash
python tools/convert_ssdd_to_yolo_obb.py --mode xywhr --visualize
```

Verify angle convention:

```bash
python tools/convert_ssdd_to_yolo_obb.py --verify-only
```

Label output format: `class x1 y1 x2 y2 x3 y3 x4 y4` (9-value, normalized).

## Evaluation

Run cross-domain evaluation:

```bash
python evaluate.py --data datasets/umbra-test/dataset.yaml --weights best.pt
```

Options:

```bash
python evaluate.py \
    --data datasets/umbra-test/dataset.yaml \
    --weights best.pt \
    --output-dir outputs/eval \
    --conf-thres 0.25 \
    --iou-thres 0.7 \
    --imgsz 640
```

Produces confusion matrix, PR curves, mAP@50, mAP@50:95, per-class metrics, and per-image latency report.

## Full Pipeline

```bash
# 1. Chip all GeoTIFFs
for tif in data/raw/sar_image_*.tif; do
    chip-tiles --input-tif "$tif" --output-dir data/chips
done

# 2. Augment training set
augment-data \
    --train-annotations annotations/labels/train/_annotations.coco.json \
    --train-images-dir annotations/labels/train \
    --output-images-dir annotations/labels/augmented \
    --output-coco annotations/labels/augmented/_annotations.coco.json

# 3. Load into FiftyOne
ingest-fiftyone \
    --aug-coco annotations/labels/augmented/_annotations.coco.json \
    --aug-img-dir annotations/labels/augmented \
    --img-root data/chips
```

## Notes

- All scripts use CLI arguments — no hardcoded paths
- Tile size: 640x640 with 64px overlap (configurable)
- Augmentation seed: 42 (fixed for reproducibility)
- SSDD labels use 9-value per-detection format with corners normalized to [0,1]
