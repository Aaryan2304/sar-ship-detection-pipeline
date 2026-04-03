# SAR Ship Detection Pipeline Usage

This document describes how to use the SAR ship detection pipeline.

## Installation

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies and the package:
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

Apply transformations to training images and update COCO annotations:

```bash
augment-data \
    --train-annotations annotations/labels/train/_annotations.coco.json \
    --train-images-dir annotations/labels/train \
    --output-images-dir annotations/labels/augmented \
    --output-coco annotations/labels/augmented/_annotations.coco.json
```

This produces 9 augmentations per training image: horizontal/vertical flips, 90/180/270 degree rotations, noise, brightness up/down, and contrast.

### 3. FiftyOne Integration (`ingest-fiftyone`)

Load the dataset into FiftyOne for visualization, QC, and analysis:

```bash
ingest-fiftyone \
    --aug-coco annotations/labels/augmented/_annotations.coco.json \
    --aug-img-dir annotations/labels/augmented \
    --img-root data/chips \
    --dataset-name sar_ships_dataset
````

Add `--run-embeddings` to compute visual embeddings (requires more time).

## Running the Full Pipeline

```bash
# Step 1: chip all GeoTIFFs
for tif in data/raw/sar_image_*.tif; do
    chip-tiles --input-tif "$tif" --output-dir data/chips
done

# Step 2: augment training set
augment-data \
    --train-annotations annotations/labels/train/_annotations.coco.json \
    --train-images-dir annotations/labels/train \
    --output-images-dir annotations/labels/augmented \
    --output-coco annotations/labels/augmented/_annotations.coco.json

# Step 3: load into FiftyOne
ingest-fiftyone \
    --aug-coco annotations/labels/augmented/_annotations.coco.json \
    --aug-img-dir annotations/labels/augmented \
    --img-root data/chips \
    --dataset-name sar_ships_dataset
```

## Notes

- The pipeline uses no hardcoded paths — all inputs are CLI arguments
- Tile size defaults to 640x640 with 64px overlap (10% stride), configurable
- Augmentation seed is fixed (42) for reproducibility
