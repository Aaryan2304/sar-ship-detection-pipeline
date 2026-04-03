# SAR Ship Detection Pipeline

## Project Description

This repository contains a complete pipeline for detecting ships in Synthetic Aperture Radar (SAR) imagery. It demonstrates an end-to-end data preparation workflow for satellite-based object detection — from raw GeoTIFF ingestion through chipping, augmentation, COCO annotation management, and FiftyOne visualization.

## What It Does

1. **Chips GeoTIFFs** into ML-ready tiles using sliding windows with configurable overlap
2. **Augments training data** with 9 transformations while maintaining correct COCO bounding boxes
3. **Manages and visualizes** the dataset through FiftyOne with QC tagging and metadata enrichment

## Project Structure

```
sar-ship-detection-pipeline/
├── data/
│   ├── raw/            # Source SAR GeoTIFFs (included)
│   │   └── README.md
│   └── chips/          # Chip metadata sidecars
│       └── *_chips_meta.json
├── annotations/
│   └── labels/         # COCO annotations + images (Roboflow format)
│       ├── train/
│       ├── valid/
│       ├── test/
│       └── augmented/
├── pipeline/
│   ├── __init__.py
│   ├── chip_tiles.py     # Sliding-window chip generation
│   ├── augment_data.py   # Image + bbox augmentations
│   └── ingest_fiftyone.py # FiftyOne dataset management
├── docs/
│   ├── methodology.md
│   ├── usage.md
│   └── process_document.md
├── assets/
│   └── screenshots/
├── setup.py
├── requirements.txt
└── README.md
```

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install .

# Run the pipeline
chip-tiles --input-tif data/raw/sar_image_1.tif --output-dir data/chips
augment-data --train-annotations annotations/labels/train/_annotations.coco.json \
    --train-images-dir annotations/labels/train \
    --output-images-dir annotations/labels/augmented \
    --output-coco annotations/labels/augmented/_annotations.coco.json
ingest-fiftyone --aug-coco annotations/labels/augmented/_annotations.coco.json \
    --aug-img-dir annotations/labels/augmented \
    --img-root data/chips
```

## Dataset

- **Source**: Umbra X-band SAR imagery (GEC format, ~0.5m resolution)
- **Images**: 3 scenes, ~232 MB total
- **Chips**: 459 tiles at 640x640px with 64px overlap
- **Annotated**: 8 chips, 13 ship instances across train/valid/test
- **After augmentation**: 60 training images, 100 annotations

See `data/README.md` for download instructions and `annotations/labels/README.md` for label format details.

## License

MIT
