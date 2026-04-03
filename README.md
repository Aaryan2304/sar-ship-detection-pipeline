# SAR Ship Detection Pipeline

End-to-end data preparation pipeline for ship detection in Synthetic Aperture Radar (SAR) imagery. Takes raw GeoTIFF scenes, chips them into tiles, augments annotated training data, and loads everything into FiftyOne for visualization and QC.

## What This Is

A data preparation pipeline — not a trained model. It covers the full path from raw SAR imagery to an analysis-ready COCO dataset:

1. **Chip** GeoTIFFs into ML-ready tiles via sliding windows
2. **Augment** training images with 9 transformations (COCO bbox-aware)
3. **Load** into FiftyOne for metadata enrichment, QC tagging, and visualization

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install .

# 1. Chip GeoTIFFs into tiles
chip-tiles --input-tif data/raw/sar_image_1.tif --output-dir data/chips

# 2. Augment training data
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

## Project Structure

```
sar-ship-detection-pipeline/
├── data/
│   ├── raw/              # Source SAR GeoTIFFs (~232 MB, 3 scenes)
│   │   ├── sar_image_1.tif
│   │   ├── sar_image_2.tif
│   │   └── sar_image_3.tif
│   └── chips/            # Chip metadata sidecars
│       ├── sar_image_1_chips_meta.json
│       ├── sar_image_2_chips_meta.json
│       └── sar_image_3_chips_meta.json
├── annotations/
│   └── labels/           # COCO format: images + JSON co-located per split
│       ├── train/        # 6 chips, 10 annotations
│       ├── valid/        # 1 chip, 1 annotation
│       ├── test/         # 1 chip, 1 annotation
│       └── augmented/    # 60 chips, ~90 annotations (9× augmentations)
├── pipeline/
│   ├── __init__.py
│   ├── chip_tiles.py       # Sliding-window chip generation
│   ├── augment_data.py     # Image + bbox augmentations (COCO-aware)
│   └── ingest_fiftyone.py  # FiftyOne dataset management + QC
├── docs/
│   ├── methodology.md      # Pipeline methodology
│   ├── process_document.md # Technical process details
│   └── usage.md            # CLI usage reference
├── assets/
│   └── screenshots/        # FiftyOne GUI screenshots
├── setup.py
├── requirements.txt
└── README.md
```

## Dataset

| Metric | Value |
|--------|-------|
| Source scenes | 3 (Umbra X-band, GEC format, ~0.5m resolution) |
| Total GeoTIFF size | ~232 MB |
| Chips generated | 459 (640x640px, 64px overlap) |
| Annotated chips | 8 |
| Ship instances | 13 (single class: `ship`) |
| After augmentation | 60 images, 100 annotations |

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

## License

MIT
