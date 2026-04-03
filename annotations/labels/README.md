# Annotations

COCO-format label files with images co-located per split (Roboflow export layout).

## Layout

```
annotations/labels/
├── train/
│   ├── _annotations.coco.json
│   ├── <chip_1>.jpg
│   ├── <chip_2>.jpg
│   └── ...
├── valid/
│   ├── _annotations.coco.json
│   └── <chip>.jpg
├── test/
│   ├── _annotations.coco.json
│   └── <chip>.jpg
└── augmented/
    ├── _annotations.coco.json
    ├── <chip_1>.png
    ├── <chip_1>_fliph.png
    └── ...
```

Each split directory contains both its JSON annotation file and the corresponding images. This is the standard Roboflow COCO export format.

## How to Load

```python
import fiftyone as fo

# Load a single split
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path="annotations/labels/train",
    labels_path="annotations/labels/train/_annotations.coco.json",
    label_types=["detections"],
)
```
