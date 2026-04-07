# SAR Ship Detection — Methodology & Pipeline

## Overview

This document describes the methodology and pipeline for detecting ships in Synthetic Aperture Radar (SAR) imagery with oriented bounding boxes (OBB).

## Pipeline Description

The pipeline covers two parallel workflows:

### 1. Data Preparation (COCO chips)
1. **Chip Generation**: Convert large GeoTIFFs into 640x640 tiles with 64px overlap
2. **Data Augmentation**: Apply 9 COCO bbox-aware transformations to training chips
3. **Dataset Management**: Load into FiftyOne for QC, metadata, and visualization

### 2. OBB Training (SSDD → Umbra cross-domain)
1. **SSDD Label Conversion**: Pascal VOC XML with `rotated_bndbox` → YOLO OBB format
2. **Angle Verification**: Confirmed standard 2D rotation (0.95px median error)
3. **Model Training**: YOLOv8/v11-OBB via Ultralytics
4. **Cross-Domain Eval**: Test on 40 Umbra chips (30 annotated, 10 negative)

## Methodology Details

### Chip Generation

The pipeline processes SAR GeoTIFFs into tiles of size 640×640 pixels with 64 pixel overlap to ensure comprehensive coverage of the entire image space.

- Tile Size: 640×640 pixels
- Overlap: 64 pixels (10% of tile width/height)
- Minimum Valid Data: 50% of pixels must be valid
- Filtered out tiles with less than 50% valid data

### Data Augmentation

The pipeline applies various transformations to the training set images and their corresponding bounding box annotations:

- Horizontal flip (fliph)
- Vertical flip (flipv)  
- Rotations (90, 180, 270 degrees)
- Noise addition (noise)
- Brightness adjustments (bright_up, bright_dn)
- Contrast changes (contrast)

Each augmentation maintains proper bounding box coordinates for the ship annotations.

### Dataset Management

The pipeline integrates FiftyOne to manage datasets with:

- Metadata tracking for tile coordinates and source images
- Quality control tagging system for edge cases:
  - Tiny boxes (area < 0.0005)  
  - Huge boxes (area > 0.60)
  - Dense ship clusters (5+ ships per tile)
- Embedding computation for visual analysis

## Dataset Statistics

### COCO-Labeled Umbra Chips
- Source GeoTIFFs: 3 (Umbra X-band, GEC format, ~0.5m resolution)
- Chips generated: 459 (640×640px, 64px overlap)
- Annotated chips: 8 (train/valid/test splits)
- Ships annotated: 13
- Training images after augmentation: 60
- Training annotations after augmentation: ~90

### SSDD Benchmark
- Images: 1,160 (TerraSAR-X, RadarSat-2, Sentinel-1)
- Ships: 2,587 instances
- Splits: 928 train / 232 test (+ inshore/offshore)

### Umbra Cross-Domain Test Set
- Chips: 40 (30 annotated, 10 negative)
- Purpose: Cross-domain evaluation (SSDD → Umbra)

## References

- COCO annotation format
- FiftyOne dataset management
- SSDD: Zhang et al., "SAR Ship Detection Dataset", Remote Sensing 2021
- OpenCV and rasterio for image processing
- Ultralytics YOLOv8/v11-OBB for training