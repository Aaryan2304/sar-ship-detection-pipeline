# SAR Ship Detection — Methodology & Pipeline

## Overview

This document describes the methodology and pipeline used for detecting ships in Synthetic Aperture Radar (SAR) imagery.

## Pipeline Description

The SAR ship detection pipeline follows these steps:

1. **Chip Generation**: Convert SAR GeoTIFFs into ML-ready tiles (640x640 pixels with 64 pixel overlap)
2. **Data Augmentation**: Apply transformations to training set images and COCO annotations
3. **Dataset Management**: Integrate FiftyOne for dataset management, metadata, QC tagging, and visualization

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

This pipeline generates the following dataset statistics:

- Source GeoTIFFs: 3 (Umbra X-band, _GEC format)  
- Chips generated: 459 (640×640px, 64px overlap)
- Annotated chips: 8
- Ships annotated: 13
- Training images after augmentation: 60
- Training annotations after augmentation: 100

## Implementation Improvements

### Configuration
- All scripts support command-line arguments for flexible configuration
- No hardcoded paths in the code
- Configurable tile sizes, overlap, and validation thresholds

### Documentation  
- Added proper type hints to all functions
- Comprehensive docstrings explaining functionality
- Clear parameter documentation

## References

This work is based on the following techniques and frameworks:
- COCO annotation format
- FiftyOne dataset management system
- OpenCV and rasterio for image processing