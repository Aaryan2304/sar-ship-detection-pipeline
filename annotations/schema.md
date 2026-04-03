# Annotation Schema

This document defines the annotation schema used for SAR ship detection.

## Class Definitions

### Ship Detection

- **Class Name**: `ship`
- **Category ID**: 1
- **Supercategory**: `none`

## Annotation Format

Annotations follow the COCO (Common Objects in Context) format with bounding box coordinates in [x_min, y_min, width, height] format.

## Edge Cases and Decisions

### Small Ships
- Ships smaller than 0.0005 in area are flagged as "review_tiny_box"
- These may be false positives or require additional validation

### Large Ships  
- Ships larger than 0.60 in area are flagged as "review_huge_box"
- May indicate mislabeling or requires additional validation

### Dense Ship Clusters
- When a tile contains 5 or more ships, it is tagged as "dense_ships"
- These may require special handling or validation

### Validation Criteria
- Minimum valid tile percentage: 50% (to ensure non-zero data)
- Tile size: 640x640 pixels with 64 pixel overlap for better coverage