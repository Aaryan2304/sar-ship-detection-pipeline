# SSDD (Ship Detection in SAR Images Dataset) - YOLO OBB Format

## Overview
SSDD dataset adapted for oriented bounding box (OBB) ship detection.

## Source
- Original: RBox_SSDD variant from public benchmark
- Annotations: Pascal VOC XML with `rotated_bndbox` (cx, cy, w, h, theta) + explicit corners
- Converted to YOLO-OBB format via `tools/convert_ssdd_to_yolo_obb.py`

## Statistics
- **Images:** 1160 JPGs (~400x300 to ~600x400 px)
- **Instances:** 2587 ships
- **Classes:** 1 (ship)
- **Splits:** 928 train / 232 test (+ inshore/offshore sub-splits)

## Label Format
- Files: `datasets/SSDD/labels/*.txt` (one per image)
- Format: `class cx cy w h angle` (5 values, all normalized [0,1])
- Angle: radians in range [-pi/4, 3pi/4)
- Convention: standard 2D rotation in image coordinates (verified 0.95px median error)

## Directory Structure
```
datasets/SSDD/
├── dataset.yaml      # Ultralytics dataset config
├── images/           # 1160 JPGs
├── annotations/      # 1160 XML (Pascal VOC with rotated_bndbox)
├── labels/           # 1160 YOLO OBB .txt files
└── splits/           # train.txt, test.txt, test_inshore.txt, test_offshore.txt
```
