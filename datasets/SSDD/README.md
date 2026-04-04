# SSDD Dataset — Rotated Bounding Boxes

SAR Ship Detection Dataset (SSDD) with rotated box (RBox) annotations.

## Source

Downloaded from the official SSDD repository. Contains 1,160 SAR images
with ship annotations in rotated bounding box format.

## Stats

| Split        | Images |
|-------------|--------|
| Train        | 928    |
| Test         | 232    |
| └ Inshore    | 46     |
| └ Offshore   | 186    |
| **Total**    | **1,160** |

- Class: `ship` (single class)
- Annotation format: VOC-style XML with 4-corner rotated bboxes + center/width/height/theta

## Structure

```
SSDD/
├── images/          # 1,160 JPEGs
├── annotations/     # 1,160 VOC XMLs (1:1 with images)
└── splits/
    ├── train.txt            # 928 image IDs
    ├── test.txt             # 232 image IDs
    ├── test_inshore.txt     # 46 test images (coastal/harbor)
    └── test_offshore.txt    # 186 test images (open water)
```

## Annotation Format

Each XML contains a `<rotated_bndbox>` block with:
- 4 corners: `(x1,y1), (x2,y2), (x3,y3), (x4,y4)`
- Center: `(rotated_bbox_cx, rotated_bbox_cy)`
- Dimensions: `rotated_bbox_w`, `rotated_bbox_h`
- Orientation: `rotated_bbox_theta` (degrees)

Images are 3-channel JPEGs (SAR data converted to RGB for visualization).
Actual SAR pixel values are NOT preserved — this is a processed benchmark.

## License

Follow the original SSDD dataset license terms. Check the source repository
for usage restrictions.
