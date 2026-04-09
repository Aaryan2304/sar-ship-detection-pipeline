# SAR Ship Detection Pipeline — v2 OBB Upgrade Plan

## Objective

Upgrade from axis-aligned HBB to oriented bounding box (OBB) ship detection on SAR imagery,
with cross-domain evaluation, ablation studies, geo-referenced inference, and optimized deployment.

---

## Phase 1 — Data Acquisition & Preparation

### 1.1 SSDD (Primary and sole training dataset)

| Variant | Ships | Labels | Resolution |
|---------|-------|--------|------------|
| BBox-SSDD | 2,456 | Axis-aligned `[x_min, y_min, x_max, y_max]` | 1–15m |
| **RBox-SSDD** | 2,456 | Rotated `[cx, cy, w, h, angle]` | 1–15m |
| PSeg-SSDD | 2,456 | Polygon segmentation | 1–15m |

**Download RBox-SSDD** (from the official release by the original authors):

- Google Drive: `https://drive.google.com/file/d/1glNJUGotrbEyk43twwB9556AdngJsynZ/view?usp=sharing`
- Baidu Pan: `https://pan.baidu.com/s/1Lpg28ZvMSgNXq00abHMZ5Q` password: `2021`
- Paper: Zhang et al., "SAR Ship Detection Dataset (SSDD): Official Release and Comprehensive Data Analysis", Remote Sensing 2021
- GitHub: `github.com/TianwenZhang0825/Official-SSDD`

1,160 SAR images from TerraSAR-X, RadarSat-2, Sentinel-1. Coastal, port, inshore scenes.

We use **RBox-SSDD** — native rotated box labels, no HBB→OBB conversion needed.

Why SSDD-only (not HRSID, CAESAR, or DOTAv1):
- SSDD is SAR-specific with native OBB labels (RBox-SSDD variant)
- HRSID/CAESAR are HBB-only — same conversion problem, more data, slower iterations
- DOTAv1 is optical/aerial imagery — wrong sensor entirely
- 2,456 ships is sufficient for an OBB detector on single-class ship detection

### 1.2 Existing Data

- **40 annotated Umbra chips** (30 positive + 10 negative) — cross-domain test set
- **7 GeoTIFF source files** — 3 original + 4 newly downloaded

### 1.3 Label Format Conversion

RBox-SSDD uses rotated boxes but likely in a different coordinate convention than Ultralytics.
New script: `tools/convert_labels.py`

- Parse RBox-SSDD labels (XML for RBox variant)
- Verify label format and angle convention (check the MDPI paper §3.2 for RBox parameterization)
- Convert to Ultralytics YOLO-OBB TXT format: `[class_id, x_ctr, y_ctr, w, h, angle]`
- Validate by overlaying converted labels on images (spot check 50+ images)

### 1.4 Dataset Splits

```
Training: RBox-SSDD train split (928 images)
Test:     RBox-SSDD test split (232 images) + 40 Umbra chips (held-out cross-domain)
```

If the official release provides a standard split, use it for comparability with literature.

### 1.5 Reusable Existing Code

| File | Action | Notes |
|------|--------|-------|
| `chip_tiles.py` | Keep | Edge handling works |
| `augment_data.py` | Keep | Disable HSV augment for SAR, keep geometric (rotation, flip, mosaic) |
| `ingest_fiftyone.py` | Update | Support OBB visualization |

---

## Phase 2 — Baseline Model Training

### 2.1 Model

YOLO11-OBB via Ultralytics (`yolo11s-obb.pt`). Fallback: `yolov8s-obb.pt`.

### 2.2 Training Config

```yaml
model: yolo11s-obb.pt
imgsz: 640
batch: 8
epochs: 100
optimizer: SGD
lr0: 0.01
lrf: 0.01
augmentations:
  mosaic: 1.0
  mixup: 0.05
  flipud: 0.5
  fliplr: 0.5
  degrees: 90.0
  hsv_h: 0.0
  hsv_s: 0.0
  hsv_v: 0.0
```

SAR is grayscale — disable HSV. Focus on geometric transforms.

### 2.3 Hardware

RTX 3050 Ti, 4GB VRAM. Batch 8 is likely the ceiling for 640px training. Use AMP.
If OOM: batch 4 + gradient accumulation 4.

### 2.4 Experiment Tracking

Weights & Biases (free tier). Log mAP@50, mAP@50:95, loss curves, hyperparams,
best checkpoint artifact.

---

## Phase 3 — Evaluation & Ablation Studies

### 3.1 Baseline Metrics (SSDD Test Set)

mAP@50, mAP@50:95, precision-recall curves, false positive/negative analysis,
inference speed per image.

### 3.2 Cross-Domain Evaluation

Test SSDD-trained model on 40 Umbra chips (30 annotated, 10 negative).

Expect significant mAP drop. SSDD is coastal/port imagery; Umbra is open-ocean,
different sensor, different resolution. Report:
- mAP@50 on Umbra chips
- Visual overlay of predictions on source GeoTIFF chips
- False positive analysis (wave clutter, rocks, SAR artifacts)
- False negative analysis (small ships, specific orientations)

This is the metric that matters for real deployment.

### 3.3 Ablation: Normalization Strategy

| Variant | Method | Hypothesis |
|---------|--------|------------|
| A | Raw pixels / 255 (Ultralytics default) | Baseline — no explicit normalization |
| B | Per-chip 1st-99th percentile stretch | Normalize intensity across sensors/scenes |

**Results (2026-04-09):**

| Metric | A: raw/255 | B: p-stretch | Delta |
|--------|-----------|-------------|-------|
| SSDD mAP@50 | 0.9868 | 0.9801 | -0.67% |
| SSDD mAP@50:95 | 0.7957 | 0.6990 | -9.67% |
| **Umbra mAP@50** | 0.1159 | **0.1754** | **+51.3%** |
| Umbra mAP@50:95 | 0.0548 | 0.0593 | +8.2% |
| Umbra Precision | 0.2705 | 0.2926 | +8.2% |
| **Umbra Recall** | 0.1081 | **0.2162** | **+100%** |
| Images w/ preds (Umbra) | 22/40 | 34/40 | +55% |

Per-chip percentile stretch significantly improves cross-domain recall (+100%) by
normalizing intensity distributions between SSDD and Umbra. However, mAP@50 remains
low (0.175) due to scale mismatch — the model predicts undersized boxes because SSDD
trains on small ships. The dominant remaining failure mode is scale, not intensity.

Scripts: `tools/normalize_chips.py` (preprocessing), `tools/evaluate.py` (eval + overlays)
Overlays: `outputs/umbra_eval/overlays/` (Variant A), `outputs/umbra_eval/variant_b_overlays/` (Variant B)

Skip speckle filtering — Lee filtering vs raw has been exhaustively characterized
in the literature (2-5% mAP improvement, scene-dependent). The normalization ablation
is more novel and directly interacts with the cross-domain gap.

---

## Phase 4 — SAHI Inference Pipeline

New file: `infer_sahi.py`

Slicing Aided Hyper Inference: slice full GeoTIFF into overlapping tiles, run OBB
detection on each, full-frame NMS merge.

```python
from sahi.predict import get_sliced_prediction

result = get_sliced_prediction(
    image="data/raw/sar_image_1.tif",
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    model=model,
    postprocess_type="GREEDYNMM",
    match_metric="IOS",
    match_threshold=0.5,
)
```

### Geo-Referenced Output

Use `chip_tiles.py` tile-to-GeoTIFF coordinate mapping to convert pixel detections
to lat/lon. OBB angle maps to vessel heading.

Output: JSON + optional GeoJSON for QGIS import.

```json
{
  "detections": [{
    "confidence": 0.94,
    "bbox_obb": [x1,y1,x2,y2,x3,y3,x4,y4],
    "heading_degrees": 245,
    "geo": {"latitude": 37.7749, "longitude": -122.4194},
    "source_chip": "sar_image_1_r2880_c2880"
  }]
}
```

CLI:

```bash
python infer_sahi.py --model weights/best.pt   --input data/raw/sar_image_1.tif   --output results/sar_image_1_detections.json   --slice-size 512 --overlap 0.2
```

---

## Phase 5 — Inference Optimization

### 5.1 Export Pipeline

```python
from ultralytics import YOLO
model = YOLO("weights/best.pt")
model.export(format="onnx", imgsz=640, half=True, simplify=True)
```

```bash
trtexec --onnx=weights/best.onnx --fp16 --saveEngine=weights/best.engine --buildOnly
```

SM 86 (3050 Ti) fully supported in TensorRT 10.x.

### 5.2 Benchmarking

| Metric | PyTorch | ONNX | TensorRT FP16 |
|--------|---------|------|---------------|
| Per-chip latency (ms) | TBD | TBD | TBD |
| Full-scene w/ SAHI (s) | TBD | TBD | TBD |
| VRAM peak (MB) | TBD | TBD | TBD |
| mAP@50 delta | baseline | ~0% | ~0-1% |

New file: `tools/export_model.py` — handles PyTorch→ONNX→TensorRT + output verification.

Expect 2-3x speedup with TensorRT. SAHI multiplies this across hundreds of tiles per scene.

---

## Phase 6 — Documentation & Packaging

### 6.1 README Rewrite

- Project overview with OBB focus
- Dataset: RBox-SSDD (SAR-specific, native OBB labels) + Umbra cross-domain test
- Quick start: install, download data, train, evaluate
- SAHI inference with geo-referenced output demo
- Benchmark results table
- Architecture diagram

### 6.2 Target File Structure

```
sar-ship-detection-pipeline/
├── pipeline/                    # Existing (chip_tiles, augment, ingest)
├── tools/
│   └── convert_labels.py        # New: RBox-SSDD → YOLO-OBB
├── infer_sahi.py                # New: SAHI inference + geo-ref
├── evaluate.py                  # New: cross-domain evaluation
├── configs/default.yaml         # Training config
├── data/raw/                    # GeoTIFFs (tracked)
├── data/chips/                  # PNG chips (tracked)
├── annotations/                 # COCO annotations (Umbra chips)
├── assets/screenshots/          # UI screenshots
├── requirements.txt
├── setup.py
├── README.md
└── UPGRADE_PLAN.md              # This file
```

### 6.3 Updated Requirements

```
# Core
numpy>=1.24.0
rasterio>=1.3.0
Pillow>=10.0.0
fiftyone>=0.23.0
fiftyone-brain>=0.13.0

# v2 additions
ultralytics>=8.2.0
sahi>=0.11.0
opencv-python>=4.8.0
onnx>=1.14.0
onnxruntime-gpu>=1.16.0
shapely>=2.0.0
pyproj>=3.0.0
wandb>=0.16.0
matplotlib>=3.7.0
```

### 6.4 setup.py entry_points

```python
entry_points={
    "console_scripts": [
        "chip-tiles=pipeline.chip_tiles:main",
        "augment-data=pipeline.augment_data:main",
        "ingest-fiftyone=pipeline.ingest_fiftyone:main",
        "infer-sahi=infer_sahi:main",
        "convert-labels=tools.convert_labels:main",
    ],
}
```

### 6.5 Git LFS (if needed)

If model checkpoints push repo past GitHub's limits:

```bash
git lfs install
git lfs track "*.pt" "*.onnx" "*.engine"
```

---

## Timeline

| Phase | Work | Effort |
|-------|------|--------|
| 1. Data | Download RBox-SSDD, convert labels to YOLO-OBB | 1 day |
| 2. Training | YOLO11-OBB baseline, W&B | 2-3 days |
| 3. Eval | Cross-domain + normalization ablation | 2 days |
| 4. SAHI | Inference pipeline + geo-referencing | 2-3 days |
| 5. Optimize | ONNX/TensorRT + benchmarking | 1-2 days |
| 6. Docs | README, cleanup, packaging | 1-2 days |

**Total: 9-12 days focused work**

---

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| RBox-SSDD angle convention differs from Ultralytics | High | Verify by visual overlay; the convert script must be validated before training |
| RBox-SSDD download link inaccessible | Medium | Mirror via Kaggle (SSDD is also available there), contact authors |
| 4GB VRAM insufficient | Medium | Batch 4-8 + gradient accumulation + AMP |
| Cross-domain mAP < 10% | Medium | Fine-tune on Umbra chips with low LR; report both raw and fine-tuned |
| SAHI inference too slow | Medium | TensorRT FP16 is mandatory, not optional |

## Key Decisions

1. **SSDD-only for training.** No HRSID, no CAESAR, no DOTAv1. SSDD is SAR-specific
   with native OBB labels (RBox-SSDD). Adding more data doesn't solve the harder
   problems (cross-domain gap, geo-referencing, optimization).
2. **Fine-tune on Umbra chips?** Both: report raw SSDD→Umbra gap first, then fine-tune.
3. **Keep geo-referencing?** Yes. This is the differentiator from a generic CV project.

## Out of Scope (for now)

- FastAPI serving layer
- Custom training loop (Ultralytics handles OBB natively)
- Speckle filtering ablation (literature already characterized)
- Web UI (FiftyOne is sufficient)
- Docker container (defer until inference serving)
- SL-SSDD sea-land masks (interesting but not core — consider after v2 is done)

