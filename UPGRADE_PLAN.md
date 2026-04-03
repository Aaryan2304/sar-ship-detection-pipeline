# SAR Ship Detection Pipeline — v2 OBB Upgrade Plan

## Objective

Upgrade from axis-aligned HBB to oriented bounding box (OBB) ship detection on SAR imagery,
with cross-domain evaluation, ablation studies, geo-referenced inference, and optimized deployment.

---

## Phase 1 — Data Acquisition & Preparation

### 1.1 External Datasets

| Dataset | Ships | Labels | Resolution | Notes |
|---------|-------|--------|------------|-------|
| **SSDD** | ~2,800 | XML / TXT | 1–15m | Coastal, port scenes. Canonical benchmark. |
| **HRSID** | ~36,000 | COCO JSON | 0.5–3m | Higher res, dense ship clusters. |
| **DOTAv1** | ~18,000 ships | 8-pt polygons | 0.3–30m | Multi-class, includes OBB labels. |

Sources: SSDD (`github.com/YoungFish0212/SSDD`), HRSID (`github.com/chaozhong2010/HRSID`),
DOTAv1 (`captain-whu.github.io/DOTA`).

### 1.2 Existing Data

- **8 annotated Umbra chips** (13 ships) — cross-domain test set only, not training
- **459 generated chips** — reproducible via `chip_tiles.py`
- **3 GeoTIFF source files** — source of truth, keep tracked

### 1.3 Label Format Conversion

Problem: SSDD uses Pascal VOC XML with HBB `[x, y, w, h]`. HRSID uses COCO JSON with HBB.
Ultralytics OBB needs `[x_ctr, y_ctr, w, h, angle]`.

New script: `tools/convert_labels.py`
- Parse SSDD XML → extract HBB
- Convert HBB to OBB: estimate rotation from ship bright-pixel geometry within the box
  (fit rotated rectangle to thresholded bright pixels, or use segmentations if available)
- Output Ultralytics YOLO-OBB TXT files per image
- If SSDD HBB→OBB quality is poor, switch to DOTAv1 (native OBB labels, ships included)

### 1.4 Dataset Splits

```
Training: 70% SSDD (converted to OBB)
Validation: 15% SSDD
Test: 15% SSDD + 8 Umbra chips (held-out cross-domain)
```

### 1.5 Reusable Existing Code

| File | Action | Notes |
|------|--------|-------|
| `chip_tiles.py` | Keep | Edge handling works |
| `augment_data.py` | Keep | Disable HSV augment for SAR, keep geometric |
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

Test SSDD-trained model on 8 Umbra chips.

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
| A | Per-chip percentile stretch (current) | Local contrast helps heterogeneous scenes |
| B | Global normalization (dataset mean/std) | Global consistency helps domain transfer |

Train two models, identical except normalization. Compare SSDD mAP and Umbra mAP.

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
python infer_sahi.py --model weights/best.pt \
  --input data/raw/sar_image_1.tif \
  --output results/sar_image_1_detections.json \
  --slice-size 512 --overlap 0.2
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

Expect 2-3x speedup with TensorRT. SAHI multiplies this across 459+ tiles per scene.

---

## Phase 6 — Documentation & Packaging

### 6.1 README Rewrite

- Project overview with OBB focus
- Dataset composition (SSDD + Umbra)
- Quick start: install, download data, train, evaluate
- SAHI inference with geo-referenced output demo
- Benchmark results table
- Architecture diagram

### 6.2 Target File Structure

```
sar-ship-detection-pipeline/
├── pipeline/                    # Existing (chip_tiles, augment, ingest)
├── tools/
│   ├── convert_labels.py        # New: SSDD/HRSID → YOLO-OBB
│   └── export_model.py          # New: ONNX + TensorRT
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
├── UPGRADE_PLAN.md              # This file
└── .gitignore
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

If model checkpoints or dataset subsets push repo past GitHub's limits:

```bash
git lfs install
git lfs track "*.pt" "*.onnx" "*.engine"
```

---

## Timeline

| Phase | Work | Effort |
|-------|------|--------|
| 1. Data | Download SSDD, convert labels | 2-3 days |
| 2. Training | YOLO11-OBB baseline, W&B | 2-3 days |
| 3. Eval | Cross-domain + normalization ablation | 2 days |
| 4. SAHI | Inference pipeline + geo-referencing | 2-3 days |
| 5. Optimize | ONNX/TensorRT + benchmarking | 1-2 days |
| 6. Docs | README, cleanup, packaging | 1-2 days |

**Total: 10-15 days focused work**

---

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| SSDD labels are HBB-only (no rotation) | High | Estimate OBB from bright-pixel geometry; switch to DOTAv1 if quality is poor |
| 4GB VRAM insufficient | Medium | Batch 4-8 + gradient accumulation + AMP |
| Cross-domain mAP < 10% | Medium | Fine-tune on Umbra chips with low LR; report both raw and fine-tuned |
| SAHI inference too slow | Medium | TensorRT FP16 is mandatory, not optional |

## Key Decisions

1. **SSDD vs DOTAv1 for training?** Start with SSDD (SAR-specific, smaller). Switch to
   DOTAv1 if HBB→OBB conversion quality is poor.
2. **Fine-tune on Umbra chips?** Both: report raw SSDD→Umbra gap first, then fine-tune.
3. **Keep geo-referencing?** Yes. This is the differentiator from a generic CV project.

## Out of Scope (for now)

- FastAPI serving layer
- Custom training loop (Ultralytics handles OBB natively)
- Speckle filtering ablation (literature already characterized)
- Web UI (FiftyOne is sufficient)
- Docker container (defer until inference serving)
