# SAR Ship Detection Pipeline

End-to-end SAR ship detection pipeline with oriented bounding boxes (OBB). Trains on the SSDD benchmark (1,160 images, 2,587 ships) and evaluates cross-domain generalization on custom Umbra Space SAR chips.

## What This Is

A complete SAR ship detection pipeline that:

1. **Chips** large GeoTIFFs into ML-ready tiles via sliding windows
2. **Converts** SSDD RBox annotations to YOLO OBB format (verified angle convention)
3. **Trains** OBB detectors (YOLO26-OBB) on the SSDD benchmark
4. **Evaluates** cross-domain generalization (SSDD → Umbra open data)
5. **Augments** COCO-labeled chips with 9 transformations (bbox-aware)
6. **Loads** everything into FiftyOne for QC and visualization

## Results

### Phase 2: Baseline (SSDD In-Domain)

| Metric | Value |
|--------|-------|
| **mAP@50** | 0.9868 |
| **mAP@50:95** | 0.7957 |
| **Precision** | 0.9662 |
| **Recall** | 0.9469 |

Model: YOLO26s-OBB, 50 epochs on SSDD (928 train / 232 val).

### Phase 3: Cross-Domain Evaluation (SSDD → Umbra)

| Variant | Normalization | Umbra mAP@50 | Umbra Recall |
|---------|--------------|-------------|-------------|
| A (baseline) | raw / 255 | 0.1159 | 0.1081 |
| **B** | **per-chip percentile stretch** | **0.1754 (+51%)** | **0.2162 (+100%)** |

Variant B applies a 1st-99th percentile stretch per chip before training. This normalizes intensity distributions between SSDD (narrow dark range) and Umbra (full dynamic range), doubling recall. See `tools/normalize_chips.py` for implementation.

The dominant remaining failure mode is **scale mismatch** — the model predicts undersized boxes because SSDD trains on small ships. This will be addressed in Phase 4 (SAHI inference) and Phase 5 (fine-tuning on Umbra chips).

## Quick Start

```bash
pip install -r requirements.txt
pip install .

# Train baseline model (all augmentations + hyperparams documented in train.py)
python tools/train.py --epochs 50 --batch 8

# Validate best model
python tools/baseline_validate.py

# Convert SSDD labels to YOLO OBB format
python tools/convert_ssdd_to_yolo_obb.py --mode corners

# Verify angle convention
python tools/verify_rbox_angle.py

# Evaluate on Umbra cross-domain test set (with visual overlays)
python tools/evaluate.py --weights runs/baseline/weights/best.pt --data datasets/umbra-test/dataset.yaml

# Generate normalized dataset (per-chip percentile stretch)
python tools/normalize_chips.py --input datasets/SSDD --output datasets/SSDD_norm

# Train on normalized dataset (Variant B)
python tools/train.py --data datasets/SSDD_norm/dataset.yaml --name variant_b_norm
```

## Project Structure

```
sar-ship-detection-pipeline/
├── pipeline/
│   ├── chip_tiles.py           # GeoTIFF → 640x640 tile chipper
│   ├── augment_data.py         # COCO-aware image + bbox augments
│   └── ingest_fiftyone.py      # FiftyOne dataset management + QC
├── tools/
│   ├── train.py                      # Training script (SAR augmentations documented)
│   ├── evaluate.py                   # Cross-domain evaluation + visual overlays
│   ├── baseline_validate.py          # Baseline validation + inference latency
│   ├── normalize_chips.py            # Per-chip percentile stretch preprocessing
│   ├── convert_ssdd_to_yolo_obb.py   # SSDD XML → YOLO OBB conversion
│   └── verify_rbox_angle.py          # Angle convention verification
├── datasets/
│   ├── SSDD/                   # SSDD benchmark (1,160 imgs, 2,587 ships)
│   │   ├── images/, annotations/, labels/
│   │   ├── train.txt, val.txt
│   │   └── dataset.yaml
│   └── umbra-test/             # Cross-domain test set (40 chips)
│       ├── test/images/, test/labels/
│       └── dataset.yaml
├── runs/
│   ├── baseline/               # Training results (plots, weights, args)
│   │   ├── weights/best.pt     # Best checkpoint
│   │   └── results.csv         # Per-epoch metrics
│   └── validation/             # Standalone validation results
├── data/
│   └── raw/                    # Source GeoTIFFs (~500 MB, 7 scenes)
├── annotations/labels/         # COCO chips (8 annotated, 13 ships)
│   ├── train/, valid/, test/, augmented/
├── docs/
│   ├── project_overview.md
│   ├── methodology.md
│   └── usage.md
├── setup.py
├── requirements.txt
├── UPGRADE_PLAN.md
└── README.md
```

## Training Augmentations

See `tools/train.py` for the full configuration. Key SAR-specific choices:

| Augmentation | Value | Rationale |
|--------------|-------|-----------|
| `hsv_h/s/v` | 0.0 | SAR is grayscale — no color transforms |
| `degrees` | 90.0 | Ships at any heading — full rotation coverage |
| `flipud/fliplr` | 0.5 | SAR has no "up" — both flips equally valid |
| `mosaic` | 0.5 | Moderate — helps context without over-augmenting |
| `mixup` | 0.0 | Disabled — SAR intensity is signal, blending hurts |
| `amp` | True | FP16 mixed precision for 4GB VRAM |

## Datasets

### SSDD Benchmark
| Metric | Value |
|--------|-------|
| Images | 1,160 (TerraSAR-X, RadarSat-2, Sentinel-1) |
| Ships | 2,587 instances |
| Splits | 928 train / 232 test (+ inshore/offshore) |
| Labels | Pascal VOC with `rotated_bndbox` (RBox variant) |

### Umbra Cross-Domain Test Set
| Metric | Value |
|--------|-------|
| Chips | 40 (30 annotated, 10 negative) |
| Source | Umbra X-band SAR (GEC, ~0.5m resolution) |
| Purpose | Cross-domain evaluation (SSDD → Umbra) |

## License

MIT
