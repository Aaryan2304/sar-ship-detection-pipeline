# SAR Data

## Source GeoTIFFs

All seven source GeoTIFFs are included in `data/raw/` via Git LFS:

| File | Size | Notes |
|------|------|-------|
| `sar_image_1.tif` | 60 MB | Original set |
| `sar_image_2.tif` | 86 MB | Original set |
| `sar_image_3.tif` | 86 MB | Original set |
| `sar_image_4.tif` | 193 MB | Added during Phase 2 |
| `sar_image_5.tif` | 303 MB | Added during Phase 2 |
| `sar_image_6.tif` | 135 MB | Added during Phase 2 |
| `sar_image_7.tif` | 458 MB | Added during Phase 2 |
| **Total** | **~1.3 GB** | |

**Note:** These files are stored via Git LFS. After cloning, run:
```bash
git lfs pull
```

### Download Additional Imagery

To download additional Umbra SAR imagery:

**Via AWS CLI (no-sign-request for open data):**
```bash
# List available scenes
aws s3 ls s3://umbra-open-data/ --no-sign-request

# Download a specific GEC scene
aws s3 cp s3://umbra-open-data/<path-to-scene>_GEC.tif data/raw/ --no-sign-request
```

**Via STAC API:**
Query the Umbra STAC catalog for GEC products and download the assets. Look for the `GEC` product type.

## Chipping

To generate chips from GeoTIFFs, use the chipping script:
```bash
python tools/chip_sar.py
```

Chips are stored in `data/chips/` (gitignored, regenerate locally).

## Notes

- All source imagery is X-band SAR at ~0.5m resolution
- Files are in GeoTIFF format with `_GEC` (Geocoded Ellipsoid Corrected) processing level
