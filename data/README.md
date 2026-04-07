# SAR Data

## Source GeoTIFFs

The seven source GeoTIFFs are included in `data/raw/`:
| File | Size |
|------|------|
| `sar_image_1.tif` | ~60 MB |
| `sar_image_2.tif` | ~86 MB |
| `sar_image_3.tif` | ~86 MB |
| `sar_image_4.tif` | ~100 MB |
| `sar_image_5.tif` | ~120 MB |
| `sar_image_6.tif` | ~100 MB |
| `sar_image_7.tif` | ~120 MB |

### Download Your Own

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

### Notes

- All source imagery is X-band SAR at ~0.5m resolution
- Files are in GeoTIFF format with `_GEC` (Geocoded Ellipsoid Corrected) processing level
- Total dataset size: ~500 MB included
