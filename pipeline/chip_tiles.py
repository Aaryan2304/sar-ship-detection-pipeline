#!/usr/bin/env python3
"""
Chip SAR GeoTIFFs into ML-ready tiles.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Chip SAR GeoTIFFs into ML-ready tiles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input-tif",
        type=str,
        required=True,
        help="Path to the input SAR GeoTIFF file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save chip images and metadata"
    )
    
    parser.add_argument(
        "--tile-size",
        type=int,
        default=640,
        help="Size of each tile in pixels"
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Overlap between tiles in pixels"
    )
    
    parser.add_argument(
        "--min-valid",
        type=float,
        default=0.5,
        help="Minimum valid pixel percentage to keep tile"
    )
    
    return parser.parse_args()


TILE_SIZE = 640
OVERLAP = 64
MIN_VALID = 0.5


def chip_sar_image(tif_path: Path, output_dir: Path, 
                  tile_size: int = TILE_SIZE, overlap: int = OVERLAP, 
                  min_valid: float = MIN_VALID) -> list:
    """
    Chip a geotiff into tiles + write metadata sidecar.
    
    Args:
        tif_path: Path to the input GeoTIFF
        output_dir: Directory to save chips and metadata
        tile_size: Size of each tile in pixels
        overlap: Overlap between tiles in pixels  
        min_valid: Minimum valid pixel percentage to keep tile
        
    Returns:
        List of chip metadata records
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    chips_meta = []

    with rasterio.open(tif_path) as src:
        w, h = src.width, src.height
        step = tile_size - overlap
        
        # TODO: handle edge tiles (right/bottom boundary strips currently skipped)
        for row_off in range(0, h, step):
            for col_off in range(0, w, step):
                # Handle edge tiles by clipping to image boundaries
                r_h = min(tile_size, h - row_off)
                r_w = min(tile_size, w - col_off)
                window = Window(col_off, row_off, r_w, r_h)
                data = src.read(1, window=window).astype(np.float32)

                valid_pct = np.sum(data > 0) / data.size
                if valid_pct < min_valid:
                    continue

                # per-chip percentile stretch
                lo, hi = np.percentile(data[data > 0], (1, 99))
                normed = np.clip((data - lo) / (hi - lo), 0, 1)
                chip_img = Image.fromarray((normed * 255).astype(np.uint8))

                stem = Path(tif_path).stem
                chip_name = f"{stem}_r{row_off}_c{col_off}.png"
                chip_img.save(output_dir / chip_name)

                chips_meta.append({
                    "filename": chip_name,
                    "row_off": row_off,
                    "col_off": col_off,
                    "tile_size": tile_size,
                    "source_tif": str(tif_path),
                })

    meta_path = output_dir / f"{Path(tif_path).stem}_chips_meta.json"
    with open(meta_path, "w") as f:
        json.dump(chips_meta, f, indent=2)
    print(f"  {len(chips_meta)} chips -> {meta_path}")

    return chips_meta


def main():
    """Main function to run the chip generation."""
    args = parse_args()
    
    print(f"Chipping {args.input_tif}...")
    meta = chip_sar_image(
        Path(args.input_tif),
        Path(args.output_dir),
        args.tile_size,
        args.overlap,
        args.min_valid
    )
    print(f"  Done. {len(meta)} chips.")


if __name__ == "__main__":
    main()