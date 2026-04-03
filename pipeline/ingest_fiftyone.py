#!/usr/bin/env python3
"""
Load dataset into FiftyOne for management and visualization.
"""

import argparse
import json
import re
from pathlib import Path

import fiftyone as fo
import fiftyone.brain as fob


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load dataset into FiftyOne for management and visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--aug-coco",
        type=str,
        required=True,
        help="Path to the augmented COCO JSON file"
    )
    
    parser.add_argument(
        "--aug-img-dir",
        type=str,
        required=True,
        help="Directory containing augmented images"
    )
    
    parser.add_argument(
        "--img-root",
        type=str,
        required=True,
        help="Root directory for image metadata lookup"
    )
    
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sar_ships_dataset",
        help="Name of the FiftyOne dataset"
    )
    
    parser.add_argument(
        "--run-embeddings",
        action="store_true",
        help="Run embeddings computation (requires significant time)"
    )
    
    return parser.parse_args()


def load_augmented(aug_coco: Path, aug_img_dir: Path, img_root: Path, 
                   name: str) -> fo.Dataset:
    """
    Load augmented dataset into FiftyOne.
    
    Args:
        aug_coco: Path to the augmented COCO JSON file
        aug_img_dir: Directory containing augmented images  
        img_root: Root directory for image metadata lookup
        name: Name of the FiftyOne dataset
        
    Returns:
        The loaded FiftyOne dataset
    """
    # Augmented train + original valid/test -> one dataset
    if name in fo.list_datasets():
        fo.delete_dataset(name)

    dataset = fo.Dataset(name)
    dataset.persistent = True

    # Train (augmented)
    train_ds = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=str(aug_img_dir),
        labels_path=str(aug_coco),
        label_types=["detections"],
    )
    
    SUFFIXES = {
        "_fliph", "_flipv", "_rot90", "_rot180", "_rot270",
        "_noise", "_bright_up", "_bright_dn", "_contrast"
    }
    
    for s in train_ds:
        stem = Path(s.filepath).stem
        s.tags.append("train")
        s.tags.append("augmented" if any(stem.endswith(x) for x in SUFFIXES) else "original")
        s.save()

    dataset.merge_samples(train_ds)
    print(f"  train: {len(train_ds)} samples")
    fo.delete_dataset(train_ds.name)

    return dataset


# Strip roboflow hash + aug suffix -> original chip name
def _extract_chip_key(filepath: str) -> str:
    """Extract original chip key from augmented filepath."""
    stem = Path(filepath).stem
    SUFFIXES = {
        "_fliph", "_flipv", "_rot90", "_rot180", "_rot270",
        "_noise", "_bright_up", "_bright_dn", "_contrast"
    }
    
    for suf in SUFFIXES:
        if stem.endswith(suf):
            stem = stem[:-len(suf)]
            break
            
    m = re.search(r"(sar_image_\d+_r\d+_c\d+)", stem)
    return m.group(1) + ".png" if m else None


def add_chip_metadata(dataset: fo.Dataset, img_root: Path) -> int:
    """
    Add chip metadata to the dataset.
    
    Args:
        dataset: The FiftyOne dataset
        img_root: Root directory for image metadata lookup
        
    Returns:
        Number of samples updated with metadata
    """
    root = Path(img_root)
    lookup = {}
    
    for mf in root.rglob("*_chips_meta.json"):
        with open(mf) as f:
            for rec in json.load(f):
                lookup[rec["filename"]] = rec
    
    updated = 0
    for sample in dataset:
        key = _extract_chip_key(sample.filepath)
        n_ships = len(sample.ground_truth.detections) if sample.ground_truth else 0
        sample["ship_count"] = n_ships

        if key and key in lookup:
            m = lookup[key]
            sample["source_image"] = Path(m["source_tif"]).name
            sample["tile_row"] = m["row_off"]
            sample["tile_col"] = m["col_off"]
            updated += 1
        sample.save()

    print(f"  Metadata: {updated}/{len(dataset)} samples")
    return updated


def tag_qc(dataset: fo.Dataset) -> None:
    """
    Apply quality control tags to dataset samples.
    
    Args:
        dataset: The FiftyOne dataset
    """
    for sample in dataset:
        dets = sample.ground_truth.detections if sample.ground_truth else []
        tags = []
        
        if not dets:
            tags.append("no_ships")
        else:
            for d in dets:
                _, _, w, h = d.bounding_box
                area = w * h
                if area < 0.0005:
                    tags.append("review_tiny_box")
                    break
                if area > 0.60:
                    tags.append("review_huge_box")
                    break
            if len(dets) >= 5:
                tags.append("dense_ships")
        if tags:
            sample.tags.extend(tags)
            sample.save()

    for tag in ("no_ships", "review_tiny_box", "review_huge_box", "dense_ships"):
        print(f"  {tag}: {len(dataset.match_tags(tag))}")


def print_stats(dataset: fo.Dataset) -> None:
    """
    Print dataset statistics.
    
    Args:
        dataset: The FiftyOne dataset
    """
    total = len(dataset)
    with_ships = len(dataset.match(fo.ViewField("ground_truth.detections").length() > 0))
    n_ships = sum(len(s.ground_truth.detections) for s in dataset if s.ground_truth)
    counts = []
    
    for s in dataset:
        if s.ground_truth and s.ground_truth.detections:
            counts.append(len(s.ground_truth.detections))
    
    print(f"\nTotal: {total} tiles, {with_ships} with ships, {n_ships} total ships")
    
    if counts:
        import statistics
        print(f"Ships/tile: mean={statistics.mean(counts):.2f}, "
              f"median={statistics.median(counts):.1f}, max={max(counts)}")
    
    if "source_image" in dataset.get_field_schema():
        for src in dataset.distinct("source_image"):
            view = dataset.match(fo.ViewField("source_image") == src)
            s = sum(len(s.ground_truth.detections) for s in view if s.ground_truth)
            print(f"  {src}: {len(view)} tiles, {s} ships")


def main():
    """Main function to load and manage dataset."""
    args = parse_args()
    
    print("loading...")
    dataset = load_augmented(
        Path(args.aug_coco),
        Path(args.aug_img_dir),
        Path(args.img_root),
        args.dataset_name
    )
    print(f"  {len(dataset)} samples")
    
    print("metadata...")
    dataset.compute_metadata()
    
    print("chip metadata...")
    add_chip_metadata(dataset, args.img_root)
    
    print("qc tags...")
    tag_qc(dataset)
    
    print_stats(dataset)
    
    if args.run_embeddings:
        print("embeddings...")
        fob.compute_visualization(dataset, brain_key="img_viz", num_workers=0)
        print("  done")
    
    print("app -> http://localhost:5151")
    session = fo.launch_app(dataset, port=5151)
    session.wait()


if __name__ == "__main__":
    main()