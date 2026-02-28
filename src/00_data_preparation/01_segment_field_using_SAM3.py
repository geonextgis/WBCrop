"""
Field Segmentation using SAM3 (Segment Anything Model 3)

This script segments agricultural field boundaries from high-resolution
satellite/aerial imagery using point prompts derived from crop sample locations.

Usage:
    python 01_segment_field_using_SAM3.py [OPTIONS]

Example:
    python 01_segment_field_using_SAM3.py \
        --data_dir /beegfs/halder/GITHUB/RESEARCH/WBCrop/data \
        --device cuda \
        --min_size 100
"""

import argparse
import logging
import os
from glob import glob

import geopandas as gpd
import numpy as np
import rasterio as rio
from pyproj import Transformer
from rasterio.transform import rowcol
from samgeo import SamGeo3
from tqdm import tqdm

# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────
def generate_field_mask(
    image_dir: str,
    point_path: str,
    out_dir: str,
    device: str = "cuda",
    min_size: int = 100,
) -> None:
    """
    Generate field segmentation masks for a directory of images using SAM3.

    For each image, the crop sample point (lat/lon) is projected into the
    image coordinate system and used as a point prompt to segment the
    corresponding agricultural field boundary.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing GeoTIFF images (*.tif).
    point_path : str
        Path to the GeoPackage (.gpkg) or shapefile containing crop sample
        points with columns: 'id', 'latitude', 'longitude'.
    out_dir : str
        Output directory where segmentation masks will be saved.
    device : str, optional
        Compute device for SAM3 inference ('cuda' or 'cpu'). Default is 'cuda'.
    min_size : int, optional
        Minimum mask size (in pixels) to retain during segmentation.
        Default is 100.

    Returns
    -------
    None
        Masks are saved as GeoTIFF files in `out_dir`.

    Raises
    ------
    FileNotFoundError
        If `image_dir` or `point_path` does not exist.
    ValueError
        If no .tif images are found in `image_dir`.
    """
    # ── Validate inputs ──
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.isfile(point_path):
        raise FileNotFoundError(f"Point file not found: {point_path}")

    os.makedirs(out_dir, exist_ok=True)

    # ── Load data ──
    logger.info("Loading crop sample points from: %s", point_path)
    crop_samples_gdf = gpd.read_file(point_path)

    image_paths = sorted(glob(os.path.join(image_dir, "*.tif")))
    if not image_paths:
        raise ValueError(f"No .tif images found in: {image_dir}")
    logger.info("Found %d images to process.", len(image_paths))

    # ── Initialize SAM3 ──
    logger.info("Initializing SAM3 on device: %s", device)
    sam3 = SamGeo3(
        backend="meta",
        device=device,
        checkpoint_path=None,
        load_from_HF=True,
        enable_inst_interactivity=True,
    )

    # ── Process each image ──
    skipped = 0
    for image_path in tqdm(image_paths, desc="Segmenting fields"):
        basename = os.path.basename(image_path)
        image_id = int(basename.split("_")[0])

        match = crop_samples_gdf.loc[crop_samples_gdf["id"] == image_id]
        if match.empty:
            logger.warning("No crop point found for image ID %d — skipping.", image_id)
            skipped += 1
            continue

        try:
            point_info = match.iloc[0]
            lat = float(point_info["latitude"])
            lon = float(point_info["longitude"])

            # ── Project lat/lon → image pixel (row, col) ──
            with rio.open(image_path, "r") as src:
                crs = src.crs
                transform = src.transform

            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            x_proj, y_proj = transformer.transform(lon, lat)
            row, col = rowcol(transform, x_proj, y_proj)
            point_coords = np.array([[int(col), int(row)]])

            # ── Run SAM3 segmentation ──
            sam3.set_image(image_path)
            sam3.generate_masks_by_points(point_coords, min_size=min_size)

            # ── Save mask ──
            out_path = os.path.join(out_dir, basename)
            sam3.save_masks(output=out_path, unique=False)
            logger.debug("Mask saved: %s", out_path)

        except rio.errors.RasterioIOError as e:
            logger.error("Failed to read image %s: %s — skipping.", basename, e)
            skipped += 1
        except Exception as e:
            logger.error("Unexpected error for image %s: %s — skipping.", basename, e)
            skipped += 1


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Segment agricultural fields using SAM3 with point prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/beegfs/halder/GITHUB/RESEARCH/WBCrop/data",
        help="Root data directory.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help=(
            "Path to image directory. Defaults to "
            "<data_dir>/raw/high_res_patches/high_res_patches."
        ),
    )
    parser.add_argument(
        "--point_path",
        type=str,
        default=None,
        help=(
            "Path to crop sample GeoPackage. Defaults to "
            "<data_dir>/processed/wbcrop_points.gpkg."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for masks. Defaults to <data_dir>/raw/high_res_masks.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device for SAM3.",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=100,
        help="Minimum mask size in pixels.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    data_dir = args.data_dir
    image_dir = args.image_dir or os.path.join(
        data_dir, "raw", "high_res_patches", "high_res_patches"
    )
    point_path = args.point_path or os.path.join(
        data_dir, "processed", "wbcrop_points.gpkg"
    )
    out_dir = args.out_dir or os.path.join(data_dir, "raw", "high_res_masks")

    logger.info("=== SAM3 Field Segmentation ===")
    logger.info("  Image dir  : %s", image_dir)
    logger.info("  Point path : %s", point_path)
    logger.info("  Output dir : %s", out_dir)
    logger.info("  Device     : %s", args.device)
    logger.info("  Min size   : %d px", args.min_size)

    generate_field_mask(
        image_dir=image_dir,
        point_path=point_path,
        out_dir=out_dir,
        device=args.device,
        min_size=args.min_size,
    )
