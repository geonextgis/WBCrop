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
import pandas as pd
import rasterio as rio
from pyproj import Transformer
from rasterio.features import shapes
from rasterio.transform import rowcol
from samgeo import SamGeo3
from scipy.ndimage import binary_fill_holes
from shapely.geometry import shape
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_objects,
)
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


def postprocessing(mask_dir: str, out_dir: str) -> None:
    """
    Post-process binary segmentation masks to remove noise and fill holes.

    Applies a sequence of morphological operations to clean up raw SAM3
    output masks:
        1. Remove small spurious objects (< 2% of image area)
        2. Binary opening  — eliminates salt noise
        3. Binary closing  — fills small gaps between field pixels
        4. Hole filling    — fills enclosed background regions inside fields

    Cleaned masks are saved as single-band uint8 GeoTIFFs with LZW
    compression, preserving the original spatial reference and transform.

    Parameters
    ----------
    mask_dir : str
        Path to the directory containing raw binary mask GeoTIFFs (*.tif).
    out_dir : str
        Path to the output directory where cleaned masks will be saved.
        Created automatically if it does not exist.

    Returns
    -------
    None
        Cleaned masks are written to `out_dir` with the same filename as
        the input.

    Raises
    ------
    FileNotFoundError
        If `mask_dir` does not exist.
    ValueError
        If no .tif files are found in `mask_dir`.
    """
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    os.makedirs(out_dir, exist_ok=True)

    mask_paths = sorted(glob(os.path.join(mask_dir, "*.tif")))
    if not mask_paths:
        raise ValueError(f"No .tif masks found in: {mask_dir}")

    logger.info("Post-processing %d masks from: %s", len(mask_paths), mask_dir)

    skipped = 0
    for path in tqdm(mask_paths, desc="Post-processing masks"):
        basename = os.path.basename(path)

        try:
            # ── Read mask ──
            with rio.open(path, "r") as src:
                mask = src.read(1)
                meta = src.meta.copy()

            mask = mask.astype(bool)

            # ── Remove small patches (< 2% of image area) ──
            min_size = int((mask.shape[0] * mask.shape[1]) * 0.02)
            mask_cleaned = remove_small_objects(mask, min_size=min_size)

            # ── Morphological opening (remove salt noise) ──
            mask_cleaned = binary_opening(mask_cleaned, disk(3))

            # ── Morphological closing (fill small gaps) ──
            mask_cleaned = binary_closing(mask_cleaned, disk(3))

            # ── Fill enclosed holes inside fields ──
            mask_cleaned = binary_fill_holes(mask_cleaned)

            mask_cleaned = mask_cleaned.astype(np.uint8)

            # ── Save cleaned mask ──
            meta.update({"dtype": "uint8", "count": 1, "compress": "lzw"})

            out_path = os.path.join(out_dir, basename)
            with rio.open(out_path, "w", **meta) as dst:
                dst.write(mask_cleaned, 1)

            logger.debug("Saved cleaned mask: %s", out_path)

        except rio.errors.RasterioIOError as e:
            logger.error("Failed to read mask %s: %s — skipping.", basename, e)
            skipped += 1
        except Exception as e:
            logger.error("Unexpected error for mask %s: %s — skipping.", basename, e)
            skipped += 1

    logger.info(
        "Post-processing done. Cleaned: %d, skipped: %d.",
        len(mask_paths) - skipped,
        skipped,
    )


def raster_to_vector(
    mask_dir: str,
    point_path: str,
    out_path: str,
    min_area: float = 50.0,
    simplify_tolerance: float = 0.25,
) -> None:
    """
    Convert cleaned binary raster masks to a single vector GeoPackage.

    For each mask GeoTIFF, the function:
        1. Vectorises the binary mask using rasterio ``shapes``.
        2. Retains only the largest polygon (assumed to be the target field).
        3. Simplifies the polygon geometry to reduce vertex count.
        4. Merges all per-image polygons into one GeoDataFrame.
        5. Joins crop metadata (crop type, district, etc.) from the point file.
        6. Saves the result to a GeoPackage.

    Parameters
    ----------
    mask_dir : str
        Path to the directory containing cleaned binary mask GeoTIFFs (*.tif).
        Each filename must start with the numeric crop sample ID followed by
        an underscore, e.g. ``12345_patch.tif``.
    point_path : str
        Path to the GeoPackage / shapefile containing crop sample points with
        at minimum the columns: ``'id'`` and ``'geometry'``.
    out_path : str
        Full output path for the resulting GeoPackage (e.g. ``fields.gpkg``).
        Parent directory is created automatically if it does not exist.
    min_area : float, optional
        Minimum polygon area (in CRS units) to retain. Polygons smaller than
        this threshold are discarded. Default is 50.0.
    simplify_tolerance : float, optional
        Tolerance passed to ``shapely.geometry.simplify`` to reduce vertex
        count while preserving shape. Default is 0.25.

    Returns
    -------
    None
        The merged GeoDataFrame is written to ``out_path``.

    Raises
    ------
    FileNotFoundError
        If ``mask_dir`` or ``point_path`` does not exist.
    ValueError
        If no .tif masks are found in ``mask_dir``, or if after filtering
        no valid polygons remain.
    """
    # ── Validate inputs ──
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    if not os.path.isfile(point_path):
        raise FileNotFoundError(f"Point file not found: {point_path}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    mask_paths = sorted(glob(os.path.join(mask_dir, "*.tif")))
    if not mask_paths:
        raise ValueError(f"No .tif masks found in: {mask_dir}")

    # ── Load crop sample metadata ──
    logger.info("Loading crop sample points from: %s", point_path)
    crop_samples_gdf = gpd.read_file(point_path)

    logger.info("Vectorising %d masks from: %s", len(mask_paths), mask_dir)

    records = []
    skipped = 0

    for path in tqdm(mask_paths, desc="Raster → vector"):
        basename = os.path.basename(path)
        point_id = int(basename.split("_")[0])

        try:
            with rio.open(path) as src:
                mask = src.read(1)  # uint8 binary mask
                transform = src.transform  # ← fix: was undefined before
                crs = src.crs

            # ── Vectorise foreground pixels (value == 1) ──
            results = [
                {"properties": {"value": v}, "geometry": s}
                for s, v in shapes(mask, transform=transform)
                if v == 1
            ]

            if not results:
                logger.warning("No foreground pixels in mask %s — skipping.", basename)
                skipped += 1
                continue

            # ── Build per-image GeoDataFrame ──
            geoms = [shape(r["geometry"]) for r in results]
            gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
            gdf["area"] = gdf.geometry.area

            # ── Filter small polygons ──
            gdf = gdf[gdf["area"] > min_area]
            if gdf.empty:
                logger.warning(
                    "All polygons below min_area=%.1f for mask %s — skipping.",
                    min_area,
                    basename,
                )
                skipped += 1
                continue

            # ── Keep only the largest polygon (target field) ──
            gdf = gdf.sort_values(by="area", ascending=False).iloc[:1].copy()

            # ── Simplify geometry ──
            gdf["geometry"] = gdf.geometry.simplify(tolerance=simplify_tolerance)
            gdf["id"] = point_id

            records.append(gdf)
            logger.debug(
                "Vectorised mask: %s (area=%.1f)", basename, gdf["area"].iloc[0]
            )

        except rio.errors.RasterioIOError as e:
            logger.error("Failed to read mask %s: %s — skipping.", basename, e)
            skipped += 1
        except Exception as e:
            logger.error("Unexpected error for mask %s: %s — skipping.", basename, e)
            skipped += 1

    if not records:
        raise ValueError("No valid polygons were produced. Check your masks.")

    # ── Concatenate all polygons ──
    wbcrop_polygons = gpd.GeoDataFrame(
        pd.concat(records, ignore_index=True),
        crs=records[0].crs,
    )

    # ── Join crop metadata ──
    wbcrop_polygons = wbcrop_polygons[["id", "area", "geometry"]]
    wbcrop_polygons = pd.merge(
        left=wbcrop_polygons,
        right=crop_samples_gdf.drop(columns="geometry"),
        how="left",
        on="id",
    )
    wbcrop_polygons = wbcrop_polygons.sort_values(by="id").reset_index(drop=True)

    # ── Save ──
    wbcrop_polygons.to_file(out_path)
    logger.info(
        "Saved %d field polygons to: %s (skipped: %d).",
        len(wbcrop_polygons),
        out_path,
        skipped,
    )


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
        help="Output directory for raw masks. Defaults to <data_dir>/raw/high_res_masks.",
    )
    parser.add_argument(
        "--out_dir_cleaned",
        type=str,
        default=None,
        help="Output directory for cleaned masks. Defaults to <data_dir>/raw/high_res_masks_cleaned.",
    )
    parser.add_argument(
        "--out_vector",
        type=str,
        default=None,
        help="Output path for vector GeoPackage. Defaults to <data_dir>/processed/wbcrop_fields.gpkg.",
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
    parser.add_argument(
        "--min_area",
        type=float,
        default=50.0,
        help="Minimum polygon area (CRS units) to retain during vectorisation.",
    )
    parser.add_argument(
        "--simplify_tolerance",
        type=float,
        default=0.25,
        help="Tolerance value to simplify the geometry.",
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
    out_dir_cleaned = args.out_dir_cleaned or os.path.join(
        data_dir, "raw", "high_res_masks_cleaned"
    )
    out_vector = args.out_vector or os.path.join(
        data_dir, "processed", "wbcrop_fields.gpkg"
    )

    logger.info("=== SAM3 Field Segmentation ===")
    logger.info("  Image dir     : %s", image_dir)
    logger.info("  Point path    : %s", point_path)
    logger.info("  Raw masks     : %s", out_dir)
    logger.info("  Cleaned masks : %s", out_dir_cleaned)
    logger.info("  Vector output : %s", out_vector)
    logger.info("  Device        : %s", args.device)
    logger.info("  Min size      : %d px", args.min_size)
    logger.info("  Min area      : %.1f CRS units", args.min_area)
    logger.info("  Simplify tol.    : %.2f", args.simplify_tolerance)

    # ── Step 1: Segmentation ──
    logger.info("=== Step 1 / 3 — Generating masks ===")
    # generate_field_mask(
    #     image_dir=image_dir,
    #     point_path=point_path,
    #     out_dir=out_dir,
    #     device=args.device,
    #     min_size=args.min_size,
    # )

    # ── Step 2: Post-processing ──
    logger.info("=== Step 2 / 3 — Post-processing masks ===")
    postprocessing(
        mask_dir=out_dir,
        out_dir=out_dir_cleaned,
    )

    # ── Step 3: Raster → Vector ──
    logger.info("=== Step 3 / 3 — Raster to vector ===")
    raster_to_vector(
        mask_dir=out_dir_cleaned,
        point_path=point_path,
        out_path=out_vector,
        min_area=args.min_area,
        simplify_tolerance=args.simplify_tolerance,
    )
