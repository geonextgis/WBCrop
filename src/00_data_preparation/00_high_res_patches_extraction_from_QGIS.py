import os

import geopandas as gpd
from osgeo import gdal, osr
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsMapRendererParallelJob,
    QgsMapSettings,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
)
from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QColor, QImage
from tqdm import tqdm

project = QgsProject.instance()

# -------- USER INPUTS --------
# Name of your basemap layer already loaded in the project
basemap_layer_name = "Google Satellite"

# Output
out_dir = r"C:\HALDER\GITHUB\RESEARCH\WBCrop\data\raw\high_res_patches"
os.makedirs(os.path.dirname(out_dir), exist_ok=True)

# 1. CHANGED: Output CRS set to EPSG:32645 (WGS 84 / UTM zone 45N)
crs_out = QgsCoordinateReferenceSystem("EPSG:32645")

gdf = gpd.read_file(
    "C:\HALDER\GITHUB\RESEARCH\WBCrop\data\processed\wbcrop_points.gpkg"
)

for i, row in tqdm(gdf.iterrows()):
    index = row["id"]
    crop = row["crop"]
    center_x, center_y = row["longitude"], row["latitude"]
    out_path = os.path.join(out_dir, f"{index}_{crop}.tif")

    # Choose a center point in some CRS:
    center_crs = QgsCoordinateReferenceSystem("EPSG:4326")

    # Define extent size (in output CRS units. EPSG:32645 is in meters)
    half_width = 50  # ~100m to left/right
    half_height = 50  # ~100m to up/down

    # Image size in pixels (controls resolution with extent)
    width_px = 512
    height_px = 512

    # -------- FIND BASEMAP LAYER --------
    layers = project.mapLayersByName(basemap_layer_name)
    if not layers:
        raise RuntimeError(f"Layer not found: {basemap_layer_name}")
    basemap = layers[0]

    # -------- TRANSFORM CENTER TO OUTPUT CRS --------
    ct = QgsCoordinateTransform(center_crs, crs_out, project)
    center_out = ct.transform(QgsPointXY(center_x, center_y))

    # -------- BUILD EXTENT AROUND CENTER --------
    extent = QgsRectangle(
        center_out.x() - half_width,
        center_out.y() - half_height,
        center_out.x() + half_width,
        center_out.y() + half_height,
    )

    # -------- MAP SETTINGS --------
    ms = QgsMapSettings()
    ms.setDestinationCrs(crs_out)
    ms.setExtent(extent)
    ms.setOutputSize(QSize(width_px, height_px))
    ms.setBackgroundColor(QColor(255, 255, 255, 0))
    ms.setLayers([basemap])

    # -------- RENDER --------
    job = QgsMapRendererParallelJob(ms)
    job.start()
    job.waitForFinished()

    img: QImage = job.renderedImage()
    # Save the raw image as TIFF first
    img.save(out_path, "TIFF")

    # -------- 2. NEW: ADD GEOREFERENCING TO TIFF USING GDAL --------
    # Open the newly saved image in Update mode
    ds = gdal.Open(out_path, gdal.GA_Update)
    if not ds:
        raise RuntimeError(
            f"Failed to open {out_path} with GDAL to add georeferencing."
        )

    # Calculate pixel width and height in map units
    pixel_width = extent.width() / width_px
    pixel_height = extent.height() / height_px

    # Define the GeoTransform array.
    # Format: (TopLeft X, Pixel Width, X Rotation, TopLeft Y, Y Rotation, Negative Pixel Height)
    geo_transform = (
        extent.xMinimum(),
        pixel_width,
        0,
        extent.yMaximum(),
        0,
        -pixel_height,  # Y must be negative since images draw top-to-bottom
    )
    ds.SetGeoTransform(geo_transform)

    # Assign the EPSG:32645 Projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32645)
    ds.SetProjection(srs.ExportToWkt())

    # Close the dataset to flush changes to disk
    ds = None

    print("Saved georeferenced GeoTIFF:", out_path)
