# WBCrop Dataset — Attribute Description

> This document provides a detailed description of the two primary geospatial datasets used in this study: **wbcrop_points** (point-level crop sample locations) and **wbcrop_fields** (field-level crop polygon boundaries). Both datasets are stored in the GeoPackage (`.gpkg`) format.

---

## 1. wbcrop_points.gpkg — Point-Level Crop Samples

### 1.1 Overview

| Property                 | Value                      |
| ------------------------ | -------------------------- |
| **Format**               | GeoPackage (`.gpkg`)       |
| **Geometry type**        | Point                      |
| **CRS**                  | EPSG:4326 (WGS 84)         |
| **Number of records**    | 45,616                     |
| **Number of attributes** | 11 (+ geometry)            |
| **Spatial extent (Lat)** | 21.5932°N – 26.9219°N      |
| **Spatial extent (Lon)** | 85.9247°E – 89.8650°E      |
| **Temporal coverage**    | January 2023 – August 2024 |
| **State**                | West Bengal, India         |

### 1.2 Attribute Table

| #   | Attribute         | Data Type         | Description                                                                                              |
| --- | ----------------- | ----------------- | -------------------------------------------------------------------------------------------------------- |
| 1   | `id`              | Integer           | Unique identifier for each sample point (0–45,615).                                                      |
| 2   | `crop`            | String            | Crop type label. One of 18 classes (see Table 1.3).                                                      |
| 3   | `district`        | String            | Administrative district name within West Bengal (21 districts).                                          |
| 4   | `state`           | String            | State name; constant value `west_bengal` for all records.                                                |
| 5   | `sowing`          | String            | Approximate sowing period (e.g., `jul_2023`, `feb_mar_2024`). Value is `NA` if unavailable.              |
| 6   | `harvest`         | String            | Approximate harvest period (e.g., `nov_2023`, `mar_apr_2024`). Value is `NA` if unavailable.             |
| 7   | `collection_date` | String (ISO 8601) | Date of field data collection (`YYYY-MM-DD`). Ranges from `2023-01-10` to `2024-08-21`. 90 unique dates. |
| 8   | `season`          | String            | Cropping season: `kharif`, `rabi`, `boro`, `zaid`, `aus`, or `perennial`.                                |
| 9   | `pheno_stage`     | String            | Phenological stage at the time of collection (e.g., `vegetative`, `reproductive`, `maturity`).           |
| 10  | `latitude`        | Float             | Latitude of the sample location in decimal degrees (WGS 84).                                             |
| 11  | `longitude`       | Float             | Longitude of the sample location in decimal degrees (WGS 84).                                            |
| 12  | `geometry`        | Point             | Point geometry (EPSG:4326).                                                                              |

> **Note:** No null values are present in any attribute column.

### 1.3 Crop Classes and Sample Counts

| Crop Type  | No. of Samples | Season    |
| ---------- | -------------: | --------- |
| Aman Rice  |          6,521 | Kharif    |
| Jute       |          3,360 | Kharif    |
| Potato     |          2,875 | Rabi      |
| Maize      |          2,756 | Rabi      |
| Vegetables |          2,482 | Rabi      |
| Boro Rice  |          2,274 | Boro      |
| Groundnut  |          2,237 | Zaid      |
| Tobacco    |          2,195 | Rabi      |
| Pineapple  |          2,189 | Perennial |
| Tea        |          2,186 | Perennial |
| Sugarcane  |          2,104 | Perennial |
| Banana     |          2,097 | Perennial |
| Wheat      |          2,083 | Rabi      |
| Betel Leaf |          2,081 | Perennial |
| Others     |          2,051 | Rabi      |
| Flower     |          2,046 | Rabi      |
| Mustard    |          2,041 | Rabi      |
| Aus Rice   |          2,038 | Aus       |
| **Total**  |     **45,616** |           |

### 1.4 Season Distribution

| Season    | No. of Samples |
| --------- | -------------: |
| Rabi      |         18,520 |
| Perennial |         10,657 |
| Kharif    |          9,881 |
| Boro      |          2,274 |
| Zaid      |          2,237 |
| Aus       |          2,047 |

### 1.5 Phenological Stage Distribution

| Phenological Stage | No. of Samples |
| ------------------ | -------------: |
| Vegetative         |         24,404 |
| Reproductive       |          8,789 |
| Maturity           |          8,337 |
| Harvest            |          2,329 |
| Grain Filling      |            875 |
| Emergence          |            386 |
| Sowing             |            238 |
| Pod Formation      |            238 |
| Maturation         |             17 |
| Harvesting         |              3 |

### 1.6 District-Wise Distribution

| District           | No. of Samples |
| ------------------ | -------------: |
| Nadia              |          6,463 |
| Pashchim Medinipur |          5,506 |
| Purba Medinipur    |          4,592 |
| Koch Bihar         |          3,792 |
| Murshidabad        |          3,789 |
| Uttar Dinajpur     |          2,937 |
| Darjiling          |          2,418 |
| North 24 Parganas  |          2,331 |
| Birbhum            |          2,053 |
| Bankura            |          1,692 |
| Purba Barddhaman   |          1,323 |
| Maldah             |          1,188 |
| Dakshin Dinajpur   |          1,168 |
| Jalpaiguri         |          1,143 |
| Haora              |          1,043 |
| Hugli              |            912 |
| South 24 Parganas  |            802 |
| Alipurduar         |            742 |
| Jhargram           |            720 |
| Puruliya           |            651 |
| Paschim Barddhaman |            351 |

---

## 2. wbcrop_fields.gpkg — Field-Level Crop Polygons

### 2.1 Overview

| Property                 | Value                      |
| ------------------------ | -------------------------- |
| **Format**               | GeoPackage (`.gpkg`)       |
| **Geometry type**        | Polygon                    |
| **CRS**                  | EPSG:32645 (UTM Zone 45N)  |
| **Number of records**    | 42,476                     |
| **Number of attributes** | 13 (+ geometry)            |
| **Spatial extent (Lat)** | 21.5932°N – 26.9053°N      |
| **Spatial extent (Lon)** | 85.9247°E – 89.8419°E      |
| **Temporal coverage**    | January 2023 – August 2024 |
| **State**                | West Bengal, India         |

### 2.2 Attribute Table

| #   | Attribute         | Data Type         | Description                                                                                                  |
| --- | ----------------- | ----------------- | ------------------------------------------------------------------------------------------------------------ |
| 1   | `id`              | Integer           | Unique identifier for each field polygon; corresponds to `id` in the points dataset.                         |
| 2   | `crop`            | String            | Crop type label. One of 18 classes (same as Section 1.3).                                                    |
| 3   | `district`        | String            | Administrative district name within West Bengal (21 districts).                                              |
| 4   | `state`           | String            | State name; constant value `west_bengal` for all records.                                                    |
| 5   | `sowing`          | String            | Approximate sowing period (e.g., `jul_2023`, `feb_mar_2024`). Value is `NA` if unavailable.                  |
| 6   | `harvest`         | String            | Approximate harvest period (e.g., `nov_2023`, `mar_apr_2024`). Value is `NA` if unavailable.                 |
| 7   | `collection_date` | String (ISO 8601) | Date of field data collection (`YYYY-MM-DD`). Ranges from `2023-01-10` to `2024-08-21`.                      |
| 8   | `season`          | String            | Cropping season: `kharif`, `rabi`, `boro`, `zaid`, `aus`, or `perennial`.                                    |
| 9   | `pheno_stage`     | String            | Phenological stage at the time of collection.                                                                |
| 10  | `latitude`        | Float             | Latitude of the original sample point (WGS 84).                                                              |
| 11  | `longitude`       | Float             | Longitude of the original sample point (WGS 84).                                                             |
| 12  | `area_ha`         | Float             | Area of the segmented field polygon in **hectares**.                                                         |
| 13  | `is_transformed`  | Boolean           | Flag indicating whether the polygon was geometrically transformed during post-processing (`True` / `False`). |
| 14  | `geometry`        | Polygon           | Polygon geometry (EPSG:32645, UTM Zone 45N).                                                                 |

> **Note:** No null values are present in any attribute column.

### 2.3 Field Area Statistics

| Statistic | Value (ha) |
| --------- | ---------: |
| Mean      |     0.1141 |
| Std. Dev. |     0.0741 |
| Min       |     0.0143 |
| 25th %ile |     0.0613 |
| Median    |     0.0900 |
| 75th %ile |     0.1463 |
| Max       |     0.3839 |

### 2.4 Crop Classes and Field Counts

| Crop Type  | No. of Fields | Season    |
| ---------- | ------------: | --------- |
| Aman Rice  |         6,189 | Kharif    |
| Jute       |         3,149 | Kharif    |
| Potato     |         2,624 | Rabi      |
| Maize      |         2,622 | Rabi      |
| Vegetables |         2,341 | Rabi      |
| Pineapple  |         2,155 | Perennial |
| Boro Rice  |         2,131 | Boro      |
| Tobacco    |         2,080 | Rabi      |
| Betel Leaf |         2,018 | Perennial |
| Sugarcane  |         1,994 | Perennial |
| Wheat      |         1,980 | Rabi      |
| Others     |         1,951 | Rabi      |
| Flower     |         1,939 | Rabi      |
| Groundnut  |         1,934 | Zaid      |
| Mustard    |         1,929 | Rabi      |
| Aus Rice   |         1,922 | Aus       |
| Banana     |         1,854 | Perennial |
| Tea        |         1,664 | Perennial |
| **Total**  |    **42,476** |           |

### 2.5 Season Distribution

| Season    | No. of Fields |
| --------- | ------------: |
| Rabi      |        17,457 |
| Perennial |         9,685 |
| Kharif    |         9,338 |
| Boro      |         2,131 |
| Zaid      |         1,934 |
| Aus       |         1,931 |

### 2.6 District-Wise Distribution

| District           | No. of Fields |
| ------------------ | ------------: |
| Nadia              |         6,078 |
| Pashchim Medinipur |         5,201 |
| Purba Medinipur    |         4,179 |
| Koch Bihar         |         3,574 |
| Murshidabad        |         3,572 |
| Uttar Dinajpur     |         2,837 |
| North 24 Parganas  |         2,212 |
| Darjiling          |         2,165 |
| Birbhum            |         1,994 |
| Bankura            |         1,500 |
| Purba Barddhaman   |         1,210 |
| Dakshin Dinajpur   |         1,091 |
| Maldah             |         1,065 |
| Jalpaiguri         |           990 |
| Haora              |           970 |
| Hugli              |           829 |
| South 24 Parganas  |           778 |
| Jhargram           |           702 |
| Puruliya           |           619 |
| Alipurduar         |           601 |
| Paschim Barddhaman |           309 |

---

## 3. Relationship Between the Two Datasets

The **wbcrop_points** dataset contains 45,616 geolocated crop sample points collected across 21 districts of West Bengal, India, between January 2023 and August 2024. Each point records the crop type, cropping season, phenological stage at the time of collection, and approximate sowing/harvest windows.

The **wbcrop_fields** dataset contains 42,476 field-boundary polygons derived from the point locations via image segmentation (SAM). The field polygons share the same thematic attributes as the parent points (linked via `id`) and additionally include:

- **`area_ha`** — the computed field area in hectares (range: 0.014–0.384 ha; median: 0.09 ha), and
- **`is_transformed`** — a boolean flag indicating whether the polygon geometry was post-processed.

The reduction from 45,616 points to 42,476 polygons (a loss of **3,140 samples; ~6.9%**) reflects quality-control filtering applied during segmentation and polygon cleaning.

---

## 4. Data Dictionary Summary

| Attribute         | Points | Fields  | Type     | Description                      |
| ----------------- | :----: | :-----: | -------- | -------------------------------- |
| `id`              |   ✓    |    ✓    | Integer  | Unique sample/field identifier   |
| `crop`            |   ✓    |    ✓    | String   | Crop type (18 classes)           |
| `district`        |   ✓    |    ✓    | String   | District name (21 districts)     |
| `state`           |   ✓    |    ✓    | String   | State name (`west_bengal`)       |
| `sowing`          |   ✓    |    ✓    | String   | Sowing period                    |
| `harvest`         |   ✓    |    ✓    | String   | Harvest period                   |
| `collection_date` |   ✓    |    ✓    | String   | Field collection date (ISO 8601) |
| `season`          |   ✓    |    ✓    | String   | Cropping season (6 classes)      |
| `pheno_stage`     |   ✓    |    ✓    | String   | Phenological stage (10 classes)  |
| `latitude`        |   ✓    |    ✓    | Float    | Sample point latitude (°N)       |
| `longitude`       |   ✓    |    ✓    | Float    | Sample point longitude (°E)      |
| `area_ha`         |        |    ✓    | Float    | Field polygon area (hectares)    |
| `is_transformed`  |        |    ✓    | Boolean  | Geometry transformation flag     |
| `geometry`        | Point  | Polygon | Geometry | Spatial feature                  |
