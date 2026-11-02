#!/usr/bin/env python
"""Fetch the CEOS pseudo-invariant calibration sites (PICS) from Earth Engine.

For every site the script downloads

* one cloud-free **Landsat 5 MSS L1TP** scene (Collection 2, tier 2: the ESA
  tier-1 archive does not cover these deserts), converted to TOA reflectance
  with the MTL coefficients carried by the image properties, plus the
  ``QA_PIXEL`` / ``QA_RADSAT`` bands; and
* one cloud-free **Sentinel-2A L1C** acquisition (``COPERNICUS/S2_HARMONIZED``),
  13 bands as uint16 reflectance x 10000, plus the s2cloudless probability and
  ``QA60`` masks.

Both are fetched on the *same* 15 km UTM grid, so the S2 -> MSS comparison needs
no reprojection: the MSS grid is 250 x 250 at 60 m and the S2 grid is
1500 x 1500 at 10 m, sharing the top-left corner.

All metadata (IDs, dates, sun angles, cloud fractions, MTL coefficients) is
written to ``<site>/metadata.json``.

Usage
-----
    python scripts/fetch_pics.py                       # all sites
    python scripts/fetch_pics.py --sites libya4_ceos   # one site
    python scripts/fetch_pics.py --out /tmp/pics --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.corrections.landsat.metadata import MSSCalibration  # noqa: E402
from satharmony.corrections.landsat.mss import dn_to_toa  # noqa: E402

DEFAULT_OUT = Path("/data/databases/CLOUD_MSS/MSS_S2/pics")

#: Half side of the PICS window: 7.5 km -> 15 km x 15 km.
HALF_KM = 7.5
MSS_GSD = 60.0
S2_GSD = 10.0
S2_BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B8A")

#: L5 MSS radiance rescaling from the MTL.  A handful of late L1TP scenes
#: (e.g. Death Valley 2012-10-26) carry bands 1-3 but omit the band-4
#: properties while still shipping a B4 raster; the standard L5 values are
#: the fallback, not a silent zero.
L5_RADIANCE_MULT = (0.88504, 0.66024, 0.55866, 0.46654)
L5_RADIANCE_ADD = (1.51496, 2.03976, 4.34134, 1.03346)

#: ``role="calibration"`` sites are the stable CEOS/desert targets used for the
#: SBAF constants; ``role="validation"`` sites are deliberately difficult
#: surfaces (vegetation, water, changing playas) used only to test the
#: extrapolation.  ``sun_max`` is 50 deg for bright targets, where a higher sun
#: would clip the 8-bit MSS, and 65 deg for dark targets, where it does not.
SITES: dict[str, dict[str, object]] = {
    # CEOS reference deserts (calibration).
    "libya4_ceos": {
        "lat": 28.55,
        "lon": 23.39,
        "label": "Libya-4 (CEOS)",
        "note": "CEOS reference PICS; the previous `libya4` pair was at 23.07 N, 22.93 E.",
    },
    "libya1": {"lat": 24.95, "lon": 13.38, "label": "Libya-1", "note": "CEOS PICS, Murzuq sand."},
    "libya2": {"lat": 27.0, "lon": 15.0, "label": "Libya-2", "note": "Saharan hamada."},
    "algeria3": {"lat": 30.0, "lon": 7.0, "label": "Algeria-3", "note": "Saharan bedrock/reg."},
    "algeria5": {"lat": 31.02, "lon": 2.87, "label": "Algeria-5", "note": "CEOS PICS, El Goléa reg."},
    "algeria6": {"lat": 29.0, "lon": 0.0, "label": "Algeria-6", "note": "Western Erg sand."},
    "mauritania1": {"lat": 19.38, "lon": -9.2, "label": "Mauritania-1", "note": "CEOS PICS, Adrar reg."},
    "mauritania2": {"lat": 20.85, "lon": -8.78, "label": "Mauritania-2", "note": "CEOS PICS, Tasiast."},
    "niger1": {"lat": 21.0, "lon": 10.0, "label": "Niger-1", "note": "Saharan reg."},
    "niger2": {"lat": 21.74, "lon": 9.35, "label": "Niger-2", "note": "CEOS PICS, Aïr foreland."},
    "sudan1": {"lat": 20.13, "lon": 29.42, "label": "Sudan-1", "note": "CEOS PICS, Nubian Desert."},
    "sudan2": {"lat": 18.0, "lon": 33.0, "label": "Sudan-2", "note": "Bayuda volcanic desert."},
    "egypt1": {"lat": 26.0, "lon": 30.0, "label": "Egypt-1", "note": "Western Desert limestone."},
    "egypt2": {"lat": 25.0, "lon": 27.0, "label": "Egypt-2", "note": "Western Desert sand sheet."},
    "arabia1": {"lat": 19.79, "lon": 46.8, "label": "Arabia-1", "note": "CEOS PICS, Najd plateau."},
    "arabia2": {"lat": 20.0, "lon": 49.0, "label": "Arabia-2", "note": "Rub' al Khali sand."},
    "arabia3": {"lat": 24.0, "lon": 45.0, "label": "Arabia-3", "note": "Arabian interior."},
    "oman1": {"lat": 21.0, "lon": 56.0, "label": "Oman-1", "note": "Omani interior gravel."},
    "chad1": {"lat": 21.0, "lon": 18.0, "label": "Chad-1", "note": "Tibesti foreland."},
    "morocco1": {"lat": 31.0, "lon": -4.0, "label": "Morocco-1", "note": "Anti-Atlas desert."},
    "kalahari": {"lat": 23.0, "lon": 21.0, "label": "Kalahari", "note": "Kalahari sand, responds to rain.", "role": "validation", "surface": "savanna"},
    "namib": {"lat": 24.0, "lon": 15.0, "label": "Namib", "note": "Namib gravel plains."},
    "sahel": {"lat": 14.0, "lon": 0.0, "label": "Sahel", "note": "Sahelian steppe, seasonal.", "role": "validation", "surface": "savanna"},
    # Sand seas and salt flats (calibration, bright).
    "thar": {"lat": 27.0, "lon": 71.0, "label": "Thar", "note": "Thar Desert sand."},
    "thar2": {"lat": 26.0, "lon": 70.0, "label": "Thar-2", "note": "Thar Desert, Pakistan."},
    "gobi": {"lat": 42.0, "lon": 105.0, "label": "Gobi", "note": "Gobi gravel desert."},
    "taklamakan": {"lat": 39.0, "lon": 83.0, "label": "Taklamakan", "note": "Taklamakan sand sea."},
    "karakum": {"lat": 39.0, "lon": 59.0, "label": "Karakum", "note": "Karakum sand."},
    "kyzylkum": {"lat": 42.0, "lon": 64.0, "label": "Kyzylkum", "note": "Kyzylkum sand."},
    "lut": {"lat": 31.0, "lon": 58.0, "label": "Dasht-e Lut", "note": "Lut Desert, darkest/brightest extremes."},
    "kavir": {"lat": 34.5, "lon": 54.0, "label": "Dasht-e Kavir", "note": "Kavir salt desert."},
    "sonora": {"lat": 30.5, "lon": -112.5, "label": "Sonora", "note": "Sonoran Desert."},
    "mojave": {"lat": 35.0, "lon": -116.0, "label": "Mojave", "note": "Mojave Desert."},
    "chihuahua": {"lat": 30.0, "lon": -106.0, "label": "Chihuahua", "note": "Chihuahuan Desert."},
    "death_valley": {"lat": 36.25, "lon": -116.85, "label": "Death Valley", "note": "Salt pan, very bright."},
    "white_sands": {"lat": 32.9, "lon": -106.3, "label": "White Sands", "note": "Gypsum, extreme brightness; SZA mismatch 9.5 deg -> validation only.", "role": "validation", "surface": "salt_flat"},
    "bonneville": {"lat": 40.8, "lon": -113.5, "label": "Bonneville", "note": "Salt flat, bright."},
    "uyuni": {"lat": -20.13, "lon": -67.49, "label": "Uyuni", "note": "Salar, very bright; southern hemisphere."},
    "coipasa": {"lat": -19.4, "lon": -68.2, "label": "Coipasa", "note": "Salar, bright; SZA mismatch 13.5 deg -> validation only.", "role": "validation", "surface": "salt_lake"},
    "atacama_salar": {"lat": -23.5, "lon": -68.25, "label": "Atacama salar", "note": "Salar, stable core."},
    "lake_eyre": {"lat": -28.4, "lon": 137.3, "label": "Lake Eyre", "note": "Salt lake, changes with floods.", "role": "validation", "surface": "salt_lake"},
    "simpson": {"lat": -25.0, "lon": 137.0, "label": "Simpson", "note": "Simpson Desert."},
    "great_sandy": {"lat": -22.0, "lon": 125.0, "label": "Great Sandy", "note": "Great Sandy Desert."},
    "gibson": {"lat": -24.0, "lon": 125.0, "label": "Gibson", "note": "Gibson Desert."},
    "nullarbor": {"lat": -31.0, "lon": 128.0, "label": "Nullarbor", "note": "Limestone plain."},
    "tanami": {"lat": -20.0, "lon": 130.0, "label": "Tanami", "note": "Tanami Desert."},
    # Dark and vegetated surfaces (validation).
    "congo": {"lat": 0.5, "lon": 23.5, "label": "Congo", "note": "Intact rainforest.", "role": "validation", "surface": "forest", "sun_max": 65.0},
    "amazon": {"lat": 3.0, "lon": -60.0, "label": "Amazon", "note": "Intact rainforest.", "role": "validation", "surface": "forest", "sun_max": 65.0},
    "borneo": {"lat": 0.5, "lon": 114.0, "label": "Borneo", "note": "Intact rainforest.", "role": "validation", "surface": "forest", "sun_max": 65.0},
    "cerrado": {"lat": -15.0, "lon": -47.0, "label": "Cerrado", "note": "Tropical savanna.", "role": "validation", "surface": "savanna", "sun_max": 65.0},
    "pampa": {"lat": -35.0, "lon": -64.0, "label": "Pampa", "note": "Cropland, changes every season.", "role": "validation", "surface": "crop", "sun_max": 65.0},
    "pantanal": {"lat": -17.0, "lon": -57.0, "label": "Pantanal", "note": "Wetland, floods seasonally.", "role": "validation", "surface": "wetland", "sun_max": 65.0},
    "okavango": {"lat": -19.3, "lon": 22.9, "label": "Okavango", "note": "Inland delta, seasonal flood.", "role": "validation", "surface": "wetland", "sun_max": 65.0},
    "deccan": {"lat": 18.0, "lon": 76.0, "label": "Deccan", "note": "Basalt plateau, cropland.", "role": "validation", "surface": "crop", "sun_max": 65.0},
    "titicaca": {"lat": -15.8, "lon": -69.4, "label": "Titicaca", "note": "Deep high-altitude lake.", "role": "validation", "surface": "water", "sun_max": 65.0},
    "victoria": {"lat": -1.0, "lon": 33.0, "label": "Victoria", "note": "Large tropical lake.", "role": "validation", "surface": "water", "sun_max": 65.0},
    "caspian": {"lat": 41.5, "lon": 51.0, "label": "Caspian", "note": "Shallow sea, level changes.", "role": "validation", "surface": "water", "sun_max": 65.0},
    "great_salt_lake": {"lat": 41.0, "lon": -112.5, "label": "Great Salt Lake", "note": "Saline lake, level changes.", "role": "validation", "surface": "water", "sun_max": 65.0},
}


def utm_epsg(lon: float, lat: float) -> str:
    """Natural UTM zone for a lon/lat point (WGS84)."""
    zone = int((lon + 180.0) // 6.0) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def grid_spec(x0: float, y0: float, size: int, scale: float, crs: str) -> dict:
    """``computePixels`` grid: north-up, top-left corner at (x0, y0)."""
    return {
        "dimensions": {"width": size, "height": size},
        "affineTransform": {
            "scaleX": scale,
            "shearX": 0,
            "translateX": x0,
            "shearY": 0,
            "scaleY": -scale,
            "translateY": y0,
        },
        "crsCode": crs,
    }


def compute_pixels(image, bands: list[str], grid: dict) -> np.ndarray:
    """Download a band stack as float64 ``(len(bands), H, W)``."""
    import ee

    array = ee.data.computePixels(
        {"expression": image, "fileFormat": "NUMPY_NDARRAY", "bandIds": bands, "grid": grid}
    )
    return np.stack([array[b].astype(np.float64) for b in bands])


def write_tif(path: Path, array: np.ndarray, x0: float, y0: float, scale: float, crs: str,
              dtype: str, nodata: float | None) -> None:
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=from_origin(x0, y0, scale, scale),
        nodata=nodata,
        compress="deflate",
        predictor=3 if dtype.startswith("float") else 2,
    ) as dst:
        dst.write(array.astype(dtype), 1)


def mss_box_stats(image, box) -> dict[str, float]:
    """Saturation, cloud, shadow and coverage fractions over the PICS window.

    The C02 ``QA_RADSAT`` band is not reliable for MSS (it flags 100 % of a
    clearly unsaturated scene), so saturation is measured on the DN themselves:
    the L1 MSS product is 8-bit with a fixed gain, so DN 255 is a clipped pixel.
    Clouds and shadows come from ``QA_PIXEL`` bits 3 and 4, which the scene
    cloud cover does not replace: a scene can be 9 % cloudy and still have a
    perfectly clear 15 km window (Libya-1 is).
    """
    import ee

    dn = image.select(["B1", "B2", "B3", "B4"])
    qa = image.select("QA_PIXEL")
    saturation = dn.eq(255).rename([f"sat_B{i + 1}" for i in range(4)])
    cloud = qa.bitwiseAnd(1 << 3).neq(0).rename("cloud")
    shadow = qa.bitwiseAnd(1 << 4).neq(0).rename("shadow")
    valid = dn.select("B1").gt(0).rename("valid")
    stack = saturation.addBands([cloud, shadow, valid])
    opts = {"scale": 60, "maxPixels": int(1e7), "bestEffort": True}
    stats = stack.reduceRegion(ee.Reducer.mean(), box, **opts).getInfo()
    out = {k: float(v or 0.0) for k, v in stats.items()}
    out["saturated_fraction"] = max(out.pop(f"sat_B{i}") for i in range(1, 5))
    out["saturation_per_band"] = [float(stats.get(f"sat_B{i}") or 0.0) for i in range(1, 5)]
    out["cloud_fraction"] = out.pop("cloud")
    out["shadow_fraction"] = out.pop("shadow")
    out["valid_fraction"] = out.pop("valid")
    return out


def coverage_fraction(image, box) -> float:
    """Fraction of the window the scene actually covers.

    Masked pixels are dropped from a stacked reduction, so this needs its own
    call; it is only paid for candidates that already passed cloud and
    saturation, which is what keeps a 20-scene search cheap.
    """
    import ee

    opts = {"scale": 60, "maxPixels": int(1e7), "bestEffort": True}
    coverage = (
        image.select("B1").mask().reduceRegion(ee.Reducer.mean(), box, **opts).getInfo()
    )
    return float(coverage.get("B1") or 0.0)


def s2_datatake(product_id: str, box) -> dict:
    """Mosaic of every tile of the S2 datatake that contains ``product_id``.

    A 15 km window can straddle a tile edge (Algeria-3 and Niger-1 do), so the
    reflectance has to come from the two tiles of the same acquisition.  The
    tiles share ``DATATAKE_IDENTIFIER`` and their ``PRODUCT_ID`` differs only in
    the tile field.
    """
    import ee

    image = ee.Image(f"COPERNICUS/S2_HARMONIZED/{product_id}")
    datatake = image.get("DATATAKE_IDENTIFIER")
    collection = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterMetadata("DATATAKE_IDENTIFIER", "equals", datatake)
        .filterMetadata("SPACECRAFT_NAME", "equals", "Sentinel-2A")
        .filterBounds(box)  # a datatake spans ~100 tiles; only the window matters
    )
    tiles = collection.aggregate_array("system:index").getInfo()
    probabilities = [
        ee.Image(f"COPERNICUS/S2_CLOUD_PROBABILITY/{tile}") for tile in tiles
    ]
    return {
        "image": collection.mosaic(),
        "probability": ee.ImageCollection(probabilities).mosaic(),
        "tiles": tiles,
        "datatake": datatake.getInfo() if hasattr(datatake, "getInfo") else datatake,
    }


def s2_cloud_fraction(datatake: dict, box) -> dict[str, float]:
    """s2cloudless probability and QA60 fractions over the box."""
    import ee

    prob = datatake["probability"]
    cloud = prob.select("probability").gte(50)
    s2 = datatake["image"]
    qa60 = s2.select("QA60").bitwiseAnd(0b110000000000).neq(0)  # opaque + cirrus
    opts = {"scale": 20, "maxPixels": int(1e8), "bestEffort": True}
    coverage = (
        s2.select(list(S2_BANDS))
        .mask()
        .reduce(ee.Reducer.min())
        .rename("coverage")
    )
    stats = ee.Dictionary(
        {
            "prob": cloud.reduceRegion(ee.Reducer.mean(), box, **opts),
            "qa60": qa60.reduceRegion(ee.Reducer.mean(), box, **opts),
            "coverage": coverage.reduceRegion(ee.Reducer.mean(), box, **opts),
        }
    ).getInfo()
    return {
        "cloud_fraction_prob50": float(stats["prob"].get("probability") or 0.0),
        "cloud_fraction_qa60": float(stats["qa60"].get("QA60") or 0.0),
        "coverage": float(stats["coverage"].get("coverage") or 0.0),
    }


def pick_mss(site: dict, box, crs: str, max_scenes: int = 20) -> dict:
    """Best unsaturated, cloud-free L1TP MSS scene covering the box.

    Landsat MSS is delivered as 8-bit DN with a fixed gain, so a bright desert
    under a high sun clips B2/B3 at DN 255: the 1984-09 Libya-4 scene is 66 %
    saturated in B2.  Bright targets are therefore restricted to
    ``SUN_ELEVATION`` between 26 and 50 degrees (zenith 40-64 deg); dark
    targets (forest, water) allow up to 65 deg, where they do not clip.  Any
    scene with more than 0.5 % clipped pixels over the window is rejected
    regardless.  The loop stops after four usable candidates: the list is
    already sorted by scene cloud cover, and evaluating all 40 costs 40
    reductions per site for no change in the choice.
    """
    import ee

    sun_max = float(site.get("sun_max", 50.0))
    base = (
        ee.ImageCollection("LANDSAT/LM05/C02/T2")
        .filterBounds(box)
        .filter(ee.Filter.stringContains("LANDSAT_PRODUCT_ID", "L1TP"))
        .filterMetadata("CLOUD_COVER", "less_than", 40)
    )
    col = base.filterMetadata("SUN_ELEVATION", "greater_than", 26.0).filterMetadata(
        "SUN_ELEVATION", "less_than", sun_max
    )
    features = col.sort("CLOUD_COVER").limit(max_scenes).getInfo()["features"]
    if not features:
        raise RuntimeError(f"no L1TP MSS scene intersects {site['label']}")

    candidates = []
    for feature in features:
        image = ee.Image(feature["id"])
        stats = mss_box_stats(image, box)
        if (
            stats["saturated_fraction"] > 0.005
            or stats["cloud_fraction"] > 0.005
            or stats["shadow_fraction"] > 0.01
        ):
            continue
        stats["coverage"] = coverage_fraction(image, box)
        if stats["coverage"] < 0.999:
            continue
        candidates.append(
            {
                "asset": feature["id"],
                "product_id": feature["properties"]["LANDSAT_PRODUCT_ID"],
                "date": feature["properties"]["DATE_ACQUIRED"],
                "sun_elevation": float(feature["properties"]["SUN_ELEVATION"]),
                "sun_azimuth": float(feature["properties"]["SUN_AZIMUTH"]),
                "earth_sun_distance": float(feature["properties"]["EARTH_SUN_DISTANCE"]),
                "wrs_path": int(feature["properties"]["WRS_PATH"]),
                "wrs_row": int(feature["properties"]["WRS_ROW"]),
                "cloud_cover_scene": float(feature["properties"]["CLOUD_COVER"]),
                "mss_crs": image.select("B1").projection().crs().getInfo(),
                **stats,
            }
        )
        if len(candidates) >= 4:
            break
    if not candidates:
        raise RuntimeError(
            f"no unsaturated cloud-free MSS scene covers the box at {site['label']}"
        )
    candidates.sort(
        key=lambda c: (
            c["saturated_fraction"] + c["cloud_fraction"],
            abs((90.0 - c["sun_elevation"]) - 52.0),
        )
    )
    return candidates[0]


def pick_s2(site: dict, box, mss: dict, max_delta: float = 6.0) -> dict:
    """Cloud-free S2A L1C acquisition matched in sun zenith to the MSS scene."""
    import ee

    mss_sza = 90.0 - mss["sun_elevation"]
    base = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(box)
        .filterMetadata("SPACECRAFT_NAME", "equals", "Sentinel-2A")
        .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", 30)
        .filterMetadata("MEAN_SOLAR_ZENITH_ANGLE", "greater_than", mss_sza - max_delta)
        .filterMetadata("MEAN_SOLAR_ZENITH_ANGLE", "less_than", mss_sza + max_delta)
        .filterDate("2019-01-01", "2022-01-01")
    )
    features = base.limit(20).getInfo()["features"]
    if not features:
        return pick_s2(site, box, mss, max_delta=max_delta + 6.0) if max_delta < 24 else None

    candidates = []
    for evaluated, feature in enumerate(
        sorted(
            features,
            key=lambda f: abs(f["properties"]["MEAN_SOLAR_ZENITH_ANGLE"] - mss_sza),
        )
    ):
        if evaluated >= 10:
            break
        props = feature["properties"]
        product_id = feature["id"].split("/")[-1]
        datatake_images = s2_datatake(product_id, box)
        stats = s2_cloud_fraction(datatake_images, box)
        if stats["coverage"] < 0.999 or stats["cloud_fraction_prob50"] > 0.01:
            continue
        candidates.append(
            {
                "asset": feature["id"],
                "product_id": product_id,
                "tiles": datatake_images["tiles"],
                "datatake": datatake_images["datatake"],
                "tile": product_id.split("_")[-2],
                "date": props["PRODUCT_ID"].split("_")[2][:8],
                "sensing_time": props["PRODUCT_ID"].split("_")[2],
                "sensor": props["SPACECRAFT_NAME"],
                "sun_zenith": float(props["MEAN_SOLAR_ZENITH_ANGLE"]),
                "sun_azimuth": float(props["MEAN_SOLAR_AZIMUTH_ANGLE"]),
                "cloud_cover_scene": float(props["CLOUDY_PIXEL_PERCENTAGE"]),
                "s2_crs": ee.Image(feature["id"]).select("B1").projection().crs().getInfo(),
                **stats,
            }
        )
        if len(candidates) >= 3:
            break
    if not candidates:
        return pick_s2(site, box, mss, max_delta=max_delta + 6.0) if max_delta < 24 else None
    return candidates[0]


def site_frame(lat: float, lon: float, crs: str) -> dict:
    """15 km grid around the site centre, in the given UTM CRS."""
    import ee
    from pyproj import Transformer

    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xc, yc = to_utm.transform(lon, lat)
    x0, y0 = xc - HALF_KM * 1000.0, yc + HALF_KM * 1000.0
    n_mss = round(2 * HALF_KM * 1000.0 / MSS_GSD)
    n_s2 = round(2 * HALF_KM * 1000.0 / S2_GSD)
    corners = [
        to_wgs.transform(x, y)
        for x, y in ((x0, y0), (x0 + n_mss * MSS_GSD, y0 - n_mss * MSS_GSD))
    ]
    box = ee.Geometry.Rectangle(
        [
            min(c[0] for c in corners),
            min(c[1] for c in corners),
            max(c[0] for c in corners),
            max(c[1] for c in corners),
        ],
        "EPSG:4326",
        False,
    )
    return {"crs": crs, "x0": x0, "y0": y0, "n_mss": n_mss, "n_s2": n_s2, "box": box}


def fetch_site(name: str, site: dict, out_root: Path, dry_run: bool) -> dict:
    lat, lon = float(site["lat"]), float(site["lon"])
    frame = site_frame(lat, lon, utm_epsg(lon, lat))
    print(f"\n=== {name} ({site['label']}) {lat:.2f} N {lon:.2f} E | {frame['crs']}")
    mss = pick_mss(site, frame["box"], frame["crs"])
    print(f"  MSS {mss['product_id']} {mss['date']} sun={mss['sun_elevation']:.1f} "
          f"cloud={100 * mss['cloud_fraction']:.3f}% cover={100 * mss['coverage']:.2f}% "
          f"sat={100 * mss['saturated_fraction']:.3f}%")
    if mss["mss_crs"] != frame["crs"]:
        # The MSS is delivered in the UTM zone of its scene centre and must not
        # be resampled; the S2 is reprojected onto that same grid instead.
        print(f"  grid CRS {frame['crs']} -> {mss['mss_crs']} (MSS native)")
        frame = site_frame(lat, lon, mss["mss_crs"])
    crs = frame["crs"]
    x0, y0, n_mss, n_s2 = frame["x0"], frame["y0"], frame["n_mss"], frame["n_s2"]
    box = frame["box"]
    s2 = pick_s2(site, box, mss)
    if s2 is None:
        raise RuntimeError(f"no cloud-free S2A scene matches the MSS sun angle at {site['label']}")
    print(f"  S2A {s2['product_id']} {s2['date']} sza={s2['sun_zenith']:.1f} "
          f"cloud={100 * s2['cloud_fraction_prob50']:.3f}% cover={100 * s2['coverage']:.2f}%")
    print(f"  SZA MSS {90 - mss['sun_elevation']:.1f} vs S2 {s2['sun_zenith']:.1f} "
          f"-> delta {abs(90 - mss['sun_elevation'] - s2['sun_zenith']):.1f} deg")

    record = {
        "site": name,
        "label": site["label"],
        "note": site["note"],
        "role": site.get("role", "calibration"),
        "surface": site.get("surface", "desert"),
        "centre_wgs84": {"lat": lat, "lon": lon},
        "window_km": 2 * HALF_KM,
        "crs": crs,
        "origin_utm": {"x": x0, "y": y0},
        "grid": {
            "mss": {"size": n_mss, "gsd_m": MSS_GSD},
            "s2": {"size": n_s2, "gsd_m": S2_GSD},
        },
        "mss": mss,
        "s2a": s2,
        "created_by": "satharmony/scripts/fetch_pics.py",
    }
    if dry_run:
        return record

    site_dir = out_root / name
    mss_dir = site_dir / "mss"
    s2_dir = site_dir / "s2"
    for stale in (mss_dir, s2_dir):
        if stale.exists():
            for old in stale.glob("*.tif"):
                old.unlink()

    import ee

    mss_image = ee.Image(mss["asset"])
    props = mss_image.toDictionary().getInfo()
    rad_mult = [
        float(props.get(f"RADIANCE_MULT_BAND_{b}", L5_RADIANCE_MULT[b - 1]))
        for b in range(1, 5)
    ]
    rad_add = [
        float(props.get(f"RADIANCE_ADD_BAND_{b}", L5_RADIANCE_ADD[b - 1]))
        for b in range(1, 5)
    ]
    mss_cfg = grid_spec(x0, y0, n_mss, MSS_GSD, crs)

    dn = compute_pixels(mss_image.select(["B1", "B2", "B3", "B4"]), ["B1", "B2", "B3", "B4"], mss_cfg)
    refl = dn_to_toa(
        np.rint(dn).astype(np.uint8),
        MSSCalibration(
            spacecraft_id="LANDSAT_5",
            sensor_id="MSS",
            date_acquired=datetime.strptime(mss["date"], "%Y-%m-%d"),
            sun_elevation=mss["sun_elevation"],
            radiance_mult=rad_mult,
            radiance_add=rad_add,
            qcal_min=[0] * 4,
            qcal_max=[255] * 4,
        ),
    )
    for band in range(4):
        band_refl = refl[band].astype(np.float32)
        band_refl[dn[band] <= 0] = np.nan
        write_tif(mss_dir / f"{mss['product_id']}.B{band + 1}.tif", band_refl, x0, y0, MSS_GSD, crs,
                  "float32", np.nan)
    qa = compute_pixels(mss_image.select(["QA_PIXEL", "QA_RADSAT"]), ["QA_PIXEL", "QA_RADSAT"], mss_cfg)
    write_tif(mss_dir / f"{mss['product_id']}.QA_PIXEL.tif", qa[0], x0, y0, MSS_GSD, crs,
              "uint16", None)
    write_tif(mss_dir / f"{mss['product_id']}.QA_RADSAT.tif", qa[1], x0, y0, MSS_GSD, crs,
              "uint16", None)
    saturated = (dn >= 255).any(axis=0).astype(np.uint8)
    write_tif(mss_dir / f"{mss['product_id']}.MSK_SATURATED.tif", saturated, x0, y0, MSS_GSD,
              crs, "uint8", None)

    datatake_images = s2_datatake(s2["product_id"], box)
    s2_image = datatake_images["image"]
    s2_cfg = grid_spec(x0, y0, n_s2, S2_GSD, crs)
    # computePixels caps the request at 48 MB: 1500 x 1500 uint16 is 4.5 MB per
    # band, so ask for four bands at a time.
    for start in range(0, len(S2_BANDS), 4):
        chunk = list(S2_BANDS[start : start + 4])
        stack = compute_pixels(s2_image.select(chunk), chunk, s2_cfg)
        for band, values in zip(chunk, stack, strict=True):
            write_tif(s2_dir / f"{s2['product_id']}.{band}.tif", values, x0, y0, S2_GSD, crs,
                      "uint16", 0)
    prob = compute_pixels(
        datatake_images["probability"].select("probability"), ["probability"], s2_cfg
    )
    write_tif(s2_dir / f"{s2['product_id']}.MSK_CLOUD_PROB.tif", prob[0], x0, y0, S2_GSD, crs,
              "uint8", None)
    qa60 = compute_pixels(s2_image.select("QA60"), ["QA60"], s2_cfg)
    write_tif(s2_dir / f"{s2['product_id']}.QA60.tif", qa60[0], x0, y0, S2_GSD, crs,
              "uint16", None)

    record["mss"]["radiance_mult"] = rad_mult
    record["mss"]["radiance_add"] = rad_add
    record["mss"]["units"] = "toa_reflectance"
    record["mss"]["conversion"] = (
        "satharmony.corrections.landsat.mss.dn_to_toa: "
        "rho = pi * (M_L * DN + A_L) * d^2 / (ESUN * sin(sun_elevation)), L5_MSS ESUN"
    )
    record["s2a"]["bands"] = list(S2_BANDS)
    record["files"] = {
        "mss": sorted(p.name for p in mss_dir.glob("*.tif")),
        "s2": sorted(p.name for p in s2_dir.glob("*.tif")),
    }
    (site_dir / "metadata.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"  wrote {site_dir}")
    return record


def try_site(
    name: str, site: dict, out_root: Path, dry_run: bool
) -> tuple[dict | None, str]:
    """Fetch one site; a missing scene must not kill the whole batch."""
    try:
        return fetch_site(name, site, out_root, dry_run), ""
    except Exception as exc:
        print(f"  SKIP {name}: {exc}", flush=True)
        return None, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sites", nargs="*", default=list(SITES))
    parser.add_argument("--project", default="ee-contrerasnetk")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="do not rewrite pics_index.json (parallel workers share one output root)",
    )
    args = parser.parse_args()

    import ee

    ee.Initialize(project=args.project)
    records: list[dict] = []
    failures: list[dict[str, str]] = []
    for name in args.sites:
        record, error = try_site(name, SITES[name], args.out, args.dry_run)
        if record is not None:
            records.append(record)
        else:
            failures.append({"site": name, "error": error})
    if not records:
        raise SystemExit("no site could be fetched")

    if not args.dry_run and not args.no_index:
        summary = args.out / "pics_index.json"
        existing = []
        if summary.exists():
            existing = json.loads(summary.read_text()).get("sites", [])
        merged = {s["site"]: s for s in existing}
        merged.update({s["site"]: s for s in records})
        ordered = [merged[name] for name in SITES if name in merged]
        summary.write_text(
            json.dumps({"sites": ordered, "failures": failures}, indent=2) + "\n"
        )
        print(f"\nwrote {summary} ({len(ordered)} sites, {len(failures)} skipped)")
        print(
            "\n| site | MSS | date | S2A | date | r-ready |\n|---|---|---|---|---|---|"
        )
        for r in records:
            print(
                f"| {r['site']} | {r['mss']['product_id']} | {r['mss']['date']} | "
                f"{r['s2a']['product_id']} | {r['s2a']['date']} | "
                f"{100 * (1 - max(r['mss']['cloud_fraction'], r['s2a']['cloud_fraction_prob50'])):.2f}% clear |"
            )


if __name__ == "__main__":
    sys.exit(main())
