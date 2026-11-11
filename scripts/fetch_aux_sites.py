#!/usr/bin/env python
"""Build an auxiliary site bank on the MSS windows that the archive really has.

The 49 CEOS PICS were chosen for surface stability, without looking at the
Landsat 1-4 archive; most of them have no early scene. These sites are the
other way round: they come from the archive itself (one scene per site), and
they only need a clear Sentinel-2 composite of the same place and season.

Per site:

- 250 x 250 at 60 m MSS window, centred on the scene centre.
- MSS DN + QA_PIXEL on that window, converted to TOA with QCALMAX = 255
  (Collection 2 MSS is resampled to 8 bits) and a per-pixel clear mask.
- Sentinel-2 SR median of the same month, 2018-2022, 1500 x 1500 at 10 m.

Outputs ``<out>/s2/<name>.tif`` (13, H, W uint16), ``<out>/mss/<name>.tif``
(4, 250, 250 float32), ``<out>/mask/<name>.tif`` (250, 250 uint8) and
``<out>/meta.json``.

Usage::

    python scripts/fetch_aux_sites.py --sensor L2 --sites /tmp/l2_sites.json --limit 3
    python scripts/fetch_aux_sites.py --sensor L2 --sites /tmp/l2_sites.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from satharmony.corrections.landsat.constants import LMAX_MSS  # noqa: E402
from satharmony.corrections.landsat.metadata import MSSCalibration  # noqa: E402
from satharmony.corrections.landsat.mss import dn_to_toa  # noqa: E402
from satharmony.emulators.s2_to_mss.sbaf import S2_BAND_ORDER  # noqa: E402

PROJECT = "ee-contrerasnetk"
ROOT = Path("/data/databases/CLOUD_MSS/MSS_S2")
QCAL_MAX = 255
S2_WINDOW_YEARS = (2018, 2022)
MSS_BANDS = {"L1": ("B4", "B5", "B6", "B7"), "L2": ("B4", "B5", "B6", "B7"), "L3": ("B4", "B5", "B6", "B7"), "L4": ("B1", "B2", "B3", "B4")}
QA_FILL = 1 << 0
QA_DILATED = 1 << 1
QA_CLOUD = 1 << 3


def site_name(sensor: str, lon: float, lat: float) -> str:
    return f"{sensor.lower()}_{abs(lon):05.1f}{'E' if lon >= 0 else 'W'}_{abs(lat):04.1f}{'N' if lat >= 0 else 'S'}"


def s2_composite(lon: float, lat: float, month: int):
    """Median of the clearest same-season Sentinel-2 L1C scenes, plus validity.

    L1C (``S2_HARMONIZED``) is the product the PICS already use, and it is the
    only one that carries B10, which the emulator stack expects.
    """
    import ee

    start_year, end_year = S2_WINDOW_YEARS
    collection = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(ee.Geometry.Point(lon, lat))
        .filter(ee.Filter.calendarRange(month, month, "month"))
        .filterDate(f"{start_year}-01-01", f"{end_year}-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    )

    def clear(image):
        qa = image.select("QA60")
        good = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return image.updateMask(good)

    masked = collection.map(clear)
    composite = masked.median()
    valid = masked.select("B2").count().gt(0).rename("valid")
    return composite.addBands(valid)


def request_site(site: dict, sensor: str) -> tuple[object, object, dict]:
    """Build the S2 and MSS requests for one site, plus its metadata."""
    import cubexpress
    import ee

    lon, lat = float(site["lon"]), float(site["lat"])
    date = site.get("date") or f"{site['id'][-8:-4]}-{site['id'][-4:-2]}-{site['id'][-2:]}"
    month = int(date[5:7])
    composite = s2_composite(lon, lat, month)
    rt_mss = cubexpress.point_to_rt(lon=lon, lat=lat, width=250, height=250, scale=60)
    rt_s2 = cubexpress.point_to_rt(lon=lon, lat=lat, width=1500, height=1500, scale=10)

    bands = MSS_BANDS[sensor]
    asset = site["asset"]
    if not asset.endswith(site["id"]):
        asset = f"{asset}/{site['id']}"
    image = ee.Image(asset).select(
        [*bands, "QA_PIXEL"], [*S2_BAND_ORDER[:4], "QA_PIXEL"]
    )
    mss_row = cubexpress.RequestRow(
        id=site["name"],
        raster_transform=rt_mss,
        image=image,
        bands=("B1", "B2", "B3", "B4", "QA_PIXEL"),
    )
    s2_row = cubexpress.RequestRow(
        id=site["name"],
        raster_transform=rt_s2,
        image=composite.select([*S2_BAND_ORDER, "valid"]),
        bands=(*S2_BAND_ORDER, "valid"),
    )
    meta = {
        "site": site["name"],
        "sensor": sensor,
        "lon": lon,
        "lat": lat,
        "mss_asset": site["asset"],
        "mss_date": date,
        "mss_id": site["id"],
        "sun_elevation": float(site["sun"]),
        "scene_cloud": float(site["cloud"]),
        "s2_month": month,
        "s2_scenes": None,
    }
    return mss_row, s2_row, meta


def mss_to_toa(dn: np.ndarray, qa: np.ndarray, date: str, sun: float, sensor: str) -> tuple[np.ndarray, np.ndarray]:
    """DN -> TOA and a clear mask from QA_PIXEL."""
    calib = MSSCalibration(
        spacecraft_id=f"LANDSAT_{sensor[-1]}",
        sensor_id="MSS",
        date_acquired=datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc),
        sun_elevation=sun,
        radiance_mult=[value / QCAL_MAX for value in LMAX_MSS[f"{sensor}_MSS"]],
        radiance_add=[0.0, 0.0, 0.0, 0.0],
        qcal_min=[0, 0, 0, 0],
        qcal_max=[QCAL_MAX] * 4,
    )
    toa = dn_to_toa(dn.astype(np.uint8), calib)
    qa = qa.astype(np.uint16)
    cloudy = ((qa & QA_DILATED) != 0) | ((qa & QA_CLOUD) != 0)
    fill = (qa & QA_FILL) != 0
    mask = (~cloudy & ~fill).astype(np.uint8)
    return toa.astype(np.float32), mask


def main() -> int:
    import cubexpress
    import ee

    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", required=True, choices=["L1", "L2", "L3", "L4"])
    parser.add_argument("--sites", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--nworkers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    out = args.out or ROOT / f"aux_{args.sensor.lower()}"
    sites = json.loads(args.sites.read_text())
    if args.limit:
        sites = sites[: args.limit]
    for site in sites:
        site["name"] = site_name(args.sensor, site["lon"], site["lat"])

    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else []
    known = {row["site"] for row in meta}
    mss_rows, s2_rows = [], []
    for site in sites:
        if site["name"] in known:
            continue
        print(f"building {site['name']} ...", flush=True)
        try:
            mss_row, s2_row, row = request_site(site, args.sensor)
        except Exception as error:  # a broken site must not stop the batch
            print(f"  skipped {site['name']}: {type(error).__name__}: {str(error)[:160]}", flush=True)
            continue
        mss_rows.append(mss_row)
        s2_rows.append(s2_row)
        meta.append(row)
        print(f"{row['site']}: {row['mss_id']} {row['mss_date']} sun={row['sun_elevation']:.1f} s2={row['s2_scenes']}", flush=True)

    for folder in ("dn", "dn_s2"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    if mss_rows:
        print(cubexpress.express(cubexpress.RequestTable(mss_rows), out / "dn", nworkers=args.nworkers))
    if s2_rows:
        print(cubexpress.express(cubexpress.RequestTable(s2_rows), out / "dn_s2", nworkers=args.nworkers))
    meta_path.write_text(json.dumps(meta, indent=2))

    for folder in ("s2", "mss", "mask"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    for row in meta:
        name = row["site"]
        mss_source = out / "dn" / f"{name}.tif"
        s2_source = out / "dn_s2" / f"{name}.tif"
        if not (mss_source.exists() and s2_source.exists()):
            continue
        stack = tifffile.imread(mss_source)
        if stack.ndim == 3 and stack.shape[-1] == 5:
            dn = np.moveaxis(stack[..., :4], -1, 0)
            qa = stack[..., 4]
        elif stack.ndim == 3 and stack.shape[0] == 5:
            dn, qa = stack[:4], stack[4]
        else:
            dn, qa = stack, np.zeros(stack.shape[1:], dtype=np.uint8)
        toa, mask = mss_to_toa(dn, qa, row["mss_date"], row["sun_elevation"], row["sensor"])
        tifffile.imwrite(out / "mss" / f"{name}.tif", toa)
        tifffile.imwrite(out / "mask" / f"{name}.tif", mask)

        s2 = tifffile.imread(s2_source)
        if s2.ndim == 3 and s2.shape[-1] == len(S2_BAND_ORDER) + 1:
            s2 = np.moveaxis(s2, -1, 0)
        tifffile.imwrite(out / "s2" / f"{name}.tif", s2[: len(S2_BAND_ORDER)].astype(np.uint16))
        row["mss_clear"] = float(mask.mean())
        row["s2_clear"] = float((s2[-1] > 0).mean())
        print(f"{name}: TOA max {toa.max():.3f} mss_clear {row['mss_clear']:.3f} s2_clear {row['s2_clear']:.3f}", flush=True)

    meta_path.write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
