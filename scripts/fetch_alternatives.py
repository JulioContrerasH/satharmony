#!/usr/bin/env python
"""Fetch an alternative scene for the pairs Julio rejected in the review app.

The first bank was chosen by scene cloud cover, which is the wrong number: a
scene can be 80 % cloudy and still be clean over the 15 km window we use. This
script ranks candidates by the cloud fraction **inside the window** (QA_PIXEL
over the exact chip rectangle), then by the sun-elevation distance to the pair
it replaces.

Output: ``MSS_S2/alt_<sensor>/`` with the same layout as the other banks, one
file per site suffixed ``_a1``.

Usage::

    python fetch_alternatives.py --sensor L3 --limit 2
    python fetch_alternatives.py --sensor L3
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

import cubexpress  # noqa: E402
import ee  # noqa: E402

from satharmony.corrections.landsat.constants import LMAX_MSS  # noqa: E402
from satharmony.corrections.landsat.metadata import MSSCalibration  # noqa: E402
from satharmony.corrections.landsat.mss import dn_to_toa  # noqa: E402
from satharmony.emulators.s2_to_mss.sbaf import S2_BAND_ORDER  # noqa: E402

ROOT = Path("/data/databases/CLOUD_MSS/MSS_S2")
REVIEW = REPO.parent / "assets/codes/checks/pair_review"
PROJECT = "ee-contrerasnetk"
CLOUD_MAX = 40.0
QCAL_MAX = 255
CHIP = 250
SCALE = 60.0

COLLECTIONS = {
    "L1": ("LANDSAT/LM01/C02/T1", "LANDSAT/LM01/C02/T2", "1972-07-01", "1979-01-01"),
    "L2": ("LANDSAT/LM02/C02/T1", "LANDSAT/LM02/C02/T2", "1975-01-01", "1982-07-01"),
    "L3": ("LANDSAT/LM03/C02/T1", "LANDSAT/LM03/C02/T2", "1978-03-01", "1983-09-01"),
    "L4": ("LANDSAT/LM04/C02/T1", "LANDSAT/LM04/C02/T2", "1982-07-01", "1994-01-01"),
    "L5": ("LANDSAT/LM05/C02/T1", "LANDSAT/LM05/C02/T2", "1984-03-01", "2013-01-01"),
}
BAND_MAP = {
    "L1": (("B4", "B5", "B6", "B7"), ("B1", "B2", "B3", "B4")),
    "L2": (("B4", "B5", "B6", "B7"), ("B1", "B2", "B3", "B4")),
    "L3": (("B4", "B5", "B6", "B7"), ("B1", "B2", "B3", "B4")),
    "L4": (("B1", "B2", "B3", "B4"), ("B1", "B2", "B3", "B4")),
    "L5": (("B1", "B2", "B3", "B4"), ("B1", "B2", "B3", "B4")),
}


def rejected() -> list[dict]:
    """Pairs not marked ok in the review app, with their diagnosis."""
    path = REVIEW / "estado_backup.json"
    items = json.loads(path.read_text())["items"]
    return [row for row in items if row.get("verdict") != "ok"]


def geometry(sensor: str, bank: str, site: str) -> dict | None:
    """Window rectangle and target sun for a pair id."""
    if bank in ("pics", "pics_l4"):
        index = json.loads((ROOT / "pics/pics_index.json").read_text())
        entry = next((e for e in index["sites"] if e["site"] == site), None)
        if entry is None:
            return None
        grid = entry["grid"]["mss"]
        return {
            "lon": entry["centre_wgs84"]["lon"],
            "lat": entry["centre_wgs84"]["lat"],
            "crs": entry["crs"],
            "x": entry["origin_utm"]["x"],
            "y": entry["origin_utm"]["y"],
            "size": grid["size"],
            "sun": float(entry["mss"].get("sun_elevation") or 45.0),
            "used": entry["mss"].get("date", ""),
            "label": entry.get("label", site),
        }
    base = ROOT / f"{bank}_{sensor.lower()}"
    meta = json.loads((base / "meta.json").read_text())
    row = next((m for m in meta if m["site"] == site), None)
    if row is None:
        return None
    rt = cubexpress.point_to_rt(lon=row["lon"], lat=row["lat"], width=CHIP, height=CHIP, scale=SCALE)
    return {
        "lon": row["lon"],
        "lat": row["lat"],
        "crs": rt.crs,
        "x": rt.translate_x,
        "y": rt.translate_y,
        "size": CHIP,
        "sun": float(row.get("sun_elevation") or 45.0),
        "used": row.get("mss_date", ""),
        "label": site,
    }


def window_cloud(image, geom: dict) -> float:
    """Clear fraction inside the chip window, from the MSS QA_PIXEL."""
    x1 = geom["x"] + geom["size"] * SCALE
    y0 = geom["y"] - geom["size"] * SCALE
    rect = ee.Geometry.Rectangle([geom["x"], y0, x1, geom["y"]], proj=geom["crs"], geodesic=False)
    qa = image.select("QA_PIXEL").toUint16()
    bad = qa.bitwiseAnd(1).neq(0).Or(qa.bitwiseAnd(2).neq(0)).Or(qa.bitwiseAnd(8).neq(0))
    value = bad.rename("bad").reduceRegion(
        reducer=ee.Reducer.mean(), geometry=rect, scale=SCALE, maxPixels=10_000_000
    ).get("bad")
    return float(value.getInfo() or 1.0)


def candidates(sensor: str, geom: dict, alt_index: int) -> dict | None:
    """Best unused scene by window cloud, then by sun distance."""
    t1, t2, start, end = COLLECTIONS[sensor]
    seen: list[tuple[str, str]] = []
    for collection in (t2, t1):
        scenes = (
            ee.ImageCollection(collection)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", CLOUD_MAX))
            .filterBounds(ee.Geometry.Point(geom["lon"], geom["lat"]).buffer(15000))
        )
        ids = scenes.aggregate_array("system:index").getInfo()
        if not ids:
            continue
        suns = scenes.aggregate_array("SUN_ELEVATION").getInfo()
        pool = []
        for scene_id, sun in zip(ids, suns, strict=True):
            date = f"{scene_id[-8:-4]}-{scene_id[-4:-2]}-{scene_id[-2:]}"
            if date == geom["used"]:
                continue
            pool.append((scene_id, date, float(sun)))
        pool.sort(key=lambda row: abs(row[2] - geom["sun"]))
        for scene_id, date, sun in pool[:12]:
            image = ee.Image(f"{collection}/{scene_id}")
            try:
                cloud = window_cloud(image, geom)
            except Exception:
                continue
            seen.append((f"{scene_id} ({date}) ventana nublada {cloud:.0%}", collection))
            if cloud <= 0.02:
                return {
                    "asset": f"{collection}/{scene_id}",
                    "collection": collection,
                    "id": scene_id,
                    "date": date,
                    "sun": sun,
                    "window_cloud": cloud,
                }
    print(f"    sin candidato limpio; probados {len(seen)}", flush=True)
    return None


def s2_composite(lon: float, lat: float, month: int):
    """Median of clear same-quarter L1C scenes, B10 included."""
    collection = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(ee.Geometry.Point(lon, lat))
        .filter(ee.Filter.calendarRange(month, month, "month"))
        .filterDate("2016-01-01", "2021-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
    )

    def clear(image):
        qa = image.select("QA60")
        good = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return image.updateMask(good)

    masked = collection.map(clear)
    composite = masked.median()
    valid = masked.select("B2").count().gt(0).rename("valid")
    return composite.addBands(valid)


def to_toa(dn: np.ndarray, date: str, sun: float, sensor: str) -> np.ndarray:
    # C02 MSS is always 8 bit: QCALMAX is 255 even when the scene is dark.
    # Guessing 127 for scenes whose maximum DN is small doubled their TOA.
    calib = MSSCalibration(
        spacecraft_id=f"LANDSAT_{sensor[-1]}",
        sensor_id="MSS",
        date_acquired=datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc),
        sun_elevation=sun,
        radiance_mult=[v / QCAL_MAX for v in LMAX_MSS[f"{sensor}_MSS"]],
        radiance_add=[0.0, 0.0, 0.0, 0.0],
        qcal_min=[0, 0, 0, 0],
        qcal_max=[QCAL_MAX] * 4,
    )
    return dn_to_toa(dn.astype(np.uint8), calib)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", default=None, choices=["L1", "L2", "L3", "L4", "L5"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--nworkers", type=int, default=6)
    parser.add_argument("--tag", default="a1")
    args = parser.parse_args()

    ee.Initialize(project=PROJECT)
    rows = rejected()
    if args.sensor:
        rows = [r for r in rows if r["id"].split("__")[0] == args.sensor]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} pares rechazados a reintentar", flush=True)

    jobs = []
    for row in rows:
        parts = row["id"].split("__")
        sensor = parts[0]
        if parts[0] == "L4":
            bank = "pics_l4"
        elif parts[0] == "L5":
            bank = "pics"
        else:
            bank = "aux" if parts[1] == "desert" else "wet"
        site = parts[-1]
        geom = geometry(sensor, bank, site)
        if geom is None:
            print(f"  {row['id']}: sin geometria, salto", flush=True)
            continue
        print(f"  {row['id']}: {geom['label']} ({geom['used']})", flush=True)
        found = candidates(sensor, geom, 0)
        if found is None:
            continue
        name = f"{site}_{args.tag}"
        jobs.append({"row": row, "sensor": sensor, "geom": geom, "found": found, "name": name})

    if not jobs:
        print("nada que descargar")
        return 0

    by_sensor: dict[str, list[dict]] = {}
    for job in jobs:
        by_sensor.setdefault(job["sensor"], []).append(job)

    for sensor, group in by_sensor.items():
        out = ROOT / f"alt_{sensor.lower()}"
        for folder in ("dn", "dn_s2", "s2", "mss", "mask"):
            (out / folder).mkdir(parents=True, exist_ok=True)
        meta_path = out / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else []
        known = {m["site"] for m in meta}

        mss_rows, s2_rows = [], []
        for job in group:
            if job["name"] in known:
                continue
            geom, found = job["geom"], job["found"]
            src_bands, dst_bands = BAND_MAP[sensor]
            image = ee.Image(found["asset"]).select([*src_bands, "QA_PIXEL"], [*dst_bands, "QA_PIXEL"])
            rt_mss = cubexpress.RasterTransform(
                crs=geom["crs"],
                translate_x=geom["x"],
                translate_y=geom["y"],
                scale_x=SCALE,
                scale_y=-SCALE,
                width=geom["size"],
                height=geom["size"],
            )
            mss_rows.append(
                cubexpress.RequestRow(
                    id=job["name"],
                    raster_transform=rt_mss,
                    image=image,
                    bands=(*dst_bands, "QA_PIXEL"),
                )
            )
            month = int(found["date"][5:7])
            composite = s2_composite(geom["lon"], geom["lat"], month)
            rt_s2 = cubexpress.point_to_rt(
                lon=geom["lon"], lat=geom["lat"], width=1500, height=1500, scale=10
            )
            s2_rows.append(
                cubexpress.RequestRow(
                    id=job["name"],
                    raster_transform=rt_s2,
                    image=composite.select([*S2_BAND_ORDER, "valid"]),
                    bands=(*S2_BAND_ORDER, "valid"),
                )
            )
            meta.append(
                {
                    "site": job["name"],
                    "sensor": sensor,
                    "replaces": job["row"]["id"],
                    "reason": job["row"].get("note", ""),
                    "mss_asset": found["asset"],
                    "mss_id": found["id"],
                    "mss_date": found["date"],
                    "sun_elevation": found["sun"],
                    "window_cloud": found["window_cloud"],
                    "lon": geom["lon"],
                    "lat": geom["lat"],
                }
            )
            print(
                f"    {job['name']}: {found['id']} {found['date']} "
                f"ventana {found['window_cloud']:.0%} sol {found['sun']:.1f}",
                flush=True,
            )

        if mss_rows:
            print(cubexpress.express(cubexpress.RequestTable(mss_rows), out / "dn", nworkers=args.nworkers))
        if s2_rows:
            print(cubexpress.express(cubexpress.RequestTable(s2_rows), out / "dn_s2", nworkers=args.nworkers))
        meta_path.write_text(json.dumps(meta, indent=2))

        for row in meta:
            name = row["site"]
            mss_source, s2_source = out / "dn" / f"{name}.tif", out / "dn_s2" / f"{name}.tif"
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
            toa = to_toa(dn, row["mss_date"], row["sun_elevation"], row["sensor"])
            qau = qa.astype(np.uint16)
            cloudy = ((qau & 2) != 0) | ((qau & 8) != 0)
            fill = (qau & 1) != 0
            mask = (~cloudy & ~fill).astype(np.uint8)
            tifffile.imwrite(out / "mss" / f"{name}.tif", toa.astype(np.float32))
            tifffile.imwrite(out / "mask" / f"{name}.tif", mask)
            tifffile.imwrite(out / "s2" / f"{name}.tif", tifffile.imread(s2_source))
            print(f"    {name}: TOA {toa.shape}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
