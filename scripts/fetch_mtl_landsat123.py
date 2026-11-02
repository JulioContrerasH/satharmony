#!/usr/bin/env python
"""Fetch a real MTL file for a Landsat 1-3 MSS scene (bands 4-7).

Three sources are tried, in this order, and every attempt is reported:

1. **USGS M2M** (``https://m2m.cr.usgs.gov``). The login endpoint was renamed
   from ``login`` to ``login-token``; it needs a USGS ERS account, which is not
   the Earthdata account in ``~/.netrc``. Credentials are read from
   ``USGS_ERS_USER`` / ``USGS_ERS_PASSWORD`` (env or
   ``~/.config/anycloud/env.sh``). Without them the source is skipped and the
   reason is printed.
2. **ESA EOWEB** (``https://eoweb.dlr.de/egp``). Login is a DLR CAS SSO; the
   script authenticates with ``EOWEB_USER`` / ``EOWEB_PASSWORD`` and checks
   whether the account is authorised for the portal.
3. **USGS Landsat Collection 2 L1 mirror on Azure**, exposed through the
   Microsoft Planetary Computer STAC API (``landsat-c2-l1``). The ``mtl.txt``
   asset is the USGS ``*_MTL.txt`` of the product, byte for byte, and needs no
   credentials. Used as the fallback that actually works.

The script never prints secret values, only whether each variable is present.

Usage
-----
    python scripts/fetch_mtl_landsat123.py
    python scripts/fetch_mtl_landsat123.py --platforms landsat-3 --out /tmp/mtl
    python scripts/fetch_mtl_landsat123.py --list --platforms landsat-1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO.parent / "assets" / "docs" / "mtl"

ENV_FILE = Path.home() / ".config" / "anycloud" / "env.sh"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
M2M = "https://m2m.cr.usgs.gov/api/api/json/stable"

#: One scene per platform. Chosen for a clean, single-sensor MSS product
#: (L1GS for Landsat 1-2, L1TP for Landsat 3) with all four bands 4-7.
SCENES = {
    "landsat-1": "LM01_L1GS_043044_19780107_02_T2",
    "landsat-2": "LM02_L1GS_223027_19820312_02_T2",
    "landsat-3": "LM03_L1TP_077017_19830331_02_T2",
}


def load_secrets() -> dict[str, str]:
    """Environment first, then ``~/.config/anycloud/env.sh`` (values not logged)."""
    secrets = dict(os.environ)
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            match = re.match(r"(?:export\s+)?([A-Z0-9_]+)=(.*)$", line.strip())
            if match and match.group(1) not in secrets:
                secrets[match.group(1)] = match.group(2).strip().strip("\"'")
    return secrets


def _get(url: str, **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", 60)
    kwargs.setdefault("headers", {"User-Agent": "satharmony-fetch-mtl/0.4"})
    return requests.get(url, **kwargs)


# --------------------------------------------------------------------------
# 1. USGS M2M
# --------------------------------------------------------------------------


def probe_m2m(secrets: dict[str, str]) -> dict:
    user = secrets.get("USGS_ERS_USER") or secrets.get("M2M_USER")
    password = secrets.get("USGS_ERS_PASSWORD") or secrets.get("M2M_PASSWORD")
    report: dict = {
        "source": "usgs-m2m",
        "endpoint": f"{M2M}/login-token",
        "credentials_present": bool(user and password),
    }
    if not report["credentials_present"]:
        report["status"] = "skipped-no-credentials"
        report["detail"] = (
            "~/.netrc only has api.wandb.ai and urs.earthdata.nasa.gov; env.sh only has "
            "EARTHDATA_* and EOWEB_*. A USGS ERS account is needed (it is not the "
            "Earthdata one). POST /login returns NOT_FOUND: the endpoint is now "
            "/login-token."
        )
        return report
    try:
        response = requests.post(
            f"{M2M}/login-token",
            json={"username": user, "password": password},
            timeout=60,
            headers={"User-Agent": "satharmony-fetch-mtl/0.4"},
        )
        payload = response.json()
        report["status"] = "ok" if payload.get("data") else "error"
        report["error_code"] = payload.get("errorCode")
        report["error_message"] = payload.get("errorMessage")
    except Exception as exc:
        report["status"] = "error"
        report["detail"] = f"{type(exc).__name__}: {exc}"
    return report


# --------------------------------------------------------------------------
# 2. ESA EOWEB (DLR CAS SSO)
# --------------------------------------------------------------------------


def probe_eoweb(secrets: dict[str, str]) -> dict:
    user = secrets.get("EOWEB_USER")
    password = secrets.get("EOWEB_PASSWORD")
    report: dict = {
        "source": "esa-eoweb",
        "endpoint": "https://eoweb.dlr.de/egp/",
        "credentials_present": bool(user and password),
    }
    if not report["credentials_present"]:
        report["status"] = "skipped-no-credentials"
        return report

    session = requests.Session()
    session.headers["User-Agent"] = "satharmony-fetch-mtl/0.4"
    try:
        # Asking for a protected resource bounces to the DLR CAS login.
        login = session.get("https://eoweb.dlr.de/egp/wsdl", timeout=60)
        execution = re.search(r'name="execution"[^>]*value="([^"]*)"', login.text) or re.search(
            r'value="([^"]*)"[^>]*name="execution"', login.text
        )
        landing = session.post(
            login.url,
            data={
                "username": user,
                "password": password,
                "execution": execution.group(1) if execution else "",
                "_eventId": "submit",
            },
            timeout=60,
        )
        report["sso_ticket"] = "TGC" in session.cookies
        report["landing_status"] = landing.status_code
        body = landing.text.lower()
        if "not authorized" in body or "register first" in body:
            report["status"] = "forbidden-not-registered"
            report["detail"] = (
                "CAS SSO authenticates the account and issues the TGC cookie, but the "
                "portal answers 403: 'Sorry, you are not authorized to use this "
                "service. Please register first!' (EOWEB needs a separate registration "
                "with a signed terms-of-use, it is not the same as an EOC login)."
            )
        elif landing.status_code == 200 and "definitions" in landing.text[:2000]:
            report["status"] = "ok"
        else:
            report["status"] = f"http-{landing.status_code}"
    except Exception as exc:
        report["status"] = "error"
        report["detail"] = f"{type(exc).__name__}: {exc}"
    return report


# --------------------------------------------------------------------------
# 3. USGS Landsat C2 L1 mirror (Planetary Computer STAC)
# --------------------------------------------------------------------------


def _stac_search(query: dict, limit: int = 10, tries: int = 4) -> list[dict]:
    """POST a STAC search; the public endpoint throws sporadic 504s."""
    last: requests.HTTPError | None = None
    for attempt in range(tries):
        try:
            response = requests.post(
                f"{STAC}/search",
                json={"collections": ["landsat-c2-l1"], "limit": limit, **query},
                timeout=60,
                headers={"User-Agent": "satharmony-fetch-mtl/0.4"},
            )
            response.raise_for_status()
            return response.json()["features"]
        except requests.HTTPError as exc:  # noqa: PERF203 - retry needs it in the loop
            last = exc
            if exc.response is not None and exc.response.status_code < 500:
                raise
            time.sleep(2 * (attempt + 1))
    assert last is not None
    raise last


def _sign(href: str) -> str:
    response = _get(
        STAC.replace("/stac/v1", "/sas/v1/sign"), params={"href": href}
    )
    response.raise_for_status()
    return response.json()["href"]


def fetch_from_stac(platform: str, scene_id: str, out_dir: Path) -> dict:
    """Download the USGS ``*_MTL.txt`` of one scene from the public mirror."""
    report: dict = {"source": "planetary-computer-usgs-mirror", "platform": platform}
    items = _stac_search({"ids": [scene_id]})
    if not items:
        report["status"] = "not-found"
        return report
    item = items[0]
    asset = item["assets"]["mtl.txt"]
    text = _get(_sign(asset["href"])).text
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{item['id']}_MTL.txt"
    path.write_text(text)
    report.update(
        {
            "status": "ok",
            "item_id": item["id"],
            "scene_id": item["properties"].get("landsat:scene_id"),
            "datetime": item["properties"].get("datetime"),
            "path": str(path),
            "bytes": path.stat().st_size,
            "product_href": asset["href"],
        }
    )
    return report


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--platforms",
        nargs="+",
        default=sorted(SCENES),
        choices=sorted(SCENES),
    )
    parser.add_argument("--list", action="store_true", help="only list candidate scenes")
    parser.add_argument("--report", type=Path, default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    secrets = load_secrets()

    if args.list:
        for platform in args.platforms:
            print(f"# {platform}")
            for feature in _stac_search({"query": {"platform": {"eq": platform}}}, limit=5):
                props = feature["properties"]
                print(
                    f"  {feature['id']:38s} {props.get('landsat:scene_id')} "
                    f"{props.get('datetime', '')[:10]} cloud={props.get('eo:cloud_cover')}"
                )
        return 0

    report: dict = {
        "sources": {"usgs-m2m": probe_m2m(secrets), "esa-eoweb": probe_eoweb(secrets)},
        "mtl": [],
    }
    for platform, scene_id in SCENES.items():
        if platform not in args.platforms:
            continue
        try:
            report["mtl"].append(fetch_from_stac(platform, scene_id, args.out))
        except Exception as exc:
            report["mtl"].append(
                {"source": "planetary-computer-usgs-mirror", "platform": platform,
                 "status": "error", "detail": f"{type(exc).__name__}: {exc}"}
            )

    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    ok = [entry for entry in report["mtl"] if entry.get("status") == "ok"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
