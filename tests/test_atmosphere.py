"""Tests for the atmospheric decomposition of ``band_scaling`` (G2 follow-up).

Two artefacts produced by the vault scripts are read here:

* ``assets/md/fuentes/srf/gas_bands.json``: 6SV band-integrated gaseous
  transmittance (``scripts/fit_gas_bands.py``).
* ``assets/md/fuentes/srf/pics_water_vapour.json``: ERA5 water-vapour column
  per PICS and date plus the gas scale per band
  (``scripts/pics_water_vapour.py``).

They are optional: without them the tests skip, because 6SV lives in its own
conda env and the ERA5 lookup needs Earth Engine.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

VAULT = Path(__file__).resolve().parents[2]
SRF = VAULT / "assets" / "md" / "fuentes" / "srf"

MSS_BANDS = ("B1", "B2", "B3", "B4")


def _load(name: str) -> dict:
    path = SRF / name
    if not path.exists():
        pytest.skip(f"missing artefact: {path}")
    return json.loads(path.read_text())


def _gas() -> tuple[list[float], dict]:
    payload = _load("gas_bands.json")
    grid = sorted(float(k) for k in payload["bands"])
    return grid, payload["bands"]


class TestGasBandTable:
    """The 6SV table has to show the right physics or the analysis is void."""

    def test_water_vapour_hits_nir_only(self):
        grid, bands = _gas()
        low = bands[f"{grid[0]:g}"]["mss"]
        high = bands[f"{grid[-1]:g}"]["mss"]

        # Visible bands: gaseous absorption is O3-dominated and flat in TCWV.
        assert abs(low["B1"] / high["B1"] - 1.0) < 0.02
        assert abs(low["B2"] / high["B2"] - 1.0) < 0.025
        # NIR bands: the 0.72/0.82/0.94 um H2O bands live here.
        assert low["B3"] / high["B3"] > 1.05
        assert low["B4"] / high["B4"] > 1.15

    def test_transmittance_decreases_with_column(self):
        grid, bands = _gas()
        for band in ("B3", "B4"):
            series = [bands[f"{w:g}"]["mss"][band] for w in grid]
            assert all(a > b for a, b in itertools.pairwise(series)), band

    def test_pinned_values_at_one_cm(self):
        _, bands = _gas()
        block = bands["1"]
        assert block["mss"]["B1"] == pytest.approx(0.9628, abs=5e-4)
        assert block["mss"]["B3"] == pytest.approx(0.9395, abs=5e-4)
        assert block["mss"]["B4"] == pytest.approx(0.8807, abs=5e-4)
        assert block["s2"]["B8A"] == pytest.approx(0.9995, abs=5e-4)
        assert block["s2"]["B9"] == pytest.approx(0.4310, abs=5e-4)

    def test_table_records_its_geometry(self):
        payload = _load("gas_bands.json")
        geometry = payload["geometry"]
        assert geometry["solar_zenith"] == 45.0
        assert geometry["view_zenith"] == 8.0
        assert payload["sensor_mss"] == "L5MSS"
        assert payload["sensor_s2"] == "S2A"


class TestBand3Gate:
    """Why B3 sits at 0.867 and what the 95 % CI says about the 0.93 gate."""

    def test_gas_explains_most_of_the_band3_gap(self):
        water = _load("pics_water_vapour.json")
        s_gas = sorted(site["bands"]["B3"]["s_gas"] for site in water["sites"].values())
        median = s_gas[len(s_gas) // 2]

        scaling = _load("sbaf_pics_band_scaling.json")
        observed = scaling["bands"]["B3"]["mean"]

        assert 0.93 <= median <= 0.97
        assert observed / median >= 0.90
        # And the correction is what B3 needs: it closes at least half the gap
        # to the B1/B2 median of 0.955 without any free parameter.
        assert (observed / median) > observed + 0.5 * (0.955 - observed)

    def test_gas_corrected_ci_contains_the_093_gate(self):
        from satharmony.emulators.s2_to_mss.sbaf import PREDICTORS  # noqa: F401

        water = _load("pics_water_vapour.json")
        s_gas = sorted(site["bands"]["B3"]["s_gas"] for site in water["sites"].values())
        median = s_gas[len(s_gas) // 2]

        regression = _load("sbaf_pics_regression.json")
        low, high = regression["bands"]["B3"]["scale_ci"]

        assert not (low <= 0.93 <= high), "the raw CI already excludes nothing"
        assert low / median <= 0.93 <= high / median

    def test_band4_correction_matches_the_fitted_gamma(self):
        """The physical gas correction and the fitted gamma agree in size.

        Before the water-vapour term, MSS4 sat at 0.820; the gamma of 0.112
        lifts it to 0.933. 6SV on the same PICS gives a B4 gas scale of 0.860
        (median per site), i.e. 0.820 / 0.860 = 0.95 without any free
        parameter. Two independent routes to the same magnitude.
        """
        water = _load("pics_water_vapour.json")
        s_gas = sorted(site["bands"]["B4"]["s_gas"] for site in water["sites"].values())
        median = s_gas[len(s_gas) // 2]

        assert 0.84 <= median <= 0.90
        corrected = 0.820 / median
        assert 0.92 <= corrected <= 0.97


class TestPicsWaterVapour:
    """ERA5 says the five PICS are a dry, narrow sample: gamma is unconstrained."""

    def test_all_sites_are_dry(self):
        water = _load("pics_water_vapour.json")
        for site, block in water["sites"].items():
            for date, values in (("mss", block["era5"]["mss"]), ("s2", block["era5"]["s2"])):
                assert values["tcwv_cm"] < 2.5, f"{site} {date} is not a dry site"

    def test_column_range_is_narrow(self):
        water = _load("pics_water_vapour.json")
        low, high = water["tcwv_range_cm"]
        assert high - low < 1.5

    def test_scale_height_sensitivity_is_bounded(self):
        water = _load("pics_water_vapour.json")
        for block in water["sites"].values():
            a = block["era5"]["mss"]["tcwv_cm_hq15"]
            b = block["era5"]["mss"]["tcwv_cm_hq25"]
            # +-25 % on the assumed humidity scale height.
            assert 1.15 < b / a < 1.9

    def test_egypt1_is_the_outlier_in_every_band(self):
        """Egypt-1 is the minimum of the five sites in all four bands (p=1/625)."""
        scaling = _load("sbaf_pics_band_scaling.json")
        for band in MSS_BANDS:
            per_site = scaling["bands"][band]["per_site"]
            assert min(per_site, key=per_site.get) == "egypt1", band

    def test_four_site_medians(self):
        """What the median would be without Egypt-1 (not applied to the profile)."""
        scaling = _load("sbaf_pics_band_scaling.json")
        expected = {"B1": 0.95955, "B2": 0.93475, "B3": 0.87865, "B4": 0.93685}
        for band, value in expected.items():
            per_site = dict(scaling["bands"][band]["per_site"])
            per_site.pop("egypt1")
            values = sorted(per_site.values())
            median = (values[1] + values[2]) / 2
            assert median == pytest.approx(value, abs=1e-4)

    def test_closing_egypt1_does_not_reach_the_0_93_gate(self):
        """The exclusion alone is not the fix for B3: it moves it to 0.879."""
        scaling = _load("sbaf_pics_band_scaling.json")
        per_site = dict(scaling["bands"]["B3"]["per_site"])
        per_site.pop("egypt1")
        values = sorted(per_site.values())
        four_site_median = (values[1] + values[2]) / 2
        assert four_site_median < 0.93
