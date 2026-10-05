"""Optional acquisition reconstruction from the owner's endpoint/module specification.

Not the original build_de_power_dataset_v2.py (not supplied). Never overwrites
the primary input. Full local days are required; the expected UTC timeline
resolves repeated autumn hours without an ambiguous naive-time merge.
"""
import argparse
from io import StringIO
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

ENDPOINT = "https://www.smard.de/nip-download-manager/nip/download/market-data"
MODULES = {8004169: "price_da_eur_mwh", 5000410: "load_actual_mwh", 6000411: "load_forecast_mwh",
           1004067: "wind_onshore_actual_mwh", 1001225: "wind_offshore_actual_mwh",
           1004068: "solar_actual_mwh", 2000123: "wind_onshore_forecast_mwh",
           2003791: "wind_offshore_forecast_mwh", 2000125: "solar_forecast_mwh"}


def parse_series(text, index, name):
    raw = pd.read_csv(StringIO(text.lstrip("\ufeff")), sep=";", dtype=str)
    if "Datum von" not in raw or "Datum bis" not in raw or len(raw.columns) != 3:
        raise ValueError("Unexpected SMARD CSV schema")
    expected = index.tz_convert("Europe/Berlin").strftime("%d.%m.%Y %H:%M")
    if len(raw) != len(index) or not np.array_equal(raw["Datum von"].to_numpy(), expected):
        raise ValueError("SMARD delivery rows do not match the expected UTC/DST timeline")
    values = raw.iloc[:, 2].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    numeric = pd.to_numeric(values, errors="raise")
    if not np.isfinite(numeric).all():
        raise ValueError("Incomplete SMARD data")
    return pd.Series(numeric.to_numpy(), index=index, name=name)


def fetch(start, end):
    start, end = pd.Timestamp(start, tz="Europe/Berlin"), pd.Timestamp(end, tz="Europe/Berlin")
    if start != start.normalize() or end != end.normalize() or end <= start:
        raise ValueError("Use an increasing interval of complete local dates; end is exclusive")
    index = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    series = []
    for module, name in MODULES.items():
        payload = {"request_form": [{"format": "CSV", "moduleIds": [module], "region": "DE",
            "timestamp_from": int(start.timestamp() * 1000), "timestamp_to": int(end.timestamp() * 1000) - 1,
            "type": "discrete", "language": "de", "resolution": "hour"}]}
        response = requests.post(ENDPOINT, json=payload, timeout=(10, 90))
        response.raise_for_status()
        series.append(parse_series(response.content.decode("utf-8-sig"), index, name))
        print(f"Fetched {name}: {len(index)} hours", flush=True)
    df = pd.concat(series, axis=1, verify_integrity=True)
    for kind in ["actual", "forecast"]:
        df[f"wind_{kind}_mwh"] = df[f"wind_onshore_{kind}_mwh"] + df[f"wind_offshore_{kind}_mwh"]
    for driver in ["load", "wind", "solar"]:
        df[f"{driver}_error_mwh"] = df[f"{driver}_actual_mwh"] - df[f"{driver}_forecast_mwh"]
    df["residual_load_forecast_mwh"] = df.load_forecast_mwh - df.wind_forecast_mwh - df.solar_forecast_mwh
    for name, source, lag in [("price_lag_24", "price_da_eur_mwh", 24), ("price_lag_168", "price_da_eur_mwh", 168),
                              ("wind_actual_lag_24", "wind_actual_mwh", 24), ("solar_actual_lag_24", "solar_actual_mwh", 24)]:
        df[name] = df[source].reindex(index - pd.Timedelta(hours=lag)).to_numpy()
    local = index.tz_convert("Europe/Berlin")
    df["hour"], df["day_of_week"], df["month"] = local.hour, local.dayofweek, local.month
    df["is_weekend"] = (local.dayofweek >= 5).astype(int)
    df.insert(0, "datetime_utc", index.astype(str))
    df.insert(0, "datetime", local.astype(str))
    df["Datum von"] = local.strftime("%d.%m.%Y %H:%M")
    df["Datum bis"] = (index + pd.Timedelta(hours=1)).tz_convert("Europe/Berlin").strftime("%d.%m.%Y %H:%M")
    df["_occurrence"] = df.groupby("Datum von").cumcount()
    assert df.index.is_unique and not df[list(MODULES.values())].isna().any().any()
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-09-27")
    parser.add_argument("--end", default="2026-09-27", help="Exclusive local date")
    parser.add_argument("--output", default=str(Path(__file__).with_name("de_power_refetched.csv")))
    args = parser.parse_args()
    output = Path(args.output)
    if output.resolve() == Path(__file__).with_name("de_power_2y_clean.csv").resolve():
        raise SystemExit("Refusing to overwrite the supplied primary dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    fetch(args.start, args.end).to_csv(output, index=False)
    print("Saved", output)
