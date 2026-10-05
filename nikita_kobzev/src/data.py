"""Input contracts and stable serialization. No network in the default pipeline."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "price_da_eur_mwh"
TZ = "Europe/Berlin"


def write_json(path: Path, value) -> None:
    def convert(x):
        if isinstance(x, (np.integer, np.floating, np.bool_)):
            return x.item()
        if isinstance(x, (pd.Timestamp, Path)):
            return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=convert,
                               allow_nan=False) + "\n", encoding="utf-8")


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["datetime_utc"], utc=True, errors="raise")
    if ts.duplicated().any() or not ts.is_monotonic_increasing:
        raise ValueError("Input timestamps must be unique and already sorted")
    if not (pd.to_datetime(df["datetime"], utc=True) == ts).all():
        raise ValueError("Local timestamps disagree with UTC")
    df.index = pd.DatetimeIndex(ts, name="timestamp")
    return df
