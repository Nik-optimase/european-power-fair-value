"""Forecast-origin-aware features; all origins are D-1 11:00 local time.

Price calendar lags use the same local delivery hour. On a spring missing
hour, interpolate only within the already-published historical day. On an
autumn fold, average the two source hours. Actual-generation lags instead
mean exactly 48/168 elapsed hours, explicitly documented.
"""
import numpy as np
import pandas as pd
from .data import TARGET, TZ

CATEGORICAL = ["hour", "day_of_week", "month"]
FEATURES = ["price_prev_day_same_hour", "price_prev_week_same_hour",
            "price_prev_day_mean", "price_prev_day_std",
            "load_actual_lag_48", "load_actual_lag_168",
            "wind_actual_lag_48", "wind_actual_lag_168",
            "solar_actual_lag_48", "solar_actual_lag_168",
            "hour", "day_of_week", "month", "is_weekend"]
FORBIDDEN = {TARGET, "load_forecast_mwh", "wind_forecast_mwh", "solar_forecast_mwh",
             "residual_load_forecast_mwh", "load_actual_mwh", "wind_actual_mwh", "solar_actual_mwh",
             "load_error_mwh", "wind_error_mwh", "solar_error_mwh",
             "wind_actual_lag_24", "solar_actual_lag_24"}


def origin_times(index, hour=11):
    naive_days = index.tz_convert(TZ).normalize().tz_localize(None)
    return (naive_days - pd.Timedelta(days=1) + pd.Timedelta(hours=hour)).tz_localize(TZ).tz_convert("UTC")


def calendar_price_lag(df, target_index, days):
    local_source = df.index.tz_convert(TZ)
    profile = pd.DataFrame({"date": local_source.tz_localize(None).normalize(),
                            "hour": local_source.hour, "value": df[TARGET].to_numpy()})
    table = profile.pivot_table(index="date", columns="hour", values="value", aggfunc="mean")
    table = table.reindex(columns=range(24)).interpolate(axis=1, limit_area="inside")
    target_local = target_index.tz_convert(TZ)
    source_dates = target_local.tz_localize(None).normalize() - pd.Timedelta(days=days)
    keys = pd.MultiIndex.from_arrays([source_dates, target_local.hour])
    values = table.stack().reindex(keys).to_numpy()
    # Conservative fixed assumed publication by 14:00 on the prior local day.
    available = (source_dates - pd.Timedelta(days=1) + pd.Timedelta(hours=14)).tz_localize(TZ).tz_convert("UTC")
    return values, available


def build_features(df, config, target_index=None):
    t = df.index if target_index is None else target_index
    local = t.tz_convert(TZ)
    origins = origin_times(t, config["forecast_hour"])
    x = pd.DataFrame(index=t)
    audit = []

    def record(name, available, values):
        valid = pd.notna(values)
        margin = (origins[valid] - available[valid]).total_seconds() / 3600
        violations = int((margin < 0).sum())
        if violations:
            raise ValueError(f"Future information in {name}: {violations} rows")
        audit.append({"feature": name, "available_rows": int(valid.sum()),
                      "timing_violations": violations,
                      "minimum_margin_hours": float(margin.min()) if len(margin) else None})

    for days, name in [(1, "price_prev_day_same_hour"), (7, "price_prev_week_same_hour")]:
        values, available = calendar_price_lag(df, t, days)
        x[name] = values
        record(name, available, values)
    prices = pd.Series(df[TARGET].to_numpy(), index=df.index.tz_convert(TZ))
    daily = prices.groupby(prices.index.tz_localize(None).normalize()).agg(["mean", "std"])
    source_dates = local.tz_localize(None).normalize() - pd.Timedelta(days=1)
    available = (source_dates - pd.Timedelta(days=1) + pd.Timedelta(hours=14)).tz_localize(TZ).tz_convert("UTC")
    for stat in ["mean", "std"]:
        name = "price_prev_day_" + stat
        x[name] = daily[stat].reindex(source_dates).to_numpy()
        record(name, available, x[name].to_numpy())
    delay = config["actual_publication_delay_hours"]
    if delay < 1:
        raise ValueError("Physical publication delay must be at least one hour after interval end")
    for driver in ["load", "wind", "solar"]:
        for lag in [48, 168]:
            name = f"{driver}_actual_lag_{lag}"
            source_time = t - pd.Timedelta(hours=lag)
            x[name] = df[f"{driver}_actual_mwh"].reindex(source_time).to_numpy()
            available = source_time + pd.Timedelta(hours=1 + delay)
            record(name, available, x[name].to_numpy())
    x["hour"] = local.hour
    x["day_of_week"] = local.dayofweek
    x["month"] = local.month
    x["is_weekend"] = (local.dayofweek >= 5).astype(int)
    for col in CATEGORICAL:
        x[col] = x[col].astype(str)
    assert not set(FEATURES) & FORBIDDEN
    assert not any("error" in c or "forecast_mwh" in c for c in FEATURES)
    return x[FEATURES], origins, pd.DataFrame(audit)
