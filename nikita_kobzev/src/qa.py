"""Fail loudly on integrity errors; preserve genuine negative/extreme prices."""
import numpy as np
import pandas as pd
from .data import TARGET, TZ, checksum, write_json


def run_qa(df, root):
    out = root / "outputs"
    checks = []

    def check(name, passed, detail=""):
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    t = df.index
    local = t.tz_convert(TZ)
    check("unique_utc", t.is_unique)
    check("sorted_utc", t.is_monotonic_increasing)
    check("continuous_hourly_utc", (t[1:] - t[:-1] == pd.Timedelta(hours=1)).all())
    check("hour_aligned", (t.minute == 0).all() and (t.second == 0).all())
    physical = [c for c in df if c.endswith("_mwh") and "error" not in c
                and c != "residual_load_forecast_mwh" and c != TARGET]
    main = [TARGET] + physical
    check("main_values_present", df[main].notna().all().all())
    check("main_values_finite", np.isfinite(df[main].to_numpy()).all())
    check("nonnegative_physical", (df[physical] >= 0).all().all())
    formulas = {
        "wind_actual_mwh": df.wind_onshore_actual_mwh + df.wind_offshore_actual_mwh,
        "wind_forecast_mwh": df.wind_onshore_forecast_mwh + df.wind_offshore_forecast_mwh,
        "load_error_mwh": df.load_actual_mwh - df.load_forecast_mwh,
        "wind_error_mwh": df.wind_actual_mwh - df.wind_forecast_mwh,
        "solar_error_mwh": df.solar_actual_mwh - df.solar_forecast_mwh,
        "residual_load_forecast_mwh": df.load_forecast_mwh - df.wind_forecast_mwh - df.solar_forecast_mwh,
    }
    for col, expected in formulas.items():
        check("arithmetic_" + col, np.allclose(df[col], expected, atol=1e-6, rtol=1e-10))
    for col, source, hours in [("price_lag_24", TARGET, 24), ("price_lag_168", TARGET, 168),
                                ("wind_actual_lag_24", "wind_actual_mwh", 24),
                                ("solar_actual_lag_24", "solar_actual_mwh", 24)]:
        expected = df[source].reindex(t - pd.Timedelta(hours=hours)).to_numpy()
        check("stored_lag_" + col, np.allclose(df[col], expected, equal_nan=True),
              f"Expected {hours} initial unavailable hours")
    for col, expected in {"hour": local.hour, "day_of_week": local.dayofweek,
                          "month": local.month, "is_weekend": (local.dayofweek >= 5).astype(int)}.items():
        check("calendar_" + col, np.array_equal(df[col], expected))
    days = pd.Series(1, index=local).groupby(local.date).sum()
    expected_days = {day: len(pd.date_range(pd.Timestamp(day, tz=TZ),
                    pd.Timestamp(day, tz=TZ) + pd.DateOffset(days=1), inclusive="left", freq="h"))
                     for day in days.index}
    check("complete_local_days", all(n == expected_days[day] for day, n in days.items()))
    # Repeated wall-clock timestamps are permitted only on the autumn fold.
    naive = local.tz_localize(None)
    check("local_fold_offsets", all(local[naive == x].nunique() == 2
          for x in naive[naive.duplicated()].unique()))
    pd.DataFrame(checks).to_csv(out / "qa_summary.csv", index=False)
    df.select_dtypes("number").describe().T.to_csv(out / "data_summary.csv")
    schema = [{"column": c, "dtype": str(df[c].dtype), "null_count": int(df[c].isna().sum())} for c in df]
    pd.DataFrame(schema).to_csv(out / "data_schema.csv", index=False)
    dst = {str(k): int(v) for k, v in days.items() if v != 24}
    summary = {"rows": len(df), "columns": len(df.columns), "start_local": str(local.min()),
               "end_local": str(local.max()), "sha256": checksum(root / "data/de_power_2y_clean.csv"),
               "dst_days": dst, "null_counts": {c: int(n) for c, n in df.isna().sum().items() if n},
               "warnings": ["CSV has no original forecast issue/revision timestamps.",
                            "Historical actuals may contain later revisions; no vintage-certified backtest is claimed."]}
    write_json(out / "qa_details.json", summary)
    failed = [r["check"] for r in checks if not r["passed"]]
    if failed:
        raise ValueError("QA failed: " + ", ".join(failed))
    return summary
