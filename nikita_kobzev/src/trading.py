"""Delivery-matched DA-to-curve re-marking, never fictional trading P&L."""
import numpy as np
import pandas as pd
from .data import TARGET, TZ, write_json
from .features import origin_times

CURVE_COLUMNS = ["observed_at", "delivery_start", "delivery_end", "product", "price_eur_mwh", "source_url"]


def daily_values(table, winner):
    local = pd.DatetimeIndex(pd.to_datetime(table.datetime, utc=True)).tz_convert(TZ)
    temporary = pd.DataFrame({"day": local.strftime("%Y-%m-%d"), "prediction": table[winner].to_numpy(),
                              "actual": table.y_true.to_numpy(),
                              "weekly_baseline": table.weekly_persistence.to_numpy()})
    return temporary.groupby("day").agg(hours=("prediction", "size"), fair_value=("prediction", "mean"),
                  actual=("actual", "mean"), weekly_baseline=("weekly_baseline", "mean")).reset_index()


def load_curve(path):
    curve = pd.read_csv(path)
    if list(curve) != CURVE_COLUMNS:
        raise ValueError("Curve input must have exactly " + ",".join(CURVE_COLUMNS))
    if curve.empty:
        return curve
    if curve.isna().any().any() or not np.isfinite(curve.price_eur_mwh).all():
        raise ValueError("Missing or non-finite curve values")
    for c in ["observed_at", "delivery_start", "delivery_end"]:
        # Reject naive dates before converting; never assume UTC implicitly.
        if any(pd.Timestamp(v).tzinfo is None for v in curve[c]):
            raise ValueError("Curve timestamps must include timezone offsets")
        curve[c] = pd.to_datetime(curve[c], utc=True)
    if not curve["product"].isin(["DE_BASE_WEEK", "DE_BASE_MONTH"]).all():
        raise ValueError("Only DE base week/month contracts supported")
    if not curve.source_url.str.startswith("https://").all():
        raise ValueError("Public source URL required")
    if curve.duplicated(["observed_at", "delivery_start", "delivery_end", "product"]).any():
        raise ValueError("Duplicate curve observations")
    for row in curve.itertuples():
        start, end = row.delivery_start.tz_convert(TZ), row.delivery_end.tz_convert(TZ)
        if start != start.normalize() or end != end.normalize():
            raise ValueError("Base contract delivery boundaries must be local midnight")
        expected = start + (pd.DateOffset(days=7) if row.product == "DE_BASE_WEEK" else pd.DateOffset(months=1))
        if end != expected or (row.product == "DE_BASE_WEEK" and start.dayofweek != 0) or (row.product == "DE_BASE_MONTH" and start.day != 1):
            raise ValueError("Wrong week/month delivery boundaries")
    return curve


def curve_revisions(df, oos, winner, curve, config):
    """At each D-1 origin, replace just the forecast day in a curve-anchored
    last-week shape. Other contract hours retain their curve-implied values.
    Quotes after the forecast origin or delivery start are never usable.
    """
    rows = []
    if curve.empty:
        return rows
    daily = daily_values(oos, winner)
    for day in daily.itertuples():
        start = pd.Timestamp(day.day, tz=TZ)
        end = start + pd.DateOffset(days=1)
        origin = origin_times(pd.DatetimeIndex([start]), config["forecast_hour"])[0]
        usable = curve[(curve.observed_at <= origin) & (curve.observed_at >= origin - pd.Timedelta(hours=120))
            & (curve.delivery_start >= origin) & (curve.delivery_start <= start.tz_convert("UTC"))
            & (curve.delivery_end >= end.tz_convert("UTC"))].sort_values("observed_at")
        for _, quote in usable.groupby(["product", "delivery_start", "delivery_end"]).tail(1).iterrows():
            hours = pd.date_range(quote.delivery_start, quote.delivery_end, freq="h", inclusive="left")
            # Previous seven full local delivery days ending at D-1. All their DA
            # prices were published by D-2 afternoon, before this origin.
            history = df[(df.index >= (start - pd.DateOffset(days=7)).tz_convert("UTC")) & (df.index < start.tz_convert("UTC"))]
            hist_local = history.index.tz_convert(TZ)
            shape = history[TARGET].groupby([hist_local.dayofweek, hist_local.hour]).mean()
            period_local = hours.tz_convert(TZ)
            baseline = shape.reindex(pd.MultiIndex.from_arrays([period_local.dayofweek, period_local.hour])).to_numpy()
            if np.isnan(baseline).any():
                continue  # Incomplete historical shape, e.g. a missing spring hour.
            implied = quote.price_eur_mwh + baseline - baseline.mean()
            target = (hours >= start.tz_convert("UTC")) & (hours < end.tz_convert("UTC"))
            implied_day = float(implied[target].mean())
            spread = float(day.fair_value - implied_day)
            contribution = float(target.sum() / len(hours) * spread)
            rows.append({"forecast_origin": origin.isoformat(), "day": day.day,
                         "product": quote["product"], "observed_at": quote.observed_at.isoformat(),
                         "delivery_start": quote.delivery_start.isoformat(), "delivery_end": quote.delivery_end.isoformat(),
                         "source_url": quote.source_url, "curve_price": float(quote.price_eur_mwh),
                         "contract_hours": len(hours), "forecast_hours": int(target.sum()),
                         "curve_implied_day": implied_day, "day_fair_value": float(day.fair_value),
                         "day_spread": spread, "conditional_curve_revision": contribution,
                         "conditional_curve_mark": float(quote.price_eur_mwh + contribution)})
    return rows


def build_trading(df, validation, oos, selection, config, root):
    winner = selection["winner"]
    vd, od = daily_values(validation, winner), daily_values(oos, winner)
    q80 = float((vd.fair_value - vd.actual).abs().quantile(0.8))
    od["validation_abs_daily_error_q80"] = q80
    od["reference_lower"] = od.fair_value - q80
    od["reference_upper"] = od.fair_value + q80
    od["model_revision_vs_weekly_baseline"] = od.fair_value - od.weekly_baseline
    od.to_csv(root / "outputs/daily_fair_value.csv", index=False)
    curve = load_curve(root / "data/prompt_curve.csv")
    revisions = curve_revisions(df, oos, winner, curve, config)
    write_json(root / "outputs/curve_revisions.json", revisions)
    last = od.iloc[-1]
    summary = {"status": "conditional_marks_available" if revisions else "no_usable_curve_quotes",
        "instrument": "EEX German Power Base Week / Month, if independently timestamped public quotes are supplied",
        "last_delivery_day": last.day, "last_day_fair_value_eur_mwh": float(last.fair_value),
        "validation_daily_absolute_error_q80": q80,
        "latest_revision_vs_weekly_baseline": float(last.model_revision_vs_weekly_baseline),
        "matched_quote_count": len(revisions), "reference_source": "https://www.eex.com/en/market-data/market-data-hub",
        "curve_formula": "F_new = F_quote + H_day/H_contract * (FV_day - curve_implied_day)",
        "shape": "Previous seven local delivery days, shifted additively to average to the observed contract quote",
        "interpretation": "Positive revision is upward pressure on conditional curve value; negative is downward. Not an executable trade signal or independent full-period forecast.",
        "limitations": ["One-day forecasts produced at different origins cannot be pooled into a same-origin front-week/month forecast.",
                        "Quoted futures contain risk premia and bid/ask costs; settlement marks are not executable bids or offers.",
                        "q80 is a descriptive validation error reference, not a calibrated confidence interval.",
                        "No trading P&L or proven strategy is claimed."],
        "invalidation": ["Material load/renewable forecast revision or plant outage", "Large gas/carbon move",
                         "Input outside training support", "Recent forecast errors exceed validation reference",
                         "Quote stale, observed after forecast, or contract delivery mismatched"],
        "quote_status_detail": "No verified public historical contract quote was acquired; supplied header is an input contract, not fabricated market data." if not revisions else "See curve_revisions.json"}
    write_json(root / "outputs/trading_view.json", summary)
    return summary
