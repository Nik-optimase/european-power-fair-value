"""Chronological holdout plus a separate frozen structural-break diagnostic."""
import numpy as np
import pandas as pd
from .data import TARGET, TZ, write_json
from .features import FEATURES, origin_times
from .models import fit_predict, make_model


def metrics(y, pred):
    error = np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
    if not len(error):
        return {"n": 0, "mae": None, "rmse": None, "bias": None}
    assert np.isfinite(error).all()
    return {"n": len(error), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(np.mean(error))}


def splits(x, config):
    start = pd.Timestamp(config["validation_start"], tz=TZ).tz_convert("UTC")
    oos = pd.Timestamp(config["oos_start"], tz=TZ).tz_convert("UTC")
    end = pd.Timestamp(config["oos_end_exclusive"], tz=TZ).tz_convert("UTC")
    if any(t.tz_convert(TZ) != t.tz_convert(TZ).normalize() for t in [start, oos, end]):
        raise ValueError("Split boundaries must be local midnight for complete next-day forecasts")
    if not x.index.min() < start < oos < end <= x.index.max() + pd.Timedelta(hours=1):
        raise ValueError("Split boundaries must increase within the available dataset")
    valid = x.notna().all(axis=1)
    masks = {"train": valid & (x.index < start),
             "validation": valid & (x.index >= start) & (x.index < oos),
             "oos": valid & (x.index >= oos) & (x.index < end)}
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"Empty {name} split")
    for name, left, right in [("validation", start, oos), ("oos", oos, end)]:
        if masks[name].sum() != len(pd.date_range(left, right, freq="h", inclusive="left")):
            raise ValueError(f"Incomplete {name} window after feature warm-up")
    assert x.index[masks["train"]].max() < x.index[masks["validation"]].min()
    assert x.index[masks["validation"]].max() < x.index[masks["oos"]].min()
    return masks


def compare_models(df, x, config, root):
    masks = splits(x, config)
    tr, va = masks["train"], masks["validation"]
    ytr, yva = df.loc[tr, TARGET], df.loc[va, TARGET]
    comparison, models, predictions, candidates = [], {}, {}, []
    best_params, best_cat_score = None, float("inf")
    for params in config["catboost_candidates"]:
        pred, model = fit_predict("catboost", config, x.loc[tr], ytr, x.loc[va], params)
        result = metrics(yva, pred)
        candidates.append({**params, **result})
        if result["mae"] < best_cat_score:
            best_params, best_cat_score = params, result["mae"]
            models["catboost"], predictions["catboost"] = model, pred
    for name in ["daily_persistence", "weekly_persistence", "linear_regression", "catboost"]:
        if name != "catboost":
            predictions[name], models[name] = fit_predict(name, config, x.loc[tr], ytr, x.loc[va])
        comparison.append({"model": name, **metrics(yva, predictions[name])})
    scores = pd.DataFrame(comparison).sort_values(["mae", "rmse", "model"])
    winner = str(scores.iloc[0].model)
    out = root / "outputs"
    scores.to_csv(out / "model_comparison.csv", index=False)
    pd.DataFrame(candidates).to_csv(out / "catboost_candidates.csv", index=False)
    vp = pd.DataFrame({"datetime": x.index[va], "y_true": yva.to_numpy(), **predictions})
    vp.to_csv(out / "validation_predictions.csv", index=False)
    periods = {name: {"start": str(x.index[mask].min().tz_convert(TZ)),
                      "end_inclusive": str(x.index[mask].max().tz_convert(TZ)), "rows": int(mask.sum())}
               for name, mask in masks.items()}
    selection = {"winner": winner, "selection_rule": "Lowest validation MAE; RMSE and name break ties",
                 "catboost_parameters": best_params, "seed": config["seed"], "periods": periods,
                 "oos_dates_status": "Researcher-selected historical holdout; employer supplied no exact dates",
                 "features": FEATURES, "validation_metrics": comparison}
    # Freeze the selection artifact BEFORE any final OOS scoring.
    write_json(out / "selection.json", selection)
    importance = pd.DataFrame({"feature": FEATURES,
                      "importance": models["catboost"].get_feature_importance()}).sort_values("importance", ascending=False)
    importance.to_csv(out / "feature_importance.csv", index=False)
    return masks, selection, vp, importance


def regime_metrics(predictions, threshold):
    t = pd.DatetimeIndex(pd.to_datetime(predictions.datetime, utc=True)).tz_convert(TZ)
    y = predictions.y_true.to_numpy()
    regimes = {"all": np.ones(len(t), dtype=bool), "negative_price": y < 0,
        f"high_price_gt_{threshold:g}": y > threshold, "weekday": t.dayofweek < 5, "weekend": t.dayofweek >= 5,
        "peak": (t.dayofweek < 5) & (t.hour >= 8) & (t.hour < 20),
        "offpeak": ~((t.dayofweek < 5) & (t.hour >= 8) & (t.hour < 20))}
    rows = []
    for name, mask in regimes.items():
        for model in [c for c in predictions if c not in {"datetime", "y_true"}]:
            rows.append({"regime": name, "model": model, **metrics(y[mask], predictions.loc[mask, model])})
    return pd.DataFrame(rows)


def structural_diagnostic(df, x, config):
    """Two adjacent historical test months, single fit, no parameter selection."""
    a = pd.Timestamp("2025-09-01", tz=TZ).tz_convert("UTC")
    b = pd.Timestamp("2025-10-01", tz=TZ).tz_convert("UTC")
    c = pd.Timestamp("2025-11-01", tz=TZ).tz_convert("UTC")
    valid = x.notna().all(axis=1)
    train = valid & (x.index < a)
    records = []
    if not train.any() or pd.Timestamp(config["validation_start"], tz=TZ).tz_convert("UTC") < c:
        return pd.DataFrame([{"status": "not_available_before_validation"}])
    for name in ["daily_persistence", "weekly_persistence", "catboost"]:
        test = valid & (x.index >= a) & (x.index < c)
        pred, _ = fit_predict(name, config, x.loc[train], df.loc[train, TARGET], x.loc[test], config["catboost_candidates"][0])
        for label, mask in [("September_2025", x.index[test] < b), ("October_2025", x.index[test] >= b)]:
            records.append({"regime": label, "model": name,
                            **metrics(df.loc[test, TARGET].to_numpy()[mask], pred[mask])})
    return pd.DataFrame(records)


def final_oos(df, x, masks, config, selection, root):
    train = masks["train"] | masks["validation"]
    test = masks["oos"]
    # Every refit label has already cleared by the first OOS forecast origin.
    train_days = x.index[train].tz_convert(TZ).tz_localize(None).normalize()
    published = (train_days - pd.Timedelta(days=1) + pd.Timedelta(hours=14)).tz_localize(TZ).tz_convert("UTC")
    assert (published <= origin_times(x.index[test], config["forecast_hour"])[0]).all()
    results, scores = {}, []
    for name in ["daily_persistence", "weekly_persistence", "linear_regression", "catboost"]:
        pred, _ = fit_predict(name, config, x.loc[train], df.loc[train, TARGET], x.loc[test],
                              selection["catboost_parameters"])
        results[name] = pred
        scores.append({"model": name, **metrics(df.loc[test, TARGET], pred)})
    winner = selection["winner"]  # Never change this using the scores above.
    detail = pd.DataFrame({"datetime": x.index[test], "y_true": df.loc[test, TARGET].to_numpy(), **results})
    detail.to_csv(root / "outputs/oos_predictions_with_actuals.csv", index=False)
    pd.DataFrame(scores).to_csv(root / "outputs/oos_metrics.csv", index=False)
    required = pd.DataFrame({"datetime": [t.isoformat() for t in x.index[test]], "y_pred": results[winner]})
    assert list(required) == ["datetime", "y_pred"]
    assert not required.isna().any().any() and np.isfinite(required.y_pred).all()
    assert required.datetime.is_unique
    required.to_csv(root / "predictions.csv", index=False)
    return detail, scores
