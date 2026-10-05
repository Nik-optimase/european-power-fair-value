"""Whitelist of deterministic validation-only model-review operations."""
import numpy as np
import pandas as pd
from .data import TARGET, write_json
from .features import FEATURES, CATEGORICAL
from .models import fit_predict
from .validation import metrics, regime_metrics, structural_diagnostic

ALLOWED = ["negative_price_analysis", "high_price_analysis", "weekday_weekend_analysis",
           "peak_offpeak_analysis", "pre_post_2025_10_market_change", "feature_ablation",
           "train_validation_drift", "largest_error_analysis"]
DEFAULT = ["negative_price_analysis", "pre_post_2025_10_market_change", "feature_ablation"]


def feature_drift(x, masks):
    train, val = x.loc[masks["train"]], x.loc[masks["validation"]]
    result = []
    for col in [c for c in FEATURES if c not in CATEGORICAL]:
        sd = train[col].std()
        result.append({"feature": col, "train_mean": float(train[col].mean()),
                       "validation_mean": float(val[col].mean()),
                       "standardized_mean_shift": float((val[col].mean() - train[col].mean()) / sd) if sd > 0 else 0.0})
    return pd.DataFrame(result)


def worst_errors(vp, winner):
    table = vp[["datetime", "y_true", winner]].copy()
    table["absolute_error"] = (table[winner] - table.y_true).abs()
    table["datetime"] = table.datetime.astype(str)
    return table.nlargest(10, "absolute_error").to_dict("records")


def run_selected(names, df, x, masks, config, selection, vp, regimes, drift, root):
    if len(names) > 3 or len(set(names)) != len(names) or not set(names) <= set(ALLOWED):
        raise ValueError("Diagnostics must be unique whitelisted names; maximum 3")
    answers = {}
    groups = {"negative_price_analysis": ["negative_price"],
              "high_price_analysis": [f"high_price_gt_{config['high_price_threshold']:g}"],
              "weekday_weekend_analysis": ["weekday", "weekend"], "peak_offpeak_analysis": ["peak", "offpeak"]}
    for name in names:
        if name in groups:
            answers[name] = regimes[regimes.regime.isin(groups[name])].to_dict("records")
        elif name == "train_validation_drift":
            answers[name] = drift.to_dict("records")
        elif name == "largest_error_analysis":
            answers[name] = worst_errors(vp, selection["winner"])
        elif name == "pre_post_2025_10_market_change":
            answers[name] = structural_diagnostic(df, x, config).to_dict("records")
        elif name == "feature_ablation":
            # This is an explanatory validation experiment, never an automatic model edit.
            tr, va = masks["train"], masks["validation"]
            columns = [c for c in FEATURES if "actual_lag" not in c]
            pred, _ = fit_predict("catboost", config, x.loc[tr, columns], df.loc[tr, TARGET],
                                  x.loc[va, columns], selection["catboost_parameters"])
            answers[name] = {"experiment": "CatBoost without all physical drivers; same parameters",
                            "features": columns, **metrics(df.loc[va, TARGET], pred),
                            "selection_effect": "none; review-only"}
    # Replace NaN for empty regimes before strict JSON encoding.
    import json
    answers = json.loads(pd.Series(answers).to_json(date_format="iso"))
    write_json(root / "outputs/selected_diagnostics.json", answers)
    return answers
