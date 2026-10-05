"""Compact training-only EDA and clearly labelled held-out diagnostics."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data import TARGET, TZ, write_json
from .validation import metrics

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "font.size": 10,
                    "axes.spines.top": False, "axes.spines.right": False,
                    "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.15})
BLUE, ORANGE = "#23618b", "#d7813b"


def save(fig, root, name):
    fig.tight_layout(pad=1.8)
    fig.savefig(root / "figures" / name, facecolor="white")
    plt.close(fig)


def eda(df, train_mask, root):
    train = df.loc[train_mask].copy()
    local = train.index.tz_convert(TZ)
    y = train[TARGET]
    stats = {"scope": "training only", "n": len(y), "mean": float(y.mean()),
             "median": float(y.median()), "std": float(y.std()), "min": float(y.min()),
             "max": float(y.max()), "negative_share": float((y < 0).mean()),
             "above_200_count": int((y > 200).sum())}
    write_json(root / "outputs/training_price_statistics.json", stats)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
    axes[0].plot(local, y, color=BLUE, linewidth=0.5, alpha=0.8)
    axes[0].set(title="Germany/Luxembourg hourly day-ahead price · training period", ylabel="EUR/MWh")
    axes[1].hist(y, bins=95, color=BLUE, alpha=0.9)
    axes[1].axvline(0, color=ORANGE, linewidth=1)
    axes[1].set(title="Full training distribution · negative and extreme prices retained", xlabel="EUR/MWh", ylabel="Hours")
    save(fig, root, "price_overview.png")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, grouping, name in zip(axes, [local.hour, local.dayofweek, local.month], ["Hour", "Weekday (Monday = 0)", "Month"]):
        series = y.groupby(grouping).mean()
        ax.bar(series.index, series, color=BLUE)
        ax.set(title=f"Training mean by {name.lower()}", xlabel=name, ylabel="EUR/MWh")
    save(fig, root, "calendar_profiles.png")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, col, label in zip(axes, ["load_actual_mwh", "wind_actual_mwh", "solar_actual_mwh"], ["Load", "Wind", "Solar"]):
        ax.hexbin(train[col] / 1000, y, gridsize=35, mincnt=1, bins="log", cmap="Blues")
        ax.set(xlabel=f"{label} (GWh per delivery hour)", ylabel="EUR/MWh", title=f"{label} vs price · descriptive only")
    save(fig, root, "fundamental_relationships.png")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    forecast_stats = []
    for ax, driver in zip(axes, ["load", "wind", "solar"]):
        actual, forecast = train[driver + "_actual_mwh"], train[driver + "_forecast_mwh"]
        ax.hexbin(actual / 1000, forecast / 1000, gridsize=35, mincnt=1, bins="log", cmap="Blues")
        hi = max(actual.max(), forecast.max()) / 1000
        ax.plot([0, hi], [0, hi], color=ORANGE, linewidth=1)
        ax.set(xlabel="Actual (GWh/hour)", ylabel="Archived forecast (GWh/hour)", title=driver.title() + " · QA only")
        forecast_stats.append({"driver": driver, "scope": "training", **metrics(actual, forecast),
                               "correlation": float(actual.corr(forecast)), "bias_definition": "forecast minus actual"})
    pd.DataFrame(forecast_stats).to_csv(root / "outputs/fundamental_forecast_metrics.csv", index=False)
    save(fig, root, "fundamental_forecast_qa.png")
    return stats


def evaluation_plots(vp, op, winner, importance, root):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
    for ax, table, label in zip(axes, [vp.tail(168), op.tail(168)], ["Validation · final week", "OOS · final week"]):
        dates = pd.to_datetime(table.datetime, utc=True).dt.tz_convert(TZ)
        ax.plot(dates, table.y_true, label="Actual", color=BLUE, linewidth=1.2)
        ax.plot(dates, table[winner], label=winner, color=ORANGE, linewidth=1.1)
        ax.set(title=label, ylabel="EUR/MWh")
        ax.legend(loc="upper left", frameon=False)
    save(fig, root, "heldout_predictions.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    data = importance.sort_values("importance")
    ax.barh(data.feature, data.importance, color=BLUE)
    ax.set(title="CatBoost feature importance · training fit", xlabel="PredictionValuesChange (relative importance)")
    save(fig, root, "feature_importance.png")
