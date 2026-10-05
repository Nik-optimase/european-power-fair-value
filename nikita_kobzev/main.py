"""Single entry point. Default execution is offline and numerically reproducible."""
import argparse
import json
import os
from pathlib import Path
import random
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from src.data import checksum, load_data, write_json
from src.qa import run_qa
from src.features import build_features, FEATURES
from src.validation import compare_models, final_oos, regime_metrics, splits
from src.diagnostics import ALLOWED, feature_drift, worst_errors, run_selected
from src.llm import choose_diagnostics, write_review
from src.plots import eda, evaluation_plots
from src.trading import build_trading
from src.reporting import render_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--llm", choices=["replay", "live", "off"], default="replay")
    parser.add_argument("--llm-only", action="store_true", help="Run diagnostics/review on existing validation artifacts")
    args = parser.parse_args()
    start = STARTED
    config = json.loads(args.config.read_text())
    if config["timezone"] != "Europe/Berlin" or config["forecast_hour"] != 11:
        raise ValueError("This implementation is specified for 11:00 Europe/Berlin")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    for name in ["outputs", "figures", "ai_logs"]:
        (ROOT / name).mkdir(exist_ok=True)
    print("1/6 Read preserved dataset and check integrity", flush=True)
    df = load_data(ROOT / "data/de_power_2y_clean.csv")
    qa = run_qa(df, ROOT)
    x, origins, audit = build_features(df, config)
    audit.to_csv(ROOT / "outputs/feature_timing_audit.csv", index=False)
    if args.llm_only:
        manifest = json.loads((ROOT / "outputs/run_manifest.json").read_text())
        if manifest["config"] != config or manifest["dataset_sha256"] != qa["sha256"]:
            raise ValueError("Saved validation outputs are stale; rerun the full pipeline first")
        masks = splits(x, config)
        selection = json.loads((ROOT / "outputs/selection.json").read_text())
        vp = pd.read_csv(ROOT / "outputs/validation_predictions.csv", float_precision="round_trip")
        importance = pd.read_csv(ROOT / "outputs/feature_importance.csv", float_precision="round_trip")
    else:
        print("2/6 Chronological model comparison (final OOS untouched)", flush=True)
        masks, selection, vp, importance = compare_models(df, x, config, ROOT)
        eda(df, masks["train"], ROOT)
    print("3/6 Validation diagnostics and logged LLM workflow", flush=True)
    regimes = regime_metrics(vp, config["high_price_threshold"])
    regimes.to_csv(ROOT / "outputs/regime_metrics.csv", index=False)
    drift = feature_drift(x, masks)
    drift.to_csv(ROOT / "outputs/feature_drift.csv", index=False)
    # Convert pandas NaN for empty regimes to JSON null. No OOS evidence here.
    evidence = {"scope": "validation only; no final OOS evidence",
        "selected_model": selection["winner"], "model_metrics": selection["validation_metrics"],
        "regime_metrics": json.loads(regimes.to_json(orient="records")),
        "feature_importance": importance.to_dict("records"), "qa_warnings": qa["warnings"],
        "largest_errors": worst_errors(vp, selection["winner"]),
        "feature_drift": drift.to_dict("records"), "allowed_diagnostics": ALLOWED}
    names, selection_status = choose_diagnostics(ROOT, evidence, args.llm)
    diagnostic_results = run_selected(names, df, x, masks, config, selection, vp, regimes, drift, ROOT)
    review_evidence = {**evidence, "executed_diagnostics": diagnostic_results}
    llm_status = write_review(ROOT, review_evidence, args.llm, selection_status)
    if not args.llm_only:
        print("4/6 Refit selected specification and score historical OOS", flush=True)
        op, oos_scores = final_oos(df, x, masks, config, selection, ROOT)
        print("5/6 Daily fair value, optional curve matching and figures", flush=True)
        build_trading(df, vp, op, selection, config, ROOT)
        evaluation_plots(vp, op, selection["winner"], importance, ROOT)
    print("6/6 Generate report and completion status", flush=True)
    render_report(ROOT)
    elapsed = time.perf_counter() - start
    if not args.llm_only:
        write_json(ROOT / "outputs/run_manifest.json", {"config": config, "dataset_sha256": qa["sha256"],
            "runtime_seconds": round(elapsed, 3), "winner": selection["winner"],
            "mode": "historical sequential next-day backtest; fixed final model",
            "predictions_sha256": checksum(ROOT / "predictions.csv"), "llm": llm_status})
    else:
        manifest["llm"] = llm_status
        manifest["last_llm_only_runtime_seconds"] = round(elapsed, 3)
        write_json(ROOT / "outputs/run_manifest.json", manifest)
    completion = {"numerical_pipeline_complete": True, "real_llm_calls_complete": llm_status["complete"],
                  "llm_integration_implemented": True,
                  "llm_execution": {"selection": llm_status["selection"], "review": llm_status["review"]},
                  "limitations": ["Employer OOS dates unspecified; configurable historical window used",
                                  "No verified historical curve quote; conditional interface provided",
                                  "No archived publication vintages; documented delay/revision assumptions apply"]}
    write_json(ROOT / "outputs/completion_status.json", completion)
    print(f"Done in {elapsed:.1f}s. Selected: {selection['winner']}. Genuine LLM calls complete: {llm_status['complete']}", flush=True)


if __name__ == "__main__":
    main()
