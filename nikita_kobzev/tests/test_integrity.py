"""Focused tests for failure modes that can invalidate a forecasting study.

Any synthetic API responses/curve prices below are isolated test fixtures and
are never written into the project's ai_logs or market-data deliverables.
"""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data import load_data, TARGET, TZ
from src.features import build_features, FEATURES, FORBIDDEN, origin_times
from src.validation import splits, compare_models
from src.llm import validate_selection, call_logged, extract_text
from src.trading import daily_values, load_curve, curve_revisions, CURVE_COLUMNS
from data.fetch_smard import parse_series

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "config.json").read_text())
        cls.df = load_data(ROOT / "data/de_power_2y_clean.csv")
        cls.x, cls.origins, cls.audit = build_features(cls.df, cls.config)

    def test_features_reject_forbidden_and_have_time_margin(self):
        self.assertFalse(set(FEATURES) & FORBIDDEN)
        self.assertEqual(int(self.audit.timing_violations.sum()), 0)
        self.assertGreater(self.audit.minimum_margin_hours.min(), 0)

    def test_future_data_perturbation_does_not_change_origin_features(self):
        for day in ["2025-03-30", "2025-10-26", "2026-08-30"]:
            start = pd.Timestamp(day, tz=TZ)
            target = pd.date_range(start, start + pd.DateOffset(days=1), freq="h", inclusive="left").tz_convert("UTC")
            before, origin, _ = build_features(self.df, self.config, target)
            changed = self.df.copy()
            # Future prices for delivery D have not cleared at the D-1 11:00 origin.
            changed.loc[changed.index >= start.tz_convert("UTC"), TARGET] = 999999.0
            unavailable = changed.index + pd.Timedelta(hours=4) > origin[0]
            for col in ["load_actual_mwh", "wind_actual_mwh", "solar_actual_mwh"]:
                changed.loc[unavailable, col] = 999999.0
            # None of these same-hour forecast/error columns may enter the model.
            for col in ["load_forecast_mwh", "wind_forecast_mwh", "solar_forecast_mwh", "load_error_mwh"]:
                changed[col] = 999999.0
            after, _, _ = build_features(changed, self.config, target)
            pd.testing.assert_frame_equal(before, after)

    def test_origins_and_daily_means_across_dst(self):
        for day, n in [("2025-03-30", 23), ("2025-10-26", 25)]:
            start = pd.Timestamp(day, tz=TZ)
            t = pd.date_range(start, start + pd.DateOffset(days=1), freq="h", inclusive="left")
            origins = origin_times(t).tz_convert(TZ)
            self.assertEqual(len(t), n)
            self.assertEqual(len(origins.unique()), 1)
            self.assertEqual(origins[0].hour, 11)
            self.assertEqual(str(origins[0].date()), str((start - pd.DateOffset(days=1)).date()))
            table = pd.DataFrame({"datetime": t, "y_true": 0.0, "winner": np.arange(n), "weekly_persistence": 0.0})
            result = daily_values(table, "winner").iloc[0]
            self.assertEqual(result.hours, n)
            self.assertEqual(result.fair_value, np.arange(n).mean())

    def test_oos_labels_cannot_change_selection(self):
        # Tiny real fits test data routing, not model performance.
        cfg = copy.deepcopy(self.config)
        cfg["catboost_candidates"] = [{"iterations": 2, "depth": 2, "learning_rate": 0.1, "l2_leaf_reg": 5}]
        altered = self.df.copy()
        masks = splits(self.x, cfg)
        altered.loc[masks["oos"], TARGET] = 999999.0
        changed_x, _, _ = build_features(altered, cfg)
        pd.testing.assert_frame_equal(self.x.loc[masks["validation"]], changed_x.loc[masks["validation"]])
        selections = []
        for frame, x in [(self.df, self.x), (altered, changed_x)]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "outputs").mkdir()
                _, selected, _, _ = compare_models(frame, x, cfg, root)
                selections.append(selected)
        self.assertEqual(selections[0], selections[1])

    def test_smard_parser_preserves_dst_and_rejects_missing_rows(self):
        for day in ["2025-03-30", "2025-10-26"]:
            start = pd.Timestamp(day, tz=TZ)
            t = pd.date_range(start, start + pd.DateOffset(days=1), inclusive="left", freq="h").tz_convert("UTC")
            lines = ["Datum von;Datum bis;Value"] + [f"{v.strftime('%d.%m.%Y %H:%M')};unused;1.234,50" for v in t.tz_convert(TZ)]
            result = parse_series("\n".join(lines), t, "sample")
            self.assertTrue(result.index.is_unique)
            self.assertTrue((result == 1234.5).all())
            with self.assertRaises(ValueError):
                parse_series("\n".join(lines[:-1]), t, "sample")

    def test_llm_whitelist_rejects_code_duplicate_and_empty(self):
        good = {"diagnostics_to_run": [{"name": "negative_price_analysis", "reason": "Review errors"}]}
        self.assertEqual(validate_selection(good), ["negative_price_analysis"])
        for obj in [{"diagnostics_to_run": []}, {"diagnostics_to_run": good["diagnostics_to_run"] * 2},
                    {"diagnostics_to_run": [{"name": "exec", "reason": "run code"}]}]:
            with self.assertRaises(ValueError):
                validate_selection(obj)

    def test_missing_key_never_sends_or_fakes_response(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True), patch("src.llm.requests.post") as post:
            response, status = call_logged(Path(tmp), "test", "prompt", {"evidence": 1}, "live")
            self.assertIsNone(response)
            self.assertEqual(status["status"], "skipped")
            post.assert_not_called()
            self.assertIsNone(json.loads((Path(tmp) / "ai_logs/test_response.json").read_text())["response"])

    def test_real_call_protocol_and_exact_request_replay(self):
        # Explicitly synthetic HTTP fixture, isolated outside delivered ai_logs.
        fixture = {"id": "synthetic-test-only", "status": "completed", "output": [{"type": "message",
                   "content": [{"type": "output_text", "text": "fixture"}]}]}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-key"}), patch("src.llm.requests.post") as post:
            post.return_value.ok = True
            post.return_value.json.return_value = fixture
            response, status = call_logged(Path(tmp), "test", "prompt", {"evidence": 1}, "live")
            self.assertEqual(status["status"], "live")
            self.assertEqual(extract_text(response), "fixture")
            replay, status = call_logged(Path(tmp), "test", "prompt", {"evidence": 1}, "replay")
            self.assertEqual(status["status"], "replayed")
            miss, status = call_logged(Path(tmp), "test", "prompt", {"evidence": 2}, "replay")
            self.assertIsNone(miss)
            self.assertEqual(post.call_count, 1)
            self.assertEqual(post.call_args.args[0], "https://api.openai.com/v1/responses")

    def test_curve_timing_and_delivery_matching(self):
        day = pd.Timestamp("2026-08-31", tz=TZ)  # Monday, before the contract starts.
        t = pd.date_range(day, day + pd.DateOffset(days=1), freq="h", inclusive="left")
        op = pd.DataFrame({"datetime": t, "y_true": 0.0, "chosen": 120.0, "weekly_persistence": 100.0})
        history = self.df.copy()
        history[TARGET] = 100.0
        quote = {"observed_at": "2026-08-28T18:00:00+02:00", "delivery_start": day.isoformat(),
                 "delivery_end": (day + pd.DateOffset(days=7)).isoformat(), "product": "DE_BASE_WEEK",
                 "price_eur_mwh": 150.0, "source_url": "https://example.com/synthetic-test-only"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curve.csv"
            pd.DataFrame([quote], columns=CURVE_COLUMNS).to_csv(path, index=False)
            curve = load_curve(path)
            rows = curve_revisions(history, op, "chosen", curve, self.config)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["conditional_curve_revision"], -30 / 7)
            curve["observed_at"] = pd.Timestamp("2026-08-30T12:00:00+02:00")
            self.assertEqual(curve_revisions(history, op, "chosen", curve, self.config), [])
            quote["observed_at"] = "2026-08-28T18:00:00"  # No offset.
            pd.DataFrame([quote], columns=CURVE_COLUMNS).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_curve(path)

    def test_submitted_predictions_contract(self):
        p = pd.read_csv(ROOT / "predictions.csv")
        self.assertEqual(list(p), ["datetime", "y_pred"])
        t = pd.to_datetime(p.datetime, utc=True)
        self.assertTrue(t.is_unique and t.is_monotonic_increasing)
        self.assertTrue(np.isfinite(p.y_pred).all())
        self.assertEqual(len(p), int(splits(self.x, self.config)["oos"].sum()))


if __name__ == "__main__":
    unittest.main()
