"""Regression checks from the second audit. All API/quote fixtures are synthetic
and confined to temporary directories; they are never submission evidence.
"""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import requests

from src.data import load_data, TZ
from src.features import build_features
from src.llm import call_logged, choose_diagnostics, write_review
from src.reporting import render_report
from src.trading import load_curve, curve_revisions, CURVE_COLUMNS
from src.validation import splits, regime_metrics
from src.diagnostics import run_selected, DEFAULT

ROOT = Path(__file__).resolve().parents[1]


class RecheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "config.json").read_text())
        cls.df = load_data(ROOT / "data/de_power_2y_clean.csv")
        cls.x, _, _ = build_features(cls.df, cls.config)

    def test_corrupt_or_invalid_cache_falls_back_without_network(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True), patch("src.llm.requests.post") as post:
            root = Path(tmp)
            _, status = call_logged(root, "review", "prompt", {}, "replay")
            cache = root / "ai_logs/live_cache" / f"review_{status['request_sha256']}.json"
            cache.parent.mkdir()
            for bad in ["{broken", "null", "[]", json.dumps({"origin": "live_api", "request_sha256": status["request_sha256"],
                         "response": {"status": "incomplete", "output": []}})]:
                with self.subTest(cache=bad):
                    cache.write_text(bad)
                    response, result = call_logged(root, "review", "prompt", {}, "replay")
                    self.assertIsNone(response)
                    self.assertEqual(result["status"], "invalid_cache")
            post.assert_not_called()

    def test_failed_or_incomplete_api_response_is_logged_in_full(self):
        fixtures = [
            (True, 200, {"id": "synthetic-only", "status": "incomplete", "output": [], "incomplete_details": {"reason": "max_output_tokens"}}),
            (True, 200, {"id": "synthetic-only", "status": "completed", "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "fixture"}]}]}),
            (True, 200, {"status": "completed", "output": [None]}),
            (False, 429, {"error": {"message": "Synthetic quota failure"}}),
        ]
        for ok, code, raw in fixtures:
            with self.subTest(code=code, raw=raw), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-only"}), patch("src.llm.requests.post") as post:
                post.return_value.ok, post.return_value.status_code = ok, code
                post.return_value.json.return_value = raw
                root = Path(tmp)
                response, status = call_logged(root, "review", "prompt", {}, "live")
                self.assertIsNone(response)
                self.assertEqual(status["status"], "failed")
                self.assertEqual(json.loads((root / "ai_logs/review_response.json").read_text()), raw)
                self.assertFalse((root / "ai_logs/live_cache").exists())

    def test_non_json_api_response_and_timeout_do_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-only"}), patch("src.llm.requests.post") as post:
            root = Path(tmp)
            post.return_value.status_code = 502
            post.return_value.text = "synthetic gateway failure"
            post.return_value.json.side_effect = ValueError("not JSON")
            _, status = call_logged(root, "review", "prompt", {}, "live")
            self.assertEqual(status["status"], "failed")
            self.assertEqual(json.loads((root / "ai_logs/review_response.json").read_text())["http_status"], 502)
            post.side_effect = requests.Timeout("synthetic timeout")
            _, status = call_logged(root, "review", "prompt", {}, "live")
            self.assertEqual(status["status"], "failed")
            self.assertFalse((root / "ai_logs/live_cache").exists())

    def test_invalid_llm_selection_and_valid_review_remain_incomplete(self):
        # Both network responses are mocked; the completion flag must still reject
        # the unsupported diagnostic selection, even if the review succeeded.
        def fixture(text):
            return {"id": "synthetic-only", "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-only"}), patch("src.llm.requests.post") as post:
            root = Path(tmp)
            (root / "outputs").mkdir()
            post.return_value.ok = True
            post.return_value.json.side_effect = [fixture('{"diagnostics_to_run":[{"name":"exec","reason":"fixture"}]}'), fixture("Synthetic test review.")]
            names, selection = choose_diagnostics(root, {}, "live")
            status = write_review(root, {}, "live", selection)
            self.assertEqual(names, DEFAULT)
            self.assertEqual(status["selection"], "invalid_response")
            self.assertEqual(status["review"], "live")
            self.assertFalse(status["complete"])

    def test_two_stage_protocol_replays_and_missing_key_preserves_cache(self):
        # Synthetic protocol test only; temporary logs are deleted afterwards.
        evidence = {"selected_model": "catboost", "model_metrics": [{"model": "catboost", "mae": 1.0, "rmse": 2.0}]}
        selection_text = json.dumps({"diagnostics_to_run": [{"name": "negative_price_analysis", "reason": "Synthetic test choice"}]})
        def fixture(text):
            return {"id": "synthetic-only", "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-only"}, clear=True), patch("src.llm.requests.post") as post:
            root = Path(tmp)
            (root / "outputs").mkdir()
            post.return_value.ok = True
            post.return_value.json.side_effect = [fixture(selection_text), fixture("Synthetic test review.")]
            names, selection = choose_diagnostics(root, evidence, "live")
            self.assertTrue(write_review(root, evidence, "live", selection)["complete"])
            self.assertEqual(names, ["negative_price_analysis"])
            cache_before = {p.name: p.read_bytes() for p in (root / "ai_logs/live_cache").glob("*.json")}
            with patch.dict(os.environ, {}, clear=True):
                _, selection = choose_diagnostics(root, evidence, "live")
                self.assertFalse(write_review(root, evidence, "live", selection)["complete"])
                _, selection = choose_diagnostics(root, evidence, "replay")
                self.assertTrue(write_review(root, evidence, "replay", selection)["complete"])
            self.assertEqual(post.call_count, 2)
            self.assertEqual(cache_before, {p.name: p.read_bytes() for p in (root / "ai_logs/live_cache").glob("*.json")})

    def test_report_reflects_partial_llm_success_curve_and_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "outputs", root / "outputs")
            (root / "ai_logs").mkdir()
            (root / "ai_logs/status.json").write_text(json.dumps({"complete": False, "selection": "failed", "review": "live"}))
            selection = json.loads((root / "outputs/selection.json").read_text())
            selection["seed"] = 123
            (root / "outputs/selection.json").write_text(json.dumps(selection))
            trading = json.loads((root / "outputs/trading_view.json").read_text())
            trading["matched_quote_count"] = 1
            (root / "outputs/trading_view.json").write_text(json.dumps(trading))
            render_report(root)
            report = (root / "report.md").read_text()
            self.assertIn("Diagnostic selection: failed; review: live", report)
            self.assertIn("current review comes from a genuine API response", report)
            self.assertNotIn("no genuine API calls have completed", report)
            self.assertNotIn("No verified historical prompt quote was acquired", report)
            self.assertIn("seed 123", report)

    def test_split_boundaries_reject_partial_days_and_warmup_gaps(self):
        for changes in [{"oos_start": "2026-08-30 12:00"}, {"validation_start": "2024-09-28"}, {"oos_start": "2026-02-01"}]:
            cfg = {**self.config, **changes}
            with self.subTest(config=changes), self.assertRaises(ValueError):
                splits(self.x, cfg)
        incomplete = self.x.copy()
        first_validation = pd.Timestamp(self.config["validation_start"], tz=TZ).tz_convert("UTC")
        incomplete.loc[first_validation, "price_prev_day_same_hour"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Incomplete validation"):
            splits(incomplete, self.config)

    def test_high_price_threshold_is_reflected_in_diagnostic_labels(self):
        table = pd.DataFrame({"datetime": pd.date_range("2026-04-01", periods=3, freq="h", tz="UTC"),
                              "y_true": [50.0, 150.0, 250.0], "chosen": [50.0, 140.0, 230.0]})
        cfg = {**self.config, "high_price_threshold": 100.0}
        regimes = regime_metrics(table, 100.0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "outputs").mkdir()
            results = run_selected(["high_price_analysis"], None, None, None, cfg, None, None, regimes, None, root)
            row = results["high_price_analysis"][0]
            self.assertEqual(row["regime"], "high_price_gt_100")
            self.assertEqual(row["n"], 2)
            self.assertEqual(row["mae"], 15.0)

    def test_curve_month_uses_dst_hour_count_and_rejects_stale_quotes(self):
        day = pd.Timestamp("2025-03-01", tz=TZ)
        times = pd.date_range(day, day + pd.DateOffset(days=1), freq="h", inclusive="left")
        table = pd.DataFrame({"datetime": times, "y_true": 0.0, "chosen": 120.0, "weekly_persistence": 100.0})
        history = self.df.copy()
        history["price_da_eur_mwh"] = 100.0
        quote = {"observed_at": "2025-02-28T10:00:00+01:00", "delivery_start": day.isoformat(),
                 "delivery_end": "2025-04-01T00:00:00+02:00", "product": "DE_BASE_MONTH",
                 "price_eur_mwh": 150.0, "source_url": "https://example.com/synthetic-test-only"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curve.csv"
            pd.DataFrame([quote], columns=CURVE_COLUMNS).to_csv(path, index=False)
            curve = load_curve(path)
            rows = curve_revisions(history, table, "chosen", curve, self.config)
            self.assertEqual(rows[0]["contract_hours"], 743)
            self.assertAlmostEqual(rows[0]["conditional_curve_revision"], 24 / 743 * -30)
            curve["observed_at"] = pd.Timestamp("2025-02-01T00:00:00Z")
            self.assertEqual(curve_revisions(history, table, "chosen", curve, self.config), [])


if __name__ == "__main__":
    unittest.main()
