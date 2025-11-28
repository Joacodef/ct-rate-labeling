import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from ctr_labeling import estimate_cost  # noqa: E402  pylint: disable=wrong-import-position
from scripts import generate_labels, evaluate_prompt  # noqa: E402  pylint: disable=wrong-import-position


class ScriptIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp_path = Path(self._tmpdir.name)
        self.runtime_dir = self.tmp_path / "hydra"
        self.runtime_dir.mkdir()
        self.input_csv = self.tmp_path / "reports.csv"
        self.integration_pricing = {
            "gpt-5-nano": {
                "input_per_million": 0.05,
                "output_per_million": 0.4,
                "cached_input_per_million": 0.005,
            }
        }

    def _hydra_runtime(self):
        return SimpleNamespace(
            runtime=SimpleNamespace(output_dir=str(self.runtime_dir)),
            overrides=SimpleNamespace(task=[])
        )

    def _write_reports(self, extra_fields=None, rows=None):
        extra_fields = extra_fields or []
        rows = rows or []
        fieldnames = ["VolumeName", "report_text", *extra_fields]
        with self.input_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _base_cfg(self, prompt_mode="multi", include_ground_truth=False):
        labels = ["Label_A", "Label_B"]
        cfg = {
            "io": {
                "reports_csv": str(self.input_csv),
                "output_csv": "labels.csv",
            },
            "api": {
                "api_key": "integration-key",
                "model": "gpt-5-nano",
                "temperature": 0.0,
                "max_retries": 2,
                "cost_precision": 8,
                "pricing": self.integration_pricing,
            },
            "prompt": {
                "system_prompt": "Integration test prompt.",
                "labels": labels,
                "mode": prompt_mode,
            },
        }
        if include_ground_truth:
            cfg["prompt"]["mode"] = prompt_mode
        return OmegaConf.create(cfg)

    def _patched_hydra(self):
        return patch("scripts.generate_labels.HydraConfig.get", return_value=self._hydra_runtime())

    def _patched_eval_hydra(self):
        return patch("scripts.evaluate_prompt.HydraConfig.get", return_value=self._hydra_runtime())

    def test_generate_labels_integration_writes_full_output(self):
        rows = [
            {"VolumeName": "vol-1", "report_text": "Report one"},
            {"VolumeName": "vol-2", "report_text": "Report two"},
        ]
        self._write_reports(rows=rows)
        cfg = self._base_cfg(prompt_mode="multi")

        meta_first = {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
            "latency_seconds": 0.4,
            "status": "success",
            "error_message": "",
            "request_id": "req-first",
            "model_version": "mock-first",
            "retry_count": 1,
            "started_at_utc": "2025-01-01T00:00:00Z",
            "ended_at_utc": "2025-01-01T00:00:01Z",
        }
        meta_second = {
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "total_tokens": 100,
            "latency_seconds": 0.3,
            "status": "success",
            "error_message": "",
            "request_id": "req-second",
            "model_version": "mock-second",
            "retry_count": 1,
            "started_at_utc": "2025-01-01T00:01:00Z",
            "ended_at_utc": "2025-01-01T00:01:01Z",
        }

        with self._patched_hydra(), patch("scripts.generate_labels.LLMClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.get_labels.side_effect = [
                ({"Label_A": 1, "Label_B": 0}, meta_first),
                ({"Label_A": 0, "Label_B": 1}, meta_second),
            ]

            generate_labels.main.__wrapped__(cfg)

        output_path = self.runtime_dir / "labels.csv"
        self.assertTrue(output_path.exists(), "labels.csv should be produced in the hydra output dir")

        df = pd.read_csv(output_path)
        self.assertEqual(len(df), 2)
        self.assertListEqual(sorted(df.columns.tolist()), sorted([
            "VolumeName",
            "Label_A",
            "Label_B",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "latency_seconds",
            "estimated_cost_usd",
            "model_version",
            "request_id",
            "retry_count",
            "started_at_utc",
            "ended_at_utc",
            "report_hash",
            "status",
            "error_message",
        ]))

        expected_cost_first = round((120 / 1_000_000 * 0.05) + (30 / 1_000_000 * 0.4), 8)
        expected_cost_second = round((80 / 1_000_000 * 0.05) + (20 / 1_000_000 * 0.4), 8)

        self.assertAlmostEqual(df.loc[0, "estimated_cost_usd"], expected_cost_first, places=9)
        self.assertAlmostEqual(df.loc[1, "estimated_cost_usd"], expected_cost_second, places=9)
        self.assertEqual(df.loc[0, "Label_A"], 1)
        self.assertEqual(df.loc[0, "Label_B"], 0)
        self.assertEqual(df.loc[1, "Label_A"], 0)
        self.assertEqual(df.loc[1, "Label_B"], 1)

        for original, produced in zip(rows, df.itertuples(index=False)):
            expected_hash = hashlib.sha256(original["report_text"].encode("utf-8")).hexdigest()
            self.assertEqual(produced.report_hash, expected_hash)

    def test_generate_labels_missing_required_columns_exits(self):
        # Omit report_text column entirely to simulate malformed input CSV
        with self.input_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["VolumeName"])
            writer.writeheader()
            writer.writerow({"VolumeName": "vol-err"})

        cfg = self._base_cfg(prompt_mode="multi")

        with self._patched_hydra(), patch("scripts.generate_labels.LLMClient") as mock_client_cls:
            with self.assertRaises(SystemExit):
                generate_labels.main.__wrapped__(cfg)
            mock_client_cls.assert_not_called()

    def test_evaluate_prompt_integration_creates_metrics_and_discrepancies(self):
        rows = [
            {"VolumeName": "vol-1", "report_text": "first", "Label_A": 1, "Label_B": 0},
            {"VolumeName": "vol-2", "report_text": "second", "Label_A": 0, "Label_B": 1},
        ]
        self._write_reports(extra_fields=["Label_A", "Label_B"], rows=rows)

        cfg = OmegaConf.create({
            "io": {
                "reports_csv": str(self.input_csv),
            },
            "api": {
                "api_key": "integration-key",
                "model": "gpt-5-nano",
                "temperature": 0.0,
                "max_retries": 2,
                "pricing": self.integration_pricing,
            },
            "prompt": {
                "system_prompt": "Integration prompt",
                "labels": ["Label_A", "Label_B"],
                "mode": "multi",
            },
        })

        meta_row_one = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "latency_seconds": 0.5,
            "status": "success",
            "error_message": "",
            "request_id": "eval-1",
            "model_version": "mock-1",
            "retry_count": 1,
            "started_at_utc": "2025-02-01T00:00:00Z",
            "ended_at_utc": "2025-02-01T00:00:01Z",
        }
        meta_row_two = {
            "prompt_tokens": 150,
            "completion_tokens": 30,
            "total_tokens": 180,
            "latency_seconds": 0.75,
            "status": "success",
            "error_message": "",
            "request_id": "eval-2",
            "model_version": "mock-2",
            "retry_count": 1,
            "started_at_utc": "2025-02-01T00:01:00Z",
            "ended_at_utc": "2025-02-01T00:01:01Z",
        }

        with self._patched_eval_hydra(), patch("scripts.evaluate_prompt.LLMClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.get_labels.side_effect = [
                ({"Label_A": 1, "Label_B": 1}, meta_row_one),
                ({"Label_A": 0, "Label_B": 0}, meta_row_two),
            ]

            evaluate_prompt.main.__wrapped__(cfg)

        predictions_csv = self.runtime_dir / "evaluation_predictions.csv"
        metrics_csv = self.runtime_dir / "evaluation_metrics.csv"
        discrepancies_csv = self.runtime_dir / "discrepancies.csv"
        summary_json = self.runtime_dir / "run_summary.json"

        for path in [predictions_csv, metrics_csv, discrepancies_csv, summary_json]:
            self.assertTrue(path.exists(), f"Expected {path.name} to be created")

        predictions = pd.read_csv(predictions_csv)
        self.assertListEqual(sorted(predictions.columns.tolist()), sorted([
            "VolumeName",
            "report_hash",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "latency_seconds",
            "model_version",
            "request_id",
            "retry_count",
            "started_at_utc",
            "ended_at_utc",
            "status",
            "error_message",
            "pred_Label_A",
            "pred_Label_B",
        ]))
        self.assertEqual(predictions.loc[0, "pred_Label_A"], 1)
        self.assertEqual(predictions.loc[0, "pred_Label_B"], 1)
        self.assertEqual(predictions.loc[1, "pred_Label_A"], 0)
        self.assertEqual(predictions.loc[1, "pred_Label_B"], 0)

        metrics = pd.read_csv(metrics_csv)
        self.assertIn("MACRO_AVERAGE", metrics["label"].values)
        label_a_row = metrics.loc[metrics["label"] == "Label_A"].iloc[0]
        label_b_row = metrics.loc[metrics["label"] == "Label_B"].iloc[0]

        self.assertAlmostEqual(label_a_row["precision"], 1.0)
        self.assertAlmostEqual(label_a_row["recall"], 1.0)
        self.assertAlmostEqual(label_a_row["f1"], 1.0)
        self.assertAlmostEqual(label_b_row["precision"], 0.0)
        self.assertAlmostEqual(label_b_row["recall"], 0.0)
        self.assertAlmostEqual(label_b_row["f1"], 0.0)

        discrepancies = pd.read_csv(discrepancies_csv)
        self.assertEqual(len(discrepancies), 2)
        self.assertSetEqual(set(discrepancies["Label"]), {"Label_B"})

        with summary_json.open() as handle:
            summary = json.load(handle)
        expected_total_cost = round(
            estimate_cost("gpt-5-nano", 100, 20, self.integration_pricing)
            + estimate_cost("gpt-5-nano", 150, 30, self.integration_pricing),
            6,
        )
        self.assertAlmostEqual(summary["total_estimated_cost_usd"], expected_total_cost)
        self.assertEqual(summary["total_reports"], 2)
        self.assertEqual(summary["model_version"], "gpt-5-nano")

    def test_evaluate_prompt_missing_ground_truth_columns_exits(self):
        # Missing Label_B column
        rows = [
            {"VolumeName": "vol-1", "report_text": "text", "Label_A": 1},
        ]
        self._write_reports(extra_fields=["Label_A"], rows=rows)

        cfg = OmegaConf.create({
            "io": {"reports_csv": str(self.input_csv)},
            "api": {
                "api_key": "integration-key",
                "model": "gpt-5-nano",
                "temperature": 0.0,
                "max_retries": 2,
                "pricing": self.integration_pricing,
            },
            "prompt": {
                "system_prompt": "Integration prompt",
                "labels": ["Label_A", "Label_B"],
                "mode": "multi",
            },
        })

        with self._patched_eval_hydra(), patch("scripts.evaluate_prompt.LLMClient") as mock_client_cls:
            with self.assertRaises(SystemExit):
                evaluate_prompt.main.__wrapped__(cfg)
            mock_client_cls.assert_not_called()

    def test_generate_labels_resume_with_extra_overrides_exits(self):
        cfg = self._base_cfg()
        overrides = SimpleNamespace(task=["foo=bar"])
        hydra_stub = SimpleNamespace(overrides=overrides)

        with patch("scripts.generate_labels.RESUME_RUN_DIR", str(self.runtime_dir)), \
                patch("scripts.generate_labels.HydraConfig.get", return_value=hydra_stub):
            with self.assertRaises(SystemExit):
                generate_labels.main.__wrapped__(cfg)


if __name__ == "__main__":
    unittest.main()
