import csv
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from omegaconf import OmegaConf

# Ensure the repository root (so that `scripts` is importable) is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts import generate_labels  # noqa: E402  pylint: disable=wrong-import-position


class GenerateLabelsPipelineTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp_path = Path(self._tmpdir.name)
        self.runtime_dir = self.tmp_path / "hydra"
        self.runtime_dir.mkdir()
        self.input_csv = self.tmp_path / "reports.csv"
        self.output_name = "labels.csv"
        self.pricing_table = {
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

    def _write_reports(self, rows):
        with self.input_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["VolumeName", "report_text"])
            writer.writeheader()
            writer.writerows(rows)

    def _build_cfg(self, mode="multi"):
        return OmegaConf.create({
            "io": {
                "reports_csv": str(self.input_csv),
                "output_csv": self.output_name,
            },
            "api": {
                "api_key": "test-key",
                "model": "gpt-5-nano",
                "temperature": 0.0,
                "max_retries": 1,
                "cost_precision": 8,
                "pricing": self.pricing_table,
            },
            "prompt": {
                "system_prompt": "You are a test model.",
                "labels": ["Label_A", "Label_B"],
                "mode": mode,
            },
        })

    def _read_output(self):
        output_path = self.runtime_dir / self.output_name
        self.assertTrue(output_path.exists(), "Expected labels.csv to be created")
        return pd.read_csv(output_path)

    def _run_main(self, cfg):
        entrypoint = getattr(generate_labels.main, "__wrapped__", generate_labels.main)
        return entrypoint(cfg)

    def test_multi_mode_generates_expected_rows(self):
        rows = [
            {"VolumeName": "vol-1", "report_text": "Report alpha"},
            {"VolumeName": "vol-2", "report_text": "Report beta"},
        ]
        self._write_reports(rows)
        cfg = self._build_cfg(mode="multi")

        meta_one = {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "latency_seconds": 0.25,
            "status": "success",
            "error_message": "",
            "request_id": "req-1",
            "model_version": "mock-1",
            "retry_count": 1,
            "started_at_utc": "2025-01-01T00:00:00Z",
            "ended_at_utc": "2025-01-01T00:00:01Z",
        }
        meta_two = {
            "prompt_tokens": 200,
            "completion_tokens": 40,
            "total_tokens": 240,
            "latency_seconds": 0.5,
            "status": "success",
            "error_message": "",
            "request_id": "req-2",
            "model_version": "mock-2",
            "retry_count": 1,
            "started_at_utc": "2025-01-01T00:01:00Z",
            "ended_at_utc": "2025-01-01T00:01:01Z",
        }

        with patch("scripts.generate_labels.HydraConfig.get", return_value=self._hydra_runtime()), \
                patch("scripts.generate_labels.LLMClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.get_labels.side_effect = [
                ({"Label_A": 1, "Label_B": 0}, meta_one),
                ({"Label_A": 0, "Label_B": 1}, meta_two),
            ]

            self._run_main(cfg)

        df = self._read_output()
        self.assertEqual(len(df), 2)
        self.assertEqual(mock_client.get_labels.call_count, 2)

        # Row order should mirror the input order
        row_one = df.iloc[0]
        row_two = df.iloc[1]

        self.assertEqual(row_one["VolumeName"], "vol-1")
        self.assertEqual(row_one["Label_A"], 1)
        self.assertEqual(row_one["Label_B"], 0)
        self.assertAlmostEqual(row_one["estimated_cost_usd"], 0.000009, places=9)
        self.assertEqual(row_one["status"], "success")

        self.assertEqual(row_two["VolumeName"], "vol-2")
        self.assertEqual(row_two["Label_A"], 0)
        self.assertEqual(row_two["Label_B"], 1)
        expected_cost_two = round((200 / 1_000_000 * 0.05) + (40 / 1_000_000 * 0.4), 8)
        self.assertAlmostEqual(row_two["estimated_cost_usd"], expected_cost_two, places=9)

        # Report hashes should be deterministic per row content
        for original, produced in zip(rows, df.itertuples()):
            expected_hash = hashlib.sha256(original["report_text"].encode("utf-8")).hexdigest()
            self.assertEqual(produced.report_hash, expected_hash)

    def test_single_mode_combines_per_label_metadata(self):
        rows = [{"VolumeName": "vol-single", "report_text": "Report gamma"}]
        self._write_reports(rows)
        cfg = self._build_cfg(mode="single")

        meta_label_a = {
            "prompt_tokens": 50,
            "completion_tokens": 10,
            "total_tokens": 60,
            "latency_seconds": 0.25,
            "status": "success",
            "error_message": "",
            "request_id": "req-a",
            "model_version": "mock-a",
            "retry_count": 1,
            "started_at_utc": "2025-01-01T00:00:00Z",
            "ended_at_utc": "2025-01-01T00:00:01Z",
        }
        meta_label_b = {
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "total_tokens": 25,
            "latency_seconds": 0.5,
            "status": "success",
            "error_message": "",
            "request_id": "req-b",
            "model_version": "mock-b",
            "retry_count": 2,
            "started_at_utc": "2025-01-01T00:00:02Z",
            "ended_at_utc": "2025-01-01T00:00:03Z",
        }

        with patch("scripts.generate_labels.HydraConfig.get", return_value=self._hydra_runtime()), \
                patch("scripts.generate_labels.LLMClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.get_labels.side_effect = [
                ({"Label_A": 1}, meta_label_a),
                ({"Label_B": 0}, meta_label_b),
            ]

            self._run_main(cfg)

        df = self._read_output()
        self.assertEqual(len(df), 1)
        row = df.iloc[0]

        self.assertEqual(row["Label_A"], 1)
        self.assertEqual(row["Label_B"], 0)
        self.assertEqual(row["prompt_tokens"], 70)
        self.assertEqual(row["completion_tokens"], 15)
        self.assertEqual(row["total_tokens"], 85)
        self.assertAlmostEqual(row["latency_seconds"], 0.75, places=6)
        self.assertEqual(row["retry_count"], 3)
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["request_id"], "req-a | req-b")
        self.assertEqual(row["model_version"], "mock-a | mock-b")
        self.assertEqual(row["started_at_utc"], "2025-01-01T00:00:00Z")
        self.assertEqual(row["ended_at_utc"], "2025-01-01T00:00:03Z")

        expected_cost = round((70 / 1_000_000 * 0.05) + (15 / 1_000_000 * 0.4), 8)
        self.assertAlmostEqual(row["estimated_cost_usd"], expected_cost, places=9)

        # Ensure we invoked the client once per target label with the correct override
        call_args = mock_client.get_labels.call_args_list
        self.assertEqual(len(call_args), 2)
        self.assertEqual(call_args[0].kwargs["labels_override"], ["Label_A"])
        self.assertEqual(call_args[1].kwargs["labels_override"], ["Label_B"])

    def test_single_mode_empty_report_skips_per_label_calls(self):
        rows = [{"VolumeName": "vol-empty", "report_text": "   "}]
        self._write_reports(rows)
        cfg = self._build_cfg(mode="single")

        meta = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
            "status": "skipped",
            "error_message": "",
            "request_id": "",
            "model_version": "mock",
            "retry_count": 0,
            "started_at_utc": "",
            "ended_at_utc": "",
        }

        with patch("scripts.generate_labels.HydraConfig.get", return_value=self._hydra_runtime()), \
                patch("scripts.generate_labels.LLMClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.get_labels.return_value = ({"Label_A": 0, "Label_B": 0}, meta)

            generate_labels.main.__wrapped__(cfg)

        mock_client.get_labels.assert_called_once()
        kwargs = mock_client.get_labels.call_args.kwargs
        self.assertNotIn("labels_override", kwargs)


if __name__ == "__main__":
    unittest.main()
