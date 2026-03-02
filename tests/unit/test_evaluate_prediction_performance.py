import unittest
import pandas as pd
import tempfile
import os
from pathlib import Path
from scripts import evaluate_prediction_performance

class TestEvaluatePredictionPerformance(unittest.TestCase):
    def setUp(self):
        self.df_left = pd.DataFrame({
            "VolumeName": ["scan1.nii.gz", "scan2.nii.gz", "scan3.nii.gz"],
            "LabelA": [1, 0, 1],
            "LabelB": [0, 1, 1],
        })
        self.df_right = pd.DataFrame({
            "VolumeName": ["scan1.nii.gz", "scan2.nii.gz", "scan3.nii.gz"],
            "LabelA": [1, 1, 0],
            "LabelB": [0, 1, 0],
        })

    def test_strip_suffixes(self):
        result = evaluate_prediction_performance._strip_suffixes("scan1.nii.gz", [".nii.gz", ".nii"])
        self.assertEqual(result, "scan1")
        result = evaluate_prediction_performance._strip_suffixes("scan2.nii", [".nii.gz", ".nii"])
        self.assertEqual(result, "scan2")
        result = evaluate_prediction_performance._strip_suffixes("scan3", [".nii.gz", ".nii"])
        self.assertEqual(result, "scan3")

    def test_strip_reconstruction_suffix(self):
        self.assertEqual(evaluate_prediction_performance._strip_reconstruction_suffix("scan1_2"), "scan1")
        self.assertEqual(evaluate_prediction_performance._strip_reconstruction_suffix("scan2"), "scan2")

    def test_normalize_columns(self):
        df = pd.DataFrame({"A": [1], "b": [2]})
        norm = evaluate_prediction_performance._normalize_columns(df)
        self.assertEqual(norm, {"a": "A", "b": "b"})

    def test_get_case_insensitive_column(self):
        df = pd.DataFrame({"A": [1], "b": [2]})
        col = evaluate_prediction_performance._get_case_insensitive_column(df, "a")
        self.assertEqual(col, "A")
        col = evaluate_prediction_performance._get_case_insensitive_column(df, "B")
        self.assertEqual(col, "b")
        with self.assertRaises(KeyError):
            evaluate_prediction_performance._get_case_insensitive_column(df, "c")

    def test_shared_label_columns(self):
        exclude = ["VolumeName"]
        shared = evaluate_prediction_performance._shared_label_columns(self.df_left, self.df_right, exclude)
        self.assertIn("LabelA", shared)
        self.assertIn("LabelB", shared)
        self.assertNotIn("VolumeName", shared)

    def test_calculate_binary_metrics(self):
        df = pd.DataFrame({"true": [1, 0, 1], "pred": [1, 1, 0]})
        metrics = evaluate_prediction_performance.calculate_binary_metrics(df, "true", "pred")
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("accuracy", metrics)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 0)

    def test_build_discrepancies(self):
        df = pd.DataFrame({
            "VolumeName": ["scan1", "scan2", "scan3"],
            "true_LabelA": [1, 0, 1],
            "pred_LabelA": [1, 1, 0],
        })
        labels = [("LabelA", "true_LabelA", "pred_LabelA")]
        rows = evaluate_prediction_performance._build_discrepancies(df, "VolumeName", labels)
        self.assertTrue(any(row["Predicted"] != row["Actual"] for row in rows))
        self.assertTrue(all("VolumeName" in row for row in rows))

if __name__ == "__main__":
    unittest.main()
