import unittest
import pandas as pd
import tempfile
import os
from pathlib import Path
from scripts import evaluate_label_agreement

class TestEvaluateLabelAgreement(unittest.TestCase):
    def setUp(self):
        # Create dummy DataFrames for testing
        self.df1 = pd.DataFrame({
            "VolumeName": ["scan1.nii.gz", "scan2.nii.gz", "scan3.nii.gz"],
            "LabelA": [1, 0, 1],
            "LabelB": [0, 1, 1],
        })
        self.df2 = pd.DataFrame({
            "VolumeName": ["scan1.nii.gz", "scan2.nii.gz", "scan3.nii.gz"],
            "LabelA": [1, 1, 0],
            "LabelB": [0, 1, 0],
        })

    def test_strip_suffixes(self):
        result = evaluate_label_agreement._strip_suffixes("scan1.nii.gz", [".nii.gz", ".nii"])
        self.assertEqual(result, "scan1")
        result = evaluate_label_agreement._strip_suffixes("scan2.nii", [".nii.gz", ".nii"])
        self.assertEqual(result, "scan2")
        result = evaluate_label_agreement._strip_suffixes("scan3", [".nii.gz", ".nii"])
        self.assertEqual(result, "scan3")

    def test_strip_reconstruction_suffix(self):
        self.assertEqual(evaluate_label_agreement._strip_reconstruction_suffix("scan1_2"), "scan1")
        self.assertEqual(evaluate_label_agreement._strip_reconstruction_suffix("scan2"), "scan2")

    def test_normalize_columns(self):
        df = pd.DataFrame({"A": [1], "b": [2]})
        norm = evaluate_label_agreement._normalize_columns(df)
        self.assertEqual(norm, {"a": "A", "b": "b"})

    def test_get_case_insensitive_column(self):
        df = pd.DataFrame({"A": [1], "b": [2]})
        col = evaluate_label_agreement._get_case_insensitive_column(df, "a")
        self.assertEqual(col, "A")
        col = evaluate_label_agreement._get_case_insensitive_column(df, "B")
        self.assertEqual(col, "b")
        with self.assertRaises(KeyError):
            evaluate_label_agreement._get_case_insensitive_column(df, "c")

    def test_confusion_counts(self):
        a = pd.Series([1, 0, 1])
        b = pd.Series([1, 1, 0])
        counts = evaluate_label_agreement._confusion_counts(a, b)
        self.assertEqual(counts, {"tp": 1, "fp": 1, "fn": 1, "tn": 0})

    def test_calculate_agreement_metrics(self):
        metrics = evaluate_label_agreement._calculate_agreement_metrics(tp=1, fp=1, fn=1, tn=0)
        self.assertIn("agreement", metrics)
        self.assertIn("cohen_kappa", metrics)
        self.assertIn("mcc", metrics)
        self.assertIn("n", metrics)
        self.assertEqual(metrics["n"], 3)

    def test_load_and_normalize(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            self.df1.to_csv(path, index=False)
            df = evaluate_label_agreement._load_and_normalize(
                csv_path=path,
                id_col_name="VolumeName",
                suffixes=[".nii.gz", ".nii"],
                keep_reconstruction_suffix=False,
            )
            self.assertIn("VolumeName", df.columns)
            self.assertEqual(len(df), 3)
            self.assertEqual(df["VolumeName"].iloc[0], "scan1")

    def test_resolve_shared_labels(self):
        dfs = [self.df1, self.df2]
        labels = evaluate_label_agreement._resolve_shared_labels(dfs, labels_arg="", exclude_cols_arg="VolumeName")
        self.assertIn("LabelA", labels)
        self.assertIn("LabelB", labels)
        labels_explicit = evaluate_label_agreement._resolve_shared_labels(dfs, labels_arg="LabelA", exclude_cols_arg="VolumeName")
        self.assertEqual(labels_explicit, ["LabelA"])

if __name__ == "__main__":
    unittest.main()
