import unittest
import pandas as pd
import tempfile
from pathlib import Path
from scripts import generate_consensus_ensemble

class TestGenerateConsensusEnsemble(unittest.TestCase):
    def setUp(self):
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
        self.set_names = ["run1", "run2"]
        self.labels = ["LabelA", "LabelB"]

    def test_strip_suffixes(self):
        result = generate_consensus_ensemble._strip_suffixes("scan1.nii.gz", [".nii.gz", ".nii"])
        self.assertEqual(result, "scan1")
        result = generate_consensus_ensemble._strip_suffixes("scan2.nii", [".nii.gz", ".nii"])
        self.assertEqual(result, "scan2")
        result = generate_consensus_ensemble._strip_suffixes("scan3", [".nii.gz", ".nii"])
        self.assertEqual(result, "scan3")

    def test_strip_reconstruction_suffix(self):
        self.assertEqual(generate_consensus_ensemble._strip_reconstruction_suffix("scan1_2"), "scan1")
        self.assertEqual(generate_consensus_ensemble._strip_reconstruction_suffix("scan2"), "scan2")

    def test_normalize_columns(self):
        df = pd.DataFrame({"A": [1], "b": [2]})
        norm = generate_consensus_ensemble._normalize_columns(df)
        self.assertEqual(norm, {"a": "A", "b": "b"})

    def test_get_case_insensitive_column(self):
        df = pd.DataFrame({"A": [1], "b": [2]})
        col = generate_consensus_ensemble._get_case_insensitive_column(df, "a")
        self.assertEqual(col, "A")
        col = generate_consensus_ensemble._get_case_insensitive_column(df, "B")
        self.assertEqual(col, "b")
        with self.assertRaises(KeyError):
            generate_consensus_ensemble._get_case_insensitive_column(df, "c")

    def test_resolve_shared_labels(self):
        dfs = [self.df1, self.df2]
        labels = generate_consensus_ensemble._resolve_shared_labels(dfs, labels_arg="", exclude_cols_arg="VolumeName")
        self.assertIn("LabelA", labels)
        self.assertIn("LabelB", labels)
        labels_explicit = generate_consensus_ensemble._resolve_shared_labels(dfs, labels_arg="LabelA", exclude_cols_arg="VolumeName")
        self.assertEqual(labels_explicit, ["LabelA"])

    def test_load_and_normalize(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            self.df1.to_csv(path, index=False)
            df = generate_consensus_ensemble._load_and_normalize(
                csv_path=path,
                id_col_name="VolumeName",
                suffixes=[".nii.gz", ".nii"],
                keep_reconstruction_suffix=False,
            )
            self.assertIn("VolumeName", df.columns)
            self.assertEqual(len(df), 3)
            self.assertEqual(df["VolumeName"].iloc[0], "scan1")

    def test_threshold_for_strategy(self):
        self.assertEqual(generate_consensus_ensemble._threshold_for_strategy("any_positive", 3), 1)
        self.assertEqual(generate_consensus_ensemble._threshold_for_strategy("majority", 3), 2)
        self.assertEqual(generate_consensus_ensemble._threshold_for_strategy("unanimity", 3), 3)
        with self.assertRaises(ValueError):
            generate_consensus_ensemble._threshold_for_strategy("invalid", 3)

    def test_build_consensus(self):
        # Prepare merged DataFrame as expected by _build_consensus
        merged = self.df1.copy()
        merged = merged.rename(columns={"LabelA": "run1__LabelA", "LabelB": "run1__LabelB"})
        merged["run2__LabelA"] = self.df2["LabelA"]
        merged["run2__LabelB"] = self.df2["LabelB"]
        # any_positive threshold = 1
        consensus = generate_consensus_ensemble._build_consensus(
            merged=merged,
            labels=self.labels,
            set_names=self.set_names,
            threshold=1,
        )
        self.assertIn("VolumeName", consensus.columns)
        self.assertIn("LabelA", consensus.columns)
        self.assertIn("LabelB", consensus.columns)
        self.assertTrue((consensus["LabelA"] == [1, 1, 1]).all())
        self.assertTrue((consensus["LabelB"] == [0, 1, 1]).all())
        # unanimity threshold = 2
        consensus_unanimity = generate_consensus_ensemble._build_consensus(
            merged=merged,
            labels=self.labels,
            set_names=self.set_names,
            threshold=2,
        )
        self.assertTrue((consensus_unanimity["LabelA"] == [1, 0, 0]).all())
        self.assertTrue((consensus_unanimity["LabelB"] == [0, 1, 0]).all())

if __name__ == "__main__":
    unittest.main()
