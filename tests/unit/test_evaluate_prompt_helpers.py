import os
import sys
import unittest

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from scripts import evaluate_prompt  # noqa: E402  pylint: disable=wrong-import-position


class TestEvaluatePromptHelpers(unittest.TestCase):
    def test_zero_support_no_false_positives_scores_one(self):
        df = pd.DataFrame({
            "true_Label_A": [0, 0, 0],
            "pred_Label_A": [0, 0, 0],
        })
        stats = evaluate_prompt.calculate_binary_metrics(df, "true_Label_A", "pred_Label_A")
        self.assertEqual(stats["precision"], 1.0)
        self.assertEqual(stats["recall"], 1.0)
        self.assertEqual(stats["f1"], 1.0)

    def test_zero_support_with_false_positive_scores_zero(self):
        df = pd.DataFrame({
            "true_Label_A": [0, 0, 0],
            "pred_Label_A": [0, 1, 0],
        })
        stats = evaluate_prompt.calculate_binary_metrics(df, "true_Label_A", "pred_Label_A")
        self.assertEqual(stats["precision"], 0.0)
        self.assertEqual(stats["recall"], 0.0)
        self.assertEqual(stats["f1"], 0.0)

    def test_handles_non_numeric_inputs(self):
        df = pd.DataFrame({
            "true_Label_A": ["1", None, "0"],
            "pred_Label_A": ["1", "0", "1"],
        })
        stats = evaluate_prompt.calculate_binary_metrics(df, "true_Label_A", "pred_Label_A")
        self.assertAlmostEqual(stats["precision"], 0.5)
        self.assertAlmostEqual(stats["recall"], 1.0)
        self.assertAlmostEqual(stats["f1"], 0.667, places=3)


if __name__ == '__main__':
    unittest.main()
