import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from ctr_labeling.utils import estimate_cost, UNPRICED_MODELS_WARNED


class TestEstimateCost(unittest.TestCase):
    def setUp(self):
        UNPRICED_MODELS_WARNED.clear()

    def test_returns_zero_when_pricing_missing(self):
        cost = estimate_cost("gpt-unknown", 1000, 100, pricing_table={})
        self.assertEqual(cost, 0.0)

    def test_returns_zero_when_rates_missing(self):
        pricing = {"gpt-5": {"input_per_million": 0.1}}
        cost = estimate_cost("gpt-5", 1000, 100, pricing)
        self.assertEqual(cost, 0.0)

    def test_matches_longer_key_first(self):
        pricing = {
            "gpt-4": {"input_per_million": 1, "output_per_million": 2},
            "gpt-4-turbo": {"input_per_million": 2, "output_per_million": 4},
        }
        cost = estimate_cost("gpt-4-turbo-preview", 1000, 100, pricing)
        expected = round((1000 / 1_000_000 * 2) + (100 / 1_000_000 * 4), 8)
        self.assertEqual(cost, expected)


if __name__ == '__main__':
    unittest.main()
