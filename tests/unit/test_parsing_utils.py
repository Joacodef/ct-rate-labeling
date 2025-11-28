import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from ctr_labeling.parsing import clean_and_parse_json


class TestParsingHelpers(unittest.TestCase):
    def test_clean_and_parse_json_strips_markdown(self):
        payload = """```json\n{\"Label_A\": 1, \"Label_B\": 0}\n```"""
        parsed = clean_and_parse_json(payload)
        self.assertEqual(parsed, {"Label_A": 1, "Label_B": 0})

    def test_clean_and_parse_json_invalid(self):
        with self.assertRaises(ValueError):
            clean_and_parse_json("```not json```")


if __name__ == '__main__':
    unittest.main()
