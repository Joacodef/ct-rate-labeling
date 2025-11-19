import unittest
import sys
import os
from typing import Dict

# Add the src directory to the system path to allow imports
# This assumes the test is run from the project root or the tests folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from ctr_labeling.parsing import validate_response

class TestGenerateLabels(unittest.TestCase):
    def setUp(self):
        """Set up a list of dummy target labels for testing."""
        self.target_labels = ["Abnormality_A", "Abnormality_B", "Abnormality_C"]

    def test_validate_response_perfect_match(self):
        """Test that a perfectly formatted response is returned as is."""
        response = {"Abnormality_A": 1, "Abnormality_B": 0, "Abnormality_C": 1}
        result = validate_response(response, self.target_labels)
        self.assertEqual(result, response)

    def test_validate_response_missing_keys(self):
        """Test that missing keys are defaulted to 0."""
        response = {"Abnormality_A": 1}
        expected = {"Abnormality_A": 1, "Abnormality_B": 0, "Abnormality_C": 0}
        result = validate_response(response, self.target_labels)
        self.assertEqual(result, expected)

    def test_validate_response_extra_keys(self):
        """Test that keys not in the target list are ignored."""
        response = {
            "Abnormality_A": 1, 
            "Abnormality_B": 0, 
            "Abnormality_C": 1,
            "Random_Hallucination": 1
        }
        expected = {"Abnormality_A": 1, "Abnormality_B": 0, "Abnormality_C": 1}
        result = validate_response(response, self.target_labels)
        self.assertEqual(result, expected)

    def test_validate_response_string_integers(self):
        """Test that string representations of 0 and 1 are converted correctly."""
        response = {"Abnormality_A": "1", "Abnormality_B": "0", "Abnormality_C": 1}
        expected = {"Abnormality_A": 1, "Abnormality_B": 0, "Abnormality_C": 1}
        result = validate_response(response, self.target_labels)
        self.assertEqual(result, expected)

    def test_validate_response_robust_normalization(self):
        """Test that values like 'present' or non-standard integers are normalized to 1."""
        response = {
            "Abnormality_A": "present", # Should normalize to 1
            "Abnormality_B": 0,
            "Abnormality_C": 5          # Should normalize to 1
        }
        expected = {"Abnormality_A": 1, "Abnormality_B": 0, "Abnormality_C": 1}
        result = validate_response(response, self.target_labels)
        self.assertEqual(result, expected)

    def test_validate_response_none_values(self):
        """Test that None values are defaulted to 0."""
        response = {"Abnormality_A": None, "Abnormality_B": 1, "Abnormality_C": 0}
        expected = {"Abnormality_A": 0, "Abnormality_B": 1, "Abnormality_C": 0}
        result = validate_response(response, self.target_labels)
        self.assertEqual(result, expected)

    def test_validate_response_truly_invalid_values(self):
        """Test that unrecognized values are safely defaulted to 0."""
        response = {
            "Abnormality_A": "banana",  # Completely invalid string
            "Abnormality_B": 0,
            "Abnormality_C": "unsure"   # Unrecognized string
        }
        expected = {"Abnormality_A": 0, "Abnormality_B": 0, "Abnormality_C": 0}
        result = validate_response(response, self.target_labels)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()