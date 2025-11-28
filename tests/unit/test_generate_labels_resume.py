import csv
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from scripts.generate_labels import load_resume_data, remove_hash_record  # noqa: E402  pylint: disable=wrong-import-position


class TestResumeHelpers(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.csv_path = os.path.join(self._tmpdir.name, 'resume.csv')

    def _write_rows(self, rows):
        fieldnames = sorted(rows[0].keys())
        with open(self.csv_path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_load_resume_data_groups_by_volume_and_hash(self):
        rows = [
            {"VolumeName": "vol-1", "report_hash": "hash-a", "status": "success"},
            {"VolumeName": "vol-2", "report_hash": "hash-a", "status": "success"},
            {"VolumeName": "vol-3", "report_hash": "hash-b", "status": "error"},
        ]
        self._write_rows(rows)

        by_volume, by_hash = load_resume_data(self.csv_path)
        self.assertEqual(set(by_volume.keys()), {"vol-1", "vol-2", "vol-3"})
        self.assertEqual(len(by_hash["hash-a"]), 2)
        self.assertEqual(by_hash["hash-b"][0]["VolumeName"], "vol-3")

    def test_remove_hash_record_pops_only_target_entry(self):
        rows = [
            {"VolumeName": "vol-1", "report_hash": "hash-a"},
            {"VolumeName": "vol-2", "report_hash": "hash-a"},
        ]
        self._write_rows(rows)
        _, by_hash = load_resume_data(self.csv_path)

        target_record = by_hash["hash-a"][0]
        remove_hash_record(by_hash, "hash-a", target_record)
        self.assertEqual(len(by_hash["hash-a"]), 1)
        self.assertEqual(by_hash["hash-a"][0]["VolumeName"], "vol-2")

        remove_hash_record(by_hash, "hash-a", by_hash["hash-a"][0])
        self.assertNotIn("hash-a", by_hash)


if __name__ == '__main__':
    unittest.main()
