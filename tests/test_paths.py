"""Path placeholders resolve correctly (python -m unittest discover -s tests)."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import uaa_paths  # noqa: E402


class TestPaths(unittest.TestCase):
    def setUp(self):
        self.env = dict(os.environ)
        os.environ.update(UAA_IMAGING_ROOT="/data/project", UAA_SCRATCH="/scratch/me", UAA_DATA_ROOT="/data/deposit")
        uaa_paths._CFG = {"cluster_prefixes": {"/cluster/imaging": "<IMAGING>", "/cluster/scratch/me": "<SCRATCH>"}}

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env)
        uaa_paths._CFG = None

    def sep(self, s):
        return s.replace("/", os.sep)

    def test_placeholders(self):
        self.assertEqual(uaa_paths.P("<IMAGING>/a/b.tif"), self.sep("/data/project/a/b.tif"))
        self.assertEqual(uaa_paths.P(r"<SCRATCH>\sweep"), self.sep("/scratch/me/sweep"))
        self.assertEqual(uaa_paths.P("<DATA>/cellpose_model/x"), self.sep("/data/deposit/cellpose_model/x"))
        self.assertTrue(uaa_paths.P("<REPO>/pipeline").endswith("pipeline"))

    def test_cluster_round_trip(self):
        local = uaa_paths.to_local("/cluster/imaging/results/qc.csv")
        self.assertEqual(local, self.sep("/data/project/results/qc.csv"))
        self.assertEqual(uaa_paths.to_cluster(local), "/cluster/imaging/results/qc.csv")

    def test_unconfigured_placeholder_raises(self):
        os.environ.pop("UAA_DATA_ROOT")
        with self.assertRaises(RuntimeError):
            uaa_paths.P("<DATA>/x")


if __name__ == "__main__":
    unittest.main()
