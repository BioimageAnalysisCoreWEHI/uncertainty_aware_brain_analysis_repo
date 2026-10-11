import os
import tempfile
import unittest

import numpy as np

from amap_validation import nifti
from amap_validation.config import load_config


class TestNifti(unittest.TestCase):
    def test_round_trip_matlab_style_header(self):
        a = (np.random.default_rng(0).random((7, 5)) > 0.5).astype(np.uint8)
        aff = np.eye(4)
        aff[:3, 3] = 1.0
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.nii")
            nifti.save(p, a, aff)
            h = nifti.read_header(p)
            b = nifti.load(p)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(h["shape"], (7, 5))
        self.assertEqual((h["sform_code"], h["qform_code"]), (1, 0))
        np.testing.assert_allclose(h["sform"], aff)

    def test_real_brain_header(self):
        cfg = load_config()
        h = nifti.read_header(cfg.brain_file("AB_Gad67_223"))
        self.assertEqual(h["shape"], (1078, 1218, 729))
        self.assertEqual(h["dtype"], np.dtype("<u2"))
        np.testing.assert_allclose(h["pixdim"][1:4], 0.0125, rtol=1e-6)
        np.testing.assert_allclose(h["sform"][:3, 3], 0.0125, rtol=1e-6)
        self.assertEqual(h["sform"].shape, (4, 4))

    def test_memmap_matches_header_shape(self):
        cfg = load_config()
        v = nifti.load(cfg.brain_file("ER_Glt25d2Cre_9"), mmap=True)
        self.assertEqual(v.shape, (860, 1280, 604))
        self.assertEqual(v.dtype, np.dtype("u1"))


if __name__ == "__main__":
    unittest.main()
