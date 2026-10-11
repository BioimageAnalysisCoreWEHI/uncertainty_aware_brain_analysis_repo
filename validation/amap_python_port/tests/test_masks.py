import unittest

import numpy as np

from amap_validation import masks as M


class TestNearest(unittest.TestCase):
    def test_indices_quarter_scale(self):
        # u (1-based out) -> x = 4u - 1.5 -> box kernel picks floor(x + 0.5) = 4u - 1 (1-based) = 4u - 2 (0-based)
        np.testing.assert_array_equal(M.nearest_indices(12, 0.25), [2, 6, 10])
        # out of range index 4*3-1 = 11 (1-based) on length 9 is mirrored: aux(11) = 8 -> 0-based 7
        np.testing.assert_array_equal(M.nearest_indices(9, 0.25), [2, 6, 7])

    def test_canvas_output_size(self):
        self.assertEqual(M.matlab_output_length(12109, 0.25), 3028)
        self.assertEqual(M.matlab_output_length(9997, 0.25), 2500)

    def test_nearest_is_pure_subsample(self):
        a = np.arange(12 * 8).reshape(12, 8)
        b = M.imresize_nearest(a, 0.25)
        np.testing.assert_array_equal(b, a[np.ix_([2, 6, 10], [2, 6])])


class TestBicubic(unittest.TestCase):
    def test_weights_normalised(self):
        w, idx = M.contributions(101, 26, 0.25, "bicubic", antialiasing=True)
        np.testing.assert_allclose(w.sum(1), 1.0)
        self.assertTrue((idx >= 0).all() and (idx < 101).all())

    def test_constant_and_empty(self):
        self.assertTrue(M.imresize_logical_bicubic(np.ones((40, 36), bool), 0.25).all())
        self.assertFalse(M.imresize_logical_bicubic(np.zeros((40, 36), bool), 0.25).any())

    def test_large_disc_area_preserved(self):
        yy, xx = np.mgrid[:400, :400]
        disc = (xx - 200.3) ** 2 + (yy - 199.7) ** 2 < 120 ** 2
        small = M.imresize_logical_bicubic(disc, 0.25)
        self.assertEqual(small.shape, (100, 100))
        self.assertAlmostEqual(small.sum() * 16 / disc.sum(), 1.0, delta=0.03)
        near = M.imresize_nearest(disc, 0.25)
        self.assertAlmostEqual(near.sum() * 16 / disc.sum(), 1.0, delta=0.03)


if __name__ == "__main__":
    unittest.main()
