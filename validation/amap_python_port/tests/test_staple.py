import unittest

import numpy as np

from amap_validation.scoring import bwperim8, dice, hausdorff, pad_to_larger
from amap_validation.staple import matlab_quantile, sba, staple, valid_slices


def synthetic(n_raters=11, shape=(300, 300), seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    truth = (xx - 150) ** 2 / 90 ** 2 + (yy - 150) ** 2 / 60 ** 2 < 1
    p_true = rng.uniform(0.75, 0.98, n_raters)
    q_true = rng.uniform(0.95, 0.995, n_raters)
    raters = []
    for p, q in zip(p_true, q_true):
        u = rng.random(shape)
        raters.append(np.where(truth, u < p, u > q))
    return truth, np.array(raters), p_true, q_true


class TestStaple(unittest.TestCase):
    def test_recovers_truth_and_parameters_without_unc(self):
        truth, raters, p_true, q_true = synthetic()
        res = staple(raters, unc=False)
        self.assertTrue(res.converged)
        self.assertGreater(dice(res.labels, truth), 0.99)
        np.testing.assert_allclose(res.sensitivity, p_true, atol=0.02)
        np.testing.assert_allclose(res.specificity, q_true, atol=0.005)

    def test_unc_keeps_consensus_voxels(self):
        truth, raters, _, _ = synthetic(seed=1)
        res = staple(raters, unc=True)
        all_on = raters.all(0)
        all_off = ~raters.any(0)
        self.assertTrue(res.labels[all_on].all())
        self.assertFalse(res.labels[all_off].any())
        self.assertGreater(dice(res.labels, truth), 0.98)

    def test_agrees_with_simpleitk(self):
        try:
            import SimpleITK as sitk
        except ImportError:
            self.skipTest("SimpleITK not installed")
        truth, raters, _, _ = synthetic(seed=2)
        ours = staple(raters, unc=False)
        imgs = [sitk.GetImageFromArray(r.astype(np.uint8)) for r in raters]
        f = sitk.STAPLEImageFilter()
        f.SetForegroundValue(1)
        prob = sitk.GetArrayFromImage(f.Execute(imgs))
        self.assertGreater(dice(ours.labels, prob > 0.5), 0.995)

    def test_identical_raters(self):
        m = np.zeros((5, 20, 20), bool)
        m[:, 5:10, 5:12] = True
        res = staple(m)
        np.testing.assert_array_equal(res.labels, m[0])

    def test_sba_of_identical_shapes(self):
        m = np.zeros((3, 60, 60), bool)
        m[:, 10:40, 20:50] = True
        np.testing.assert_array_equal(sba(m), m[0])


class TestSlices(unittest.TestCase):
    def test_default_all(self):
        np.testing.assert_array_equal(valid_slices([3, 9, 5]), np.arange(3, 10))

    def test_range_window(self):
        s = [1, 2, 10, 11, 11, 12, 30]
        np.testing.assert_array_equal(valid_slices(s, "range", 3), [10, 11, 12])

    def test_matlab_quantile(self):
        np.testing.assert_allclose(matlab_quantile([1, 2, 3, 4], [0.25, 0.75]), [1.5, 3.5])
        np.testing.assert_allclose(matlab_quantile([5], [0.25, 0.75]), [5, 5])


class TestScoring(unittest.TestCase):
    def test_dice(self):
        a = np.zeros((10, 10)); a[:5] = 1
        b = np.zeros((10, 10)); b[:, :5] = 1
        self.assertAlmostEqual(dice(a, b), 0.5)

    def test_pad_top_left(self):
        a, b = pad_to_larger(np.ones((2, 3)), np.ones((4, 1)))
        self.assertEqual(a.shape, (4, 3))
        self.assertEqual(a[3, 2], 0)
        self.assertEqual(b[0, 2], 0)

    def test_hausdorff_shifted_square(self):
        a = np.zeros((50, 50), bool); a[10:20, 10:20] = True
        b = np.zeros((50, 50), bool); b[10:20, 13:23] = True
        self.assertAlmostEqual(hausdorff(a, b), 3.0)
        self.assertTrue(np.isnan(hausdorff(a, np.zeros_like(a))))

    def test_bwperim_8(self):
        a = np.zeros((7, 7), bool); a[1:6, 1:6] = True
        per = bwperim8(a)
        self.assertEqual(per.sum(), 16)
        self.assertFalse(per[3, 3])


if __name__ == "__main__":
    unittest.main()
