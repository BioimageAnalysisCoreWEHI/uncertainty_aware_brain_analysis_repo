import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)
import unittest
from pathlib import Path

import numpy as np

from amap_validation.compare import (SAMPLE_PREFIX, _reliability, _score_stack, hemisphere_keep, md_table,
                                     to_our_grid)
from amap_validation.config import MIDLINES, load_config
from amap_validation.reslice import AsrGrid

PIPELINE_ROOT = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\results"))


def sform(n):
    s = np.diag([0.0125, 0.0125, 0.0125, 1.0])
    s[:3, 3] = 0.0125
    return s


class TestHelpers(unittest.TestCase):
    def test_hemisphere_keep_matches_midline_and_mirror(self):
        shape = (1078, 1218, 729)
        grid = AsrGrid(shape, sform(shape))
        keep = hemisphere_keep(grid, "AB_Gad67_223", "RSP")
        m = MIDLINES["AB_Gad67_223"]  # 1-based last zeroed x -> keep 0-based x >= m
        x_of_R = grid.nx - 1 - (2 * np.arange(grid.shape25[2]) + 0.5)
        np.testing.assert_array_equal(keep, x_of_R >= m - 0.5)
        self.assertTrue(keep[0] and not keep[-1])  # mirrored: segmented side at low R
        flipped = hemisphere_keep(AsrGrid((1074, 1220, 730), sform(0)), "MV_Ntsr1_169", "SG")
        self.assertFalse(flipped[0])

    def test_to_our_grid_pads_and_shifts(self):
        sl = np.arange(12).reshape(3, 4)
        out = to_our_grid(sl, (4, 4))
        np.testing.assert_array_equal(out[:3], sl)
        self.assertTrue((out[3] == 0).all())
        out = to_our_grid(sl, (4, 4), offset_sr=(1, 0))
        np.testing.assert_array_equal(out[1:], sl)

    def test_score_stack_picks_best_slice(self):
        truth = np.zeros((10, 10), bool)
        truth[2:6, 2:6] = True
        pages = np.zeros((3, 10, 10), np.uint8)
        pages[0, 0:3, 0:3] = 1
        pages[1, 2:6, 2:6] = 1        # identical at A = 11
        pages[2, 2:6, 3:7] = 1
        keep = np.ones(10, bool)
        s = _score_stack(pages, [10, 11, 12], truth, A0=12, keep_r=keep, target_sr=(10, 10))
        self.assertEqual(s["A_best"], 11)
        self.assertAlmostEqual(s["dice_paper"], 1.0)
        self.assertAlmostEqual(s["dice_fixed"], 0.75)
        keep_left = np.arange(10) < 4  # hemisphere cut removes the right half
        s = _score_stack(pages, [10, 11, 12], truth, A0=12, keep_r=keep_left, target_sr=(10, 10))
        self.assertLess(s["dice_paper"], 1.0)

    def test_score_stack_with_per_slice_hemisphere(self):
        truth = np.zeros((10, 10), bool)
        truth[2:6, 2:6] = True
        pages = np.zeros((2, 10, 10), np.uint8)
        pages[:, 2:6, 2:8] = 1                      # pipeline spills 2 columns past the midline
        keep = np.ones((2, 10, 10), bool)
        keep[1, :, 6:] = False                      # slice 1's hemisphere map removes the spill
        s = _score_stack(pages, [0, 1], truth, A0=0, keep_r=keep, target_sr=(10, 10))
        self.assertEqual(s["A_best"], 1)
        self.assertAlmostEqual(s["dice_paper"], 1.0)
        self.assertAlmostEqual(s["dice_fixed"], 2 * 16 / (16 + 24))

    def test_hemisphere_vote_and_label(self):
        from amap_validation.compare import RegisteredHemispheres
        hem = RegisteredHemispheres.__new__(RegisteredHemispheres)
        hem.A_list, hem.shape = [5], (1, 4, 4)
        h = np.array([[[1, 1, 2, 2]] * 4])
        hem.vote = h.astype(np.uint8)
        hem.packed = {"r": (np.packbits(h == 1), np.packbits(h == 2))}
        truth = np.zeros((4, 4), bool)
        truth[:, 2:] = True
        self.assertEqual(hem.annotated_label(truth, 5), 2)
        np.testing.assert_array_equal(hem.run_label("r", 2), h == 2)

    def test_reliability_bins(self):
        p = np.array([[0, 0.1, 0.5, 1.0]])
        truth = np.array([[True, False, True, True]])
        r = _reliability(p, truth)                     # bins: p = 0, (0, .2], ..., (.8, 1), p = 1
        self.assertEqual(r[0]["n_pixels"], 1)          # the expert pixel the pipeline missed (p = 0)
        self.assertEqual(r[0]["frac_expert"], 1.0)
        self.assertEqual(r[1]["n_pixels"], 1)          # p = 0.1, outside the expert consensus
        self.assertEqual(r[1]["frac_expert"], 0.0)
        self.assertEqual(r[-1]["frac_expert"], 1.0)    # p = 1 inside

    def test_md_table(self):
        import pandas as pd
        txt = md_table(pd.DataFrame(dict(a=[1.23456], b=["x"])), index=False)
        self.assertIn("| 1.235 | x |", txt)


class TestPipelineFolders(unittest.TestCase):
    @unittest.skipUnless((PIPELINE_ROOT / "region_consensus").exists(), "pipeline results not reachable")
    def test_every_brain_maps_to_one_sample_folder(self):
        names = [p.name for p in (PIPELINE_ROOT / "region_consensus").iterdir() if p.is_dir()]
        cfg = load_config()
        for brain in cfg.brains:
            self.assertEqual(sum(n.startswith(SAMPLE_PREFIX[brain]) for n in names), 1, brain)


if __name__ == "__main__":
    unittest.main()
