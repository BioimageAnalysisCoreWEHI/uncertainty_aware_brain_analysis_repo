import glob
import os
import re
import unittest

import numpy as np

from amap_validation.config import DEFAULT_SEG_DIR, load_config
from amap_validation.geometry import StackMapping, matlab_round
from amap_validation.roi import read_roi

ROI_RE = re.compile(r"Seg(\d+)(.+)_([A-Za-z0-9]+)\.tif\d{3}-([A-Za-z0-9]+)_Slice(\d+)")


def all_rois():
    for p in sorted(glob.glob(str(DEFAULT_SEG_DIR / "User_*" / "*.roi"))):
        m = ROI_RE.match(os.path.basename(p))
        paths, _ = read_roi(p)
        if paths:
            yield m.group(2), m.group(3), int(m.group(5)), np.vstack(paths)


class TestCodeIndexing(unittest.TestCase):
    def test_permute_flip_matches_formula(self):
        # pavelAtlas.m:99-100: permute(img,[1 3 2]) then flip(.,2) => atlasImg(i,j,J) = ref(I=i, J, K=Hs+1-j)
        ws, n, hs = 5, 3, 4
        ref = np.zeros((ws, n, hs), int)
        for i in range(ws):
            for j in range(n):
                for k in range(hs):
                    ref[i, j, k] = (i + 1) * 10000 + (j + 1) * 100 + (k + 1)
        atlas = np.transpose(ref, (0, 2, 1))[:, ::-1, :]
        for i in range(1, ws + 1):
            for j in range(1, hs + 1):
                for J in range(1, n + 1):
                    self.assertEqual(atlas[i - 1, j - 1, J - 1], i * 10000 + J * 100 + (hs + 1 - j))

    def test_matlab_round_half_up(self):
        self.assertEqual(matlab_round(2396.75), 2397)
        self.assertEqual(matlab_round(2.5), 3)


class TestMatrices(unittest.TestCase):
    def test_rigid_and_near_identity(self):
        cfg = load_config()
        self.assertEqual(len(cfg.stacks), 48)
        for g in cfg.stacks.values():
            r = g.matrix[:3, :3]
            self.assertLess(np.abs(r @ r.T - np.eye(3)).max(), 5e-6)
            self.assertAlmostEqual(np.linalg.det(r), 1.0, places=5)
            self.assertLess(np.abs(r - np.eye(3)).max(), 2e-3)
            np.testing.assert_array_equal(g.matrix[3], [0, 0, 0, 1])


class TestRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.rois = list(all_rois())

    def test_have_all_usable_rois(self):
        self.assertGreater(len(self.rois), 870)

    def test_exact_round_trip(self):
        worst_xy = worst_s = 0.0
        for brain, stack, s, pts in self.rois:
            m = StackMapping(self.cfg.stack(brain, stack), self.cfg.scale)
            v = m.tiff_to_nii(pts[:, 0], pts[:, 1], s)
            c, r, s2 = m.nii_to_tiff(v)
            worst_xy = max(worst_xy, np.hypot(c - pts[:, 0], r - pts[:, 1]).max())
            worst_s = max(worst_s, np.abs(s2 - s).max())
        self.assertLess(worst_xy, 1e-6)
        self.assertLess(worst_s, 1e-9)

    def test_discretised_round_trip(self):
        # TIFF px -> containing STAPLE pixel -> nearest nii voxel -> back to TIFF px
        errs, slice_ok, n = [], 0, 0
        for brain, stack, s, pts in self.rois:
            m = StackMapping(self.cfg.stack(brain, stack), self.cfg.scale)
            i0, j0 = np.floor(pts[:, 0] / 4), np.floor(pts[:, 1] / 4)
            v = matlab_round(m.staple_to_nii(i0, j0, s))
            c, r, s2 = m.nii_to_tiff(v)
            errs.append(np.hypot(c - pts[:, 0], r - pts[:, 1]))
            slice_ok += int((matlab_round(s2) == s).sum())
            n += len(pts)
        errs = np.concatenate(errs)
        # bound: STAPLE block (<= 3 um from its centre per axis... sqrt(2)*2.5) + half a 12.5 um voxel per axis
        self.assertLess(errs.max(), np.sqrt(2) * (2.5 + 6.25) + 1e-6)
        self.assertEqual(slice_ok, n)  # 0.5 voxel / 1.2 voxel-per-slice < 0.5 slice


class TestStapleCanvasMapsInsideBrain(unittest.TestCase):
    def test_roi_centroids_inside_volume(self):
        from amap_validation.nifti import read_header
        cfg = load_config()
        shapes = {b: read_header(cfg.brain_file(b))["shape"] for b in cfg.brains}
        for brain, stack, s, pts in all_rois():
            m = StackMapping(cfg.stack(brain, stack), cfg.scale)
            v = m.tiff_to_nii(pts[:, 0].mean(), pts[:, 1].mean(), s)
            for axis in range(3):
                self.assertTrue(0 <= v[axis] < shapes[brain][axis], (brain, stack, s, v))


if __name__ == "__main__":
    unittest.main()
