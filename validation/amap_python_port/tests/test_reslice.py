import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from amap_validation.reslice import AsrGrid, downsample_to_asr25, save_tiff, sheet_to_asr25


def sform12(nx, ny, nz):
    s = np.diag([0.0125, 0.0125, 0.0125, 1.0])
    s[:3, 3] = 0.0125
    return s


def dense_asr12(vol, mirror_lr):
    """Reference reorientation: [a, s, r] = vol[x(r), ny-1-a, nz-1-s] with x(r) = nx-1-r (mirror) or r."""
    v = vol[::-1, ::-1, ::-1] if mirror_lr else vol[:, ::-1, ::-1]
    return np.transpose(v, (1, 2, 0))


class TestAsrGrid(unittest.TestCase):
    def setUp(self):
        self.shape = (7, 9, 5)  # odd everywhere
        self.vol = np.random.default_rng(0).integers(0, 60000, self.shape).astype(np.uint16)

    def grids(self):
        return [(m, AsrGrid(self.shape, sform12(*self.shape), mirror_lr=m)) for m in (True, False)]

    def test_default_is_mirrored(self):
        self.assertTrue(AsrGrid(self.shape, sform12(*self.shape)).mirror_lr)

    def test_reorientation_is_exact_permutation(self):
        for m, g in self.grids():
            dense = dense_asr12(self.vol, m)
            self.assertEqual(dense.shape, g.shape_asr12)
            idx = np.argwhere(np.ones(g.shape_asr12, bool))
            v = g.from_asr12(idx)
            np.testing.assert_array_equal(dense[tuple(idx.T)], self.vol[tuple(v.T)])
            np.testing.assert_array_equal(g.to_asr12(v), idx)
            # a grows posterior (y decreases), s grows inferior (z decreases); r = nx-1-x (mirror) or x
            self.assertEqual(tuple(g.from_asr12(np.array([0, 0, 0]))), (6 if m else 0, 8, 4))

    def test_block_mean_matches_reference_loop(self):
        for m, g in self.grids():
            out = downsample_to_asr25(self.vol, g)
            dense = dense_asr12(self.vol, m).astype(float)
            self.assertEqual(out.shape, g.shape25)
            for A in range(g.shape25[0]):
                for S in range(g.shape25[1]):
                    for R in range(g.shape25[2]):
                        blk = dense[2 * A:2 * A + 2, 2 * S:2 * S + 2, 2 * R:2 * R + 2]
                        self.assertEqual(out[A, S, R], np.floor(blk.mean() + 0.5))

    def test_mirror_is_exact_left_right_flip(self):
        (_, gm), (_, gp) = self.grids()
        np.testing.assert_array_equal(downsample_to_asr25(self.vol[::-1], gp), downsample_to_asr25(self.vol, gm))

    def test_affine_maps_block_centre_to_mean_of_children(self):
        for _, g in self.grids():
            for A in range(g.ny // 2):
                for S in range(g.nz // 2):
                    for R in range(g.nx // 2):
                        kids = np.array([[a, s, r] for a in (2 * A, 2 * A + 1) for s in (2 * S, 2 * S + 1)
                                         for r in (2 * R, 2 * R + 1)])
                        w = (g.sform12 @ np.c_[g.from_asr12(kids), np.ones(8)].T)[:3].mean(1)
                        np.testing.assert_allclose((g.affine25 @ [A, S, R, 1])[:3], w, atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(g.affine25[:3, :3], axis=0), 0.025)

    def test_sheet_to_asr25_matches_dense_reference(self):
        g = AsrGrid((12, 10, 8), sform12(12, 10, 8))
        rng = np.random.default_rng(3)
        xs, zs = np.meshgrid(np.arange(2, 11), np.arange(1, 7), indexing="ij")
        ys = np.where(xs > 6, 4, 5)  # a sheet crossing two y levels (tilt)
        vox = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], 1)
        vals = rng.random(len(vox)).astype(np.float32)
        img, A, A_all, a_rng = sheet_to_asr25(vox, vals, g, binary=False)
        ref = np.zeros((g.nz, g.nx))
        u = g.to_asr12(vox)
        ref[u[:, 1], u[:, 2]] = vals
        ref25 = np.array([[ref[2 * S:2 * S + 2, 2 * R:2 * R + 2].mean() for R in range(6)] for S in range(4)])
        np.testing.assert_allclose(img, ref25, rtol=1e-6)
        self.assertEqual(A_all, sorted({(10 - 1 - 4) // 2, (10 - 1 - 5) // 2}))
        mimg, *_ = sheet_to_asr25(vox, np.ones(len(vox)), g, binary=True)
        coverage = np.array([[(ref[2 * S:2 * S + 2, 2 * R:2 * R + 2] > 0).mean() for R in range(6)] for S in range(4)])
        np.testing.assert_array_equal(mimg, (coverage >= 0.5).astype(np.uint8))


class TestLayout(unittest.TestCase):
    def test_paths_and_legacy_migration(self):
        import pandas as pd
        from amap_validation.reslice import AsrLayout, migrate_legacy_layout
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            lay = AsrLayout(d)
            self.assertEqual(lay.rel(lay.staple("AB_Gad67_223", "CING")),
                             "AB_Gad67_223/ACA/AB_Gad67_223CING_STAPLE_asr25um.nii")
            self.assertEqual(lay.rel(lay.rater("MV_Ntsr1_169", "SG", "ownSlice", "User_01", 10, 1, 35)),
                             "MV_Ntsr1_169/DG-sg/raters/ownSlice/User_01_Seg10_att1_slice35_ownSlice_asr25um.nii")
            (d / "brains").mkdir()
            (d / "structures").mkdir()
            for n in ("AB_Gad67_223_asr_25um.nii", "AB_Gad67_223_transform.json"):
                (d / "brains" / n).write_text("x")
            for n in ("AB_Gad67_223V2M_STAPLE_asr25um.tif", "AB_Gad67_223V1_raterProbability_asr25um.nii"):
                (d / "structures" / n).write_text("x")
            (d / "asr25um_checks_AB_Gad67_223.json").write_text("{}")
            pd.DataFrame(dict(brain=["AB_Gad67_223"], region=["V2M"], staple_file=["old"],
                              probability_file=["old"])).to_csv(lay.slices_log, index=False)
            migrate_legacy_layout(d, log=lambda *_: None)
            self.assertTrue((d / "AB_Gad67_223" / "AB_Gad67_223_asr_25um.nii").exists())
            self.assertTrue((d / "AB_Gad67_223" / "VISam" / "AB_Gad67_223V2M_STAPLE_asr25um.tif").exists())
            self.assertTrue((d / "AB_Gad67_223" / "VISp" / "AB_Gad67_223V1_raterProbability_asr25um.nii").exists())
            self.assertTrue(lay.checks("AB_Gad67_223").exists())
            self.assertFalse((d / "brains").exists() or (d / "structures").exists())
            t = pd.read_csv(lay.slices_log)
            self.assertEqual(t.staple_file[0], "AB_Gad67_223/VISam/AB_Gad67_223V2M_STAPLE_asr25um.nii")

    def test_empty_sheet(self):
        g = AsrGrid((8, 6, 4), sform12(8, 6, 4))
        img, A, A_all, a_rng = sheet_to_asr25(np.zeros((0, 3), int), np.zeros(0), g, binary=True)
        self.assertIsNone(A)
        self.assertEqual(img.shape, g.shape25[1:])
        self.assertFalse(img.any())


class TestTiff(unittest.TestCase):
    def test_round_trip_with_spacing(self):
        data = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.tif"
            save_tiff(p, data)
            with tifffile.TiffFile(p) as tf:
                back = tf.asarray()
                meta = tf.imagej_metadata
        np.testing.assert_array_equal(back, data)
        self.assertAlmostEqual(meta["spacing"], 25.0)
        self.assertEqual(meta["unit"], "um")


if __name__ == "__main__":
    unittest.main()
