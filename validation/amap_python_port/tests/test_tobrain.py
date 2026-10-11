import os
import tempfile
import unittest

import numpy as np

from amap_validation import nifti
from amap_validation.config import StackGeometry, load_config
from amap_validation.geometry import StackMapping, canvas_to_volume, stack_plane_from_volume
from amap_validation.scoring import dice
from amap_validation.tobrain import write_label_map


def rot(ax, ay, az):
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def synthetic_mapping(tilt=2e-3):
    m = np.eye(4)
    m[:3, :3] = rot(tilt, -tilt, 0.5 * tilt)  # larger than any real stack (max |R-I| = 1.7e-3)
    m[:3, 3] = [0.2, 0.01, -0.3]
    geom = StackGeometry("t", "s", (10, 60), m, width=1600, height=1200)
    return StackMapping(geom, 0.25), (200, 90, 150)


def ellipse(shape=(400, 300), c=(210, 140), r=(110, 70)):
    i, j = np.mgrid[: shape[0], : shape[1]]
    return (((i - c[0]) / r[0]) ** 2 + ((j - c[1]) / r[1]) ** 2 < 1).astype(np.uint8)


class TestCanvasToVolume(unittest.TestCase):
    def setUp(self):
        self.mp, self.shape = synthetic_mapping()
        self.mask = ellipse()
        self.J0 = 17.5
        self.vox, _ = canvas_to_volume(self.mask, self.mp, self.J0, self.shape)

    def test_one_voxel_per_column_on_the_plane(self):
        # a deliberately strong tilt (0.05 rad) so the plane crosses several nii y levels within the mask
        for mp in (self.mp, synthetic_mapping(tilt=0.05)[0]):
            vox, _ = canvas_to_volume(self.mask, mp, self.J0, self.shape)
            cols = {(x, z) for x, _, z in vox}
            self.assertEqual(len(cols), len(vox))
            J = mp.nii_to_staple(vox.astype(float))[2]
            half_voxel_in_J = 0.5 * abs(mp._plane_coeffs()[0][1])  # ~0.42 slice
            self.assertLessEqual(np.abs(J - self.J0).max(), half_voxel_in_J + 1e-9)
        self.assertGreater(vox[:, 1].max() - vox[:, 1].min(), 2)  # strong tilt: several y levels

    def test_area_preserved(self):
        ratio = len(self.vox) * 12.5 ** 2 / (self.mask.sum() * 4 ** 2)
        self.assertAlmostEqual(ratio, 1.0, delta=0.05)

    def test_round_trip_through_code_forward_path(self):
        back = stack_plane_from_volume(self.vox, np.ones(len(self.vox), np.uint8), self.mp, self.J0, self.mask.shape)
        self.assertGreater(dice(back, self.mask), 0.9)

    def test_probability_equals_mean_of_rater_volumes(self):
        raters = [ellipse(c=(200 + 8 * k, 140 - 5 * k)) for k in range(3)]
        prob = np.mean(np.stack(raters), 0).astype(np.float32)
        pv, pval = canvas_to_volume(prob, self.mp, self.J0, self.shape)
        count = {}
        for r in raters:
            v, _ = canvas_to_volume(r, self.mp, self.J0, self.shape)
            for t in map(tuple, v):
                count[t] = count.get(t, 0) + 1
        pmap = dict(zip(map(tuple, pv), pval))
        self.assertEqual(set(pmap), set(count))
        for t, c in count.items():
            self.assertAlmostEqual(pmap[t], c / 3, places=6)


class TestSparseNifti(unittest.TestCase):
    def test_streaming_writer_matches_dense_and_reads_back(self):
        rng = np.random.default_rng(1)
        shape = (13, 7, 9)
        dense = np.zeros(shape, np.float32)
        idx = rng.integers(0, 13 * 7 * 9, 40)
        dense.flat[idx] = rng.random(40).astype(np.float32) + 0.1
        vox = np.argwhere(dense)
        aff = np.diag([0.0125, 0.0125, 0.0125, 1.0])
        aff[:3, 3] = 0.0125
        with tempfile.TemporaryDirectory() as d:
            a, b, c = (os.path.join(d, n) for n in ("dense.nii", "sparse.nii", "sparse.nii.gz"))
            nifti.save(a, dense, aff)
            nifti.save_sparse(b, shape, np.float32, vox, dense[tuple(vox.T)], aff)
            nifti.save_sparse(c, shape, np.float32, vox, dense[tuple(vox.T)], aff)
            from pathlib import Path
            self.assertEqual(Path(a).read_bytes(), Path(b).read_bytes())
            v2, val2 = nifti.load_sparse(c)
            back = np.zeros(shape, np.float32)
            back[tuple(v2.T)] = val2
            np.testing.assert_array_equal(back, dense)
            np.testing.assert_array_equal(nifti.load(c), dense)
            self.assertEqual(nifti.read_header(c)["shape"], shape)


class TestLabelMap(unittest.TestCase):
    def test_lowest_id_wins_and_overlap_counted(self):
        cfg = load_config()
        aff = np.eye(4)
        grid = ((5, 5, 5), aff)
        a = np.array([[1, 1, 1], [2, 2, 2]])
        b = np.array([[2, 2, 2], [3, 3, 3]])
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            meta = write_label_map(cfg, "AB_Gad67_223", [(6, "V2M", b), (4, "RSP", a)], grid, Path(d))
            v, lab = nifti.load_sparse(Path(d) / "AB_Gad67_223_STAPLE_labels.nii.gz")
        got = {tuple(x): int(y) for x, y in zip(v, lab)}
        self.assertEqual(got, {(1, 1, 1): 4, (2, 2, 2): 4, (3, 3, 3): 6})
        self.assertEqual(meta["overlap_voxels"], 1)
        self.assertFalse(meta["complete"])


if __name__ == "__main__":
    unittest.main()
