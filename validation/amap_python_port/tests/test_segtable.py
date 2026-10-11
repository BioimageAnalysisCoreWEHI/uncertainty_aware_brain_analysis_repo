import unittest

from amap_validation.config import DEFAULT_SEG_DIR, load_config
from amap_validation.segtable import brain_name, build_segtable, region_name, seg_idx, slice_num


class TestParsing(unittest.TestCase):
    def test_filename_fields(self):
        f = "Seg10MV_Ntsr1_169_sg.tif001-SG_Slice35..tif"
        self.assertEqual(seg_idx(f), 10)
        self.assertEqual(brain_name(f), "MV_Ntsr1_169")
        self.assertEqual(region_name(f), "SG")
        self.assertEqual(slice_num(f), 35)
        f = "Seg37AB_Gad67_223_rspV1V2M.tif001-RSP_Slice23..tif"
        self.assertEqual((brain_name(f), region_name(f), slice_num(f)), ("AB_Gad67_223", "RSP", 23))
        f = "Seg5MV131017_7_cingul.tif001-CING_Slice3..tif"
        self.assertEqual(brain_name(f), "MV131017_7")


class TestRealTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = build_segtable(DEFAULT_SEG_DIR)
        cls.cfg = load_config()

    def test_counts(self):
        self.assertEqual(len(self.t), 880)
        counts = self.t.groupby("brainRegionComb").size()
        self.assertEqual(len(counts), 60)
        for brc, n in counts.items():
            expected = 22 if brc.startswith(("MV_Ntsr1_165", "MV_Ntsr1_169")) else 11
            self.assertEqual(n, expected, brc)

    def test_names_match_config(self):
        self.assertEqual(set(self.t["brain"]), set(self.cfg.brains))
        self.assertEqual(set(self.t["region"]), {r.abbrev for r in self.cfg.regions})

    def test_attempts(self):
        rep = self.t[self.t["brain"].isin(["MV_Ntsr1_165", "MV_Ntsr1_169"])]
        self.assertEqual((rep["attempt"] == 1).sum(), (rep["attempt"] == 2).sum())
        self.assertTrue((self.t[~self.t.index.isin(rep.index)]["attempt"] == 1).all())


if __name__ == "__main__":
    unittest.main()
