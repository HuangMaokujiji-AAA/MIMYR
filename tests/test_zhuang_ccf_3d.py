"""Engineering checks for ID joins, metadata loading, extraction, and HTML output."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.zhuang_ccf_3d import extract_archive, join_ccf, load_ccf_cells, visualization_sample, write_3d_html


class ZhuangCCFTests(unittest.TestCase):
    def test_string_join_backed_loading_and_plot(self):
        ids = ["212059428959805613780123472670405935608", "101833224400913170993812156276156584394", "unregistered"]
        obs = pd.DataFrame({"class": ["Astro", "Glut", "Glut"],
                            "class_color": ["#112233", "#445566", "#445566"]}, index=ids)
        ccf = pd.DataFrame({"cell_label": ids[:2][::-1], "x": [2., 1.], "y": [4., 3.], "z": [6., 5.]})
        table, audit = join_ccf(obs, ccf.set_index("cell_label"), "sec_001")
        self.assertEqual(table["cell_id"].tolist(), ids[:2])
        self.assertEqual(table["x_ccf"].tolist(), [1., 2.])
        self.assertEqual(audit["n_without_ccf"], 1)
        with tempfile.TemporaryDirectory(prefix="mimyr-ccf-test-", dir=Path(__file__).resolve().parents[1] / "tmp") as temporary:
            root = Path(temporary)
            (root / "Zhuang-ABCA-2").mkdir()
            a = ad.AnnData(X=np.ones((3, 2), dtype=np.float32), obs=obs)
            a.write_h5ad(root / "Zhuang-ABCA-2/sec_001.h5ad")
            ccf.to_csv(root / "ccf-ABCA-2.csv", index=False)
            cells, audits = load_ccf_cells(root, 2)
            self.assertEqual(cells["cell_id"].tolist(), ids[:2])
            self.assertEqual(int(audits.iloc[0]["n_ccf"]), 2)
            pd.testing.assert_frame_equal(visualization_sample(cells, 1), visualization_sample(cells, 1))
            html = write_3d_html(cells, root / "test.html", total_cells=2).read_text(encoding="utf-8")
            self.assertIn('CCF x (mm)', html)
            self.assertIn('Plotly.react', html)
            self.assertIn('切片 / Section', html)
            self.assertIn('细胞类型 / Class', html)
            self.assertIn(ids[0], html)
            self.assertNotIn('src="https://cdn.plot.ly', html)
            # The embedded payload is independently parseable and preserves the mm values.
            payload = html.split('const d = ', 1)[1].split(';\n', 1)[0]
            self.assertEqual(json.loads(payload)["x"], [1., 2.])

    def test_reject_duplicate_ids_and_unsafe_zip(self):
        obs = pd.DataFrame(index=["a"])
        ccf = pd.DataFrame({"x": [1., 2.], "y": [1., 2.], "z": [1., 2.]}, index=["a", "a"])
        with self.assertRaises(ValueError):
            join_ccf(obs, ccf, "sec_001")
        with tempfile.TemporaryDirectory(prefix="mimyr-zip-test-", dir=Path(__file__).resolve().parents[1] / "tmp") as temporary:
            root = Path(temporary)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../outside.txt", "not a valid dataset")
            with self.assertRaises(ValueError):
                extract_archive(archive, root / "data")
            self.assertFalse((root / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
