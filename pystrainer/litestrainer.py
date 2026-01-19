#!/usr/bin/env python3
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import cooler
import hicstraw
import numpy as np
import yaml
from skimage.filters import threshold_otsu
from skimage.transform import hough_line, hough_line_peaks

DEFAULT_RESOLUTIONS = [5000, 10000]
DIMENSION_MAP = {2000: 162, 5000: 64, 10000: 32}


class LiteHiCWriter:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.current_key_id = 1

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            if self.config_path.suffix in (".yaml", ".yml"):
                return yaml.safe_load(f)
            return json.load(f)

    def validate_dimensions(self, resolutions: List[int]) -> Dict[int, int]:
        dims: Dict[int, int] = {}
        for res in resolutions:
            dim = DIMENSION_MAP.get(res, int(326000 / res))
            if dim > 200:
                resp = input(
                    f"Warning: Resolution {res} will create {dim}x{dim} images. Continue? [Y/n]: "
                )
                if resp.lower() == "n":
                    raise ValueError("User cancelled due to large image dimensions")
            dims[res] = dim
        return dims

    def _downsample_view(self, mat: np.ndarray, target: int = 64) -> np.ndarray:
        n = mat.shape[0]
        if n <= target:
            return mat
        step = max(1, n // target)
        return mat[::step, ::step]

    def choose_vmax(self, nimage: np.ndarray) -> Tuple[np.ndarray, float]:
        # same logic as your original
        thresh = threshold_otsu(nimage)
        binary = nimage > thresh
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = hough_line(binary, theta=tested_angles)

        c_angle, c_dist = 0.0, 0.0
        chosen = 10.0

        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            slope = np.tan(angle + np.pi / 2)
            if abs(1 - abs(slope)) < abs(1 - abs(chosen)):
                c_angle = float(angle)
                c_dist = float(dist)
                chosen = float(slope)

        slope = float(np.tan(c_angle + np.pi / 2))
        n = nimage.shape[0]

        if slope > 10:
            vmax = float(np.max(nimage))
            return nimage, vmax if vmax > 0 else 1.0

        mimage = nimage.copy()
        if int(np.round(c_dist * (np.sin(c_angle) - np.cos(c_angle)))) <= n // 10:
            mimage = np.triu(mimage, k=n // 10 + 2)
        else:
            mimage = np.triu(mimage, k=-n // 10)

        vmax = float(np.max(mimage))
        return mimage, vmax if vmax > 0 else 1.0

    def windowing(
        self, x1: int, x2: int, y1: int, y2: int, res: int, width: int
    ) -> Tuple[int, int, int, int]:
        target = res * width
        if (x2 - x1) >= target and (y2 - y1) >= target:
            return x1, x2, y1, y2

        cx = x1 + (x2 - x1) // 2
        cy = y1 + (y2 - y1) // 2
        r1 = max(0, cx - target // 2)
        r2 = r1 + target
        r3 = max(0, cy - target // 2)
        r4 = r3 + target
        return r1, r2, r3, r4

    def _read_feature_lines(self, feature_path: str):
        with open(feature_path, "r") as f:
            first = f.readline()
            if not (first.startswith("#") or "chr" in first.lower()):
                f.seek(0)
            for row_num, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    yield row_num, line.split("\t")

    def process_hic_file(
        self,
        hic_path: str,
        feature_path: str,
        resolution: int,
        dimension: int,
        norm: str,
    ) -> Iterator[Tuple[bytes, float, float, int]]:
        """
        Yields tuples: (numpyarr_blob, viewing_vmax, true_max, dimensions)
        """
        if hic_path.endswith(".hic"):
            hic = hicstraw.HiCFile(hic_path)
            last_pair = None
            matrix_obj = None

            for _, parts in self._read_feature_lines(feature_path):
                if len(parts) < 6:
                    continue
                c1, x1, x2, c2, y1, y2 = parts[:6]
                x1, x2, y1, y2 = map(int, (x1, x2, y1, y2))

                c1c, c2c = c1.lstrip("chr"), c2.lstrip("chr")
                pair = (c1c, c2c)
                if pair != last_pair:
                    matrix_obj = hic.getMatrixZoomData(
                        f"chr{c1c}", f"chr{c2c}", "observed", norm, "BP", resolution
                    )
                    last_pair = pair

                r1, r2, r3, r4 = self.windowing(x1, x2, y1, y2, resolution, dimension)
                mat = matrix_obj.getRecordsAsMatrix(r1, r2, r3, r4)

                # keep your original acceptance logic, but fix the condition
                if mat.shape not in {(dimension, dimension), (dimension + 1, dimension + 1)}:
                    continue

                mat = mat.astype(np.float32, copy=False)
                _, vmax = self.choose_vmax(self._downsample_view(mat))
                true_max = float(np.max(mat)) or 1.0

                # store raw numpy bytes (fast, compact, no metadata beyond shape/dtype)
                numpy_blob, _, _ = serialize_window_and_hists(mat, vmax, true_max)
                yield numpy_blob, float(vmax), float(true_max), int(dimension)

        elif hic_path.endswith((".cool", ".mcool")):
            clr = cooler.Cooler(
                f"{hic_path}::/resolutions/{resolution}"
                if hic_path.endswith(".mcool")
                else hic_path
            )
            balance = norm in ("KR", "VC", "VC_SQRT")
            last_pair = None
            matrix_data = None

            for _, parts in self._read_feature_lines(feature_path):
                if len(parts) < 3:
                    continue

                if len(parts) < 6:
                    c1, x1, y1 = parts[:3]
                    x1, y1 = int(x1), int(y1)
                    x2, y2, c2 = x1 + resolution, y1 + resolution, c1
                else:
                    c1, x1, x2, c2, y1, y2 = parts[:6]
                    x1, x2, y1, y2 = map(int, (x1, x2, y1, y2))

                if not c1.startswith("chr"):
                    c1, c2 = f"chr{c1}", f"chr{c2}"

                pair = (c1, c2)
                if pair != last_pair:
                    matrix_data = clr.matrix(balance=balance).fetch(c1, c2)
                    last_pair = pair

                r1, r2, r3, r4 = self.windowing(x1, x2, y1, y2, resolution, dimension)
                b = resolution
                mat = matrix_data[
                    r1 // b : r1 // b + dimension + 1,
                    r3 // b : r3 // b + dimension + 1,
                ]
                mat = np.nan_to_num(mat, nan=0.0).astype(np.float32, copy=False)

                if mat.shape != (dimension + 1, dimension + 1):
                    continue

                _, vmax = self.choose_vmax(self._downsample_view(mat))
                true_max = float(np.max(mat)) or 1.0

                numpy_blob = mat.tobytes(order="C")
                yield numpy_blob, float(vmax), float(true_max), int(dimension)
        else:
            raise ValueError(f"Unsupported Hi-C format: {hic_path}")

    def create_database(self, output_path: str):
        conn = sqlite3.connect(output_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS imag_with_seqs (
                key_id INTEGER PRIMARY KEY,
                numpyarr BLOB,
                viewing_vmax REAL,
                true_max REAL,
                dimensions INTEGER
            )
            """
        )
        conn.commit()
        conn.close()

    def run(self, output_db: str):
        self.create_database(output_db)
        conn = sqlite3.connect(output_db)
        cur = conn.cursor()

        BATCH = 1000
        batch: List[Tuple[int, bytes, float, float, int]] = []

        try:
            for ds in self.config.get("datasets", []):
                dims = self.validate_dimensions(ds.get("resolutions", DEFAULT_RESOLUTIONS))
                norm = ds.get("options", {}).get("norm", "NONE")

                for res, dim in dims.items():
                    for numpy_blob, vmax, true_max, dimension in self.process_hic_file(
                        ds["hic_path"],
                        ds["feature_path"],
                        res,
                        dim,
                        norm,
                    ):
                        key_id = self.current_key_id
                        self.current_key_id += 1

                        batch.append((key_id, numpy_blob, vmax, true_max, dimension))

                        if len(batch) >= BATCH:
                            cur.executemany(
                                "INSERT INTO imag_with_seqs (key_id, numpyarr, viewing_vmax, true_max, dimensions) "
                                "VALUES (?,?,?,?,?)",
                                batch,
                            )
                            conn.commit()
                            batch.clear()

            if batch:
                cur.executemany(
                    "INSERT INTO imag_with_seqs (key_id, numpyarr, viewing_vmax, true_max, dimensions) "
                    "VALUES (?,?,?,?,?)",
                    batch,
                )
                conn.commit()
                batch.clear()
        finally:
            conn.close()

        print(f"Database saved to: {output_db}")
        print(f"Total records written: {self.current_key_id - 1}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python lite_writer.py <config.json/yaml> [output.db]")
        raise SystemExit(1)

    config_path = sys.argv[1]
    output_db = sys.argv[2] if len(sys.argv) > 2 else "output.db"
    LiteHiCWriter(config_path).run(output_db)


if __name__ == "__main__":
    main()
