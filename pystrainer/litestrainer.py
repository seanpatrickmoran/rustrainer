#!/usr/bin/env python3
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

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
        # keep same general triangular masking idea
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
        r1 = cx - target // 2
        r2 = cx + target // 2
        r3 = cy - target // 2
        r4 = cy + target // 2

        if r1 < 0:
            shift = -r1
            r1, r2 = 0, r2 + shift
        if r3 < 0:
            shift = -r3
            r3, r4 = 0, r4 + shift

        return r1, r2, r3, r4

    def _read_feature_lines(self, feature_path: str):
        with open(feature_path, "r") as f:
            first = f.readline()
            # same heuristic as your working script
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
        Yields: (numpyarr_blob, viewing_vmax, true_max, dimensions)
        """
        expected = (dimension + 1, dimension + 1)

        if hic_path.endswith(".hic"):
            hic = hicstraw.HiCFile(hic_path)

            last_pair = None
            matrix_obj = None

            for row_num, parts in self._read_feature_lines(feature_path):
                if len(parts) < 6:
                    continue
                c1, x1, x2, c2, y1, y2 = parts[:6]
                try:
                    x1, x2, y1, y2 = map(int, (x1, x2, y1, y2))
                except Exception:
                    continue

                # MATCH YOUR WORKING VERSION:
                # hicstraw wants chromosome IDs WITHOUT "chr" prefix (e.g., "1", "X")
                c1c = c1.lstrip("chr")
                c2c = c2.lstrip("chr")

                pair = (c1c, c2c)
                if pair != last_pair:
                    try:
                        matrix_obj = hic.getMatrixZoomData(
                            c1c, c2c, "observed", norm, "BP", resolution
                        )
                    except Exception as e:
                        # skip bad pairs instead of risking native crash later
                        print(f"[hic] zoomdata fail line {row_num}: {c1c},{c2c} ({e})")
                        matrix_obj = None
                    last_pair = pair

                if matrix_obj is None:
                    continue

                r1, r2, r3, r4 = self.windowing(x1, x2, y1, y2, resolution, dimension)
                if r1 < 0 or r3 < 0:
                    continue

                # hicstraw can sometimes crash if asked for nonsense ranges; keep it guarded
                try:
                    mat = matrix_obj.getRecordsAsMatrix(r1, r2, r3, r4)
                except Exception as e:
                    print(f"[hic] getRecordsAsMatrix fail line {row_num}: {e}")
                    continue

                # Enforce the exact shape your working script expects
                if mat.shape != expected:
                    continue

                mat = mat.astype(np.float32, copy=False)
                _, vmax = self.choose_vmax(self._downsample_view(mat))
                true_max = float(np.max(mat)) if float(np.max(mat)) > 0 else 1.0

                numpy_blob = mat.tobytes(order="C")
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

            for row_num, parts in self._read_feature_lines(feature_path):
                if len(parts) < 3:
                    continue

                try:
                    if len(parts) < 6:
                        c1, x1, y1 = parts[:3]
                        x1, y1 = int(x1), int(y1)
                        x2, y2, c2 = x1 + resolution, y1 + resolution, c1
                    else:
                        c1, x1, x2, c2, y1, y2 = parts[:6]
                        x1, x2, y1, y2 = map(int, (x1, x2, y1, y2))
                except Exception:
                    continue

                if not c1.startswith("chr"):
                    c1, c2 = f"chr{c1}", f"chr{c2}"

                pair = (c1, c2)
                if pair != last_pair:
                    try:
                        matrix_data = clr.matrix(balance=balance).fetch(c1, c2)
                    except Exception as e:
                        print(f"[cool] fetch fail line {row_num}: {c1},{c2} ({e})")
                        matrix_data = None
                    last_pair = pair

                if matrix_data is None:
                    continue

                r1, r2, r3, r4 = self.windowing(x1, x2, y1, y2, resolution, dimension)
                b = resolution

                try:
                    mat = matrix_data[
                        r1 // b : r1 // b + dimension + 1,
                        r3 // b : r3 // b + dimension + 1,
                    ]
                except Exception as e:
                    print(f"[cool] slice fail line {row_num}: {e}")
                    continue

                mat = np.nan_to_num(mat, nan=0.0).astype(np.float32, copy=False)
                if mat.shape != expected:
                    continue

                _, vmax = self.choose_vmax(self._downsample_view(mat))
                true_max = float(np.max(mat)) if float(np.max(mat)) > 0 else 1.0

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

    def run(self, output_db: str, limit: int = 1000, max_records: int = 0):
        """
        limit: batch size (commit every `limit` inserts)
        max_records: optional cap; 0 means no cap
        """
        self.create_database(output_db)
        conn = sqlite3.connect(output_db)
        cur = conn.cursor()

        batch: List[Tuple[int, bytes, float, float, int]] = []
        written = 0

        try:
            for ds in self.config.get("datasets", []):
                dims = self.validate_dimensions(ds.get("resolutions", DEFAULT_RESOLUTIONS))
                norm = ds.get("options", {}).get("norm", "NONE")

                for res, dim in dims.items():
                    for numpy_blob, vmax, true_max, dimension in self.process_hic_file(
                        ds["hic_path"], ds["feature_path"], res, dim, norm
                    ):
                        # optional hard cap if you ever want it (leave max_records=0 to disable)
                        if max_records and written >= max_records:
                            break

                        key_id = self.current_key_id
                        self.current_key_id += 1

                        # keep "try and just save" behavior:
                        try:
                            batch.append((key_id, numpy_blob, vmax, true_max, dimension))
                            written += 1
                        except Exception as e:
                            print(f"Skipping record key_id={key_id} (prep fail): {e}")
                            continue

                        # COMMIT cadence: every `limit` rows
                        if len(batch) >= limit:
                            try:
                                cur.executemany(
                                    "INSERT INTO imag_with_seqs (key_id, numpyarr, viewing_vmax, true_max, dimensions) "
                                    "VALUES (?,?,?,?,?)",
                                    batch,
                                )
                                conn.commit()
                                print(f"Wrote batch of {len(batch)} rows (total={written})")
                                batch.clear()
                            except Exception as e:
                                print(f"Batch write failed ({e}); trying individual inserts...")
                                # fallback: try individually
                                for row in batch:
                                    try:
                                        cur.execute(
                                            "INSERT INTO imag_with_seqs (key_id, numpyarr, viewing_vmax, true_max, dimensions) "
                                            "VALUES (?,?,?,?,?)",
                                            row,
                                        )
                                    except Exception as e2:
                                        print(f"  Failed key_id={row[0]}: {e2}")
                                conn.commit()
                                batch.clear()

                    if max_records and written >= max_records:
                        break
                if max_records and written >= max_records:
                    break

            # flush remainder
            if batch:
                try:
                    cur.executemany(
                        "INSERT INTO imag_with_seqs (key_id, numpyarr, viewing_vmax, true_max, dimensions) "
                        "VALUES (?,?,?,?,?)",
                        batch,
                    )
                    conn.commit()
                    print(f"Wrote final batch of {len(batch)} rows (total={written})")
                except Exception as e:
                    print(f"Final batch write failed ({e}); trying individual inserts...")
                    for row in batch:
                        try:
                            cur.execute(
                                "INSERT INTO imag_with_seqs (key_id, numpyarr, viewing_vmax, true_max, dimensions) "
                                "VALUES (?,?,?,?,?)",
                                row,
                            )
                        except Exception as e2:
                            print(f"  Failed key_id={row[0]}: {e2}")
                    conn.commit()
                batch.clear()

        finally:
            conn.close()

        print(f"Database saved to: {output_db}")
        print(f"Total records written (attempted adds): {written}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python lite_writer.py <config.json/yaml> [output.db] [batch_size]")
        raise SystemExit(1)

    config_path = sys.argv[1]
    output_db = sys.argv[2] if len(sys.argv) > 2 else "output.db"
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    LiteHiCWriter(config_path).run(output_db, limit=batch_size, max_records=0)


if __name__ == "__main__":
    main()

