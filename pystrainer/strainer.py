#!/usr/bin/env python3
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import cooler
import hicstraw
import numpy as np
import py2bit
import yaml
from skimage.filters import threshold_otsu
from skimage.transform import hough_line, hough_line_peaks

from rstrainer import serialize_window_and_hists

DEFAULT_RESOLUTIONS = [ 5000, 10000]
DIMENSION_MAP = {2000: 162, 5000: 64, 10000: 32}


class UnifiedHiCPipeline:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.current_key_id = 1
        self.feature_mapping: Dict[int, str] = {}
        self._twobit_cache: Dict[str, Any] = {}


    def load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            if self.config_path.suffix in (".yaml", ".yml"):
                return yaml.safe_load(f)
            return json.load(f)

    def validate_dimensions(self, resolutions: List[int]) -> Dict[int, int]:
        dimensions: Dict[int, int] = {}
        for res in resolutions:
            dim = DIMENSION_MAP.get(res, int(326000 / res))
            if dim > 200:
                response = input(
                    f"Warning: Resolution {res} will create {dim}x{dim} images. Continue? [Y/n]: "
                )
                if response.lower() == "n":
                    raise ValueError("User cancelled due to large image dimensions")
            dimensions[res] = dim
        return dimensions

    def _downsample_view(self, mat: np.ndarray, target: int = 64) -> np.ndarray:
        n = mat.shape[0]
        if n <= target:
            return mat
        step = max(1, n // target)
        return mat[::step, ::step]


    def choose_vmax(self, nimage: np.ndarray) -> Tuple[np.ndarray, float]:
        thresh = threshold_otsu(nimage)
        binary = nimage > thresh
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = hough_line(binary, theta=tested_angles)

        c_angle, c_dist = 0, 0
        chosen = 10

        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            slope = np.tan(angle + np.pi / 2)
            if abs(1 - abs(slope)) < abs(1 - abs(chosen)):
                c_angle = angle
                c_dist = dist
                chosen = slope

        slope = np.tan(c_angle + np.pi / 2)
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
    ) -> Iterator[Tuple[int, Dict[str, Any]]]:

        if hic_path.endswith(".hic"):
            hic = hicstraw.HiCFile(hic_path)
            last_pair = None
            matrix_obj = None

            for row_num, parts in self._read_feature_lines(feature_path):
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
                np_mat = matrix_obj.getRecordsAsMatrix(r1, r2, r3, r4)

                if (np_mat.shape != (dimension + 1, dimension + 1) or np_mat.shape != (dimension , dimension)):
                    continue

                np_mat = np_mat.astype(np.float32, copy=False)
                _, vmax = self.choose_vmax(self._downsample_view(np_mat))
                true_max = float(np.max(np_mat)) or 1.0

                numpy_bytes, hrel, htrue = serialize_window_and_hists(np_mat, vmax, true_max)

                key_id = self.current_key_id
                self.current_key_id += 1
                feature_key = f"{feature_path}:{row_num}"
                self.feature_mapping[key_id] = feature_key

                yield key_id, {
                    "coordinates": [c1, x1, x2, c2, y1, y2],
                    "numpy_bytes": numpy_bytes,
                    "viewing_vmax": vmax,
                    "true_max": true_max,
                    "hist_rel_bytes": hrel,
                    "hist_true_bytes": htrue,
                    "feature_source": feature_key,
                }

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

                numpy_bytes, hrel, htrue = serialize_window_and_hists(mat, vmax, true_max)

                key_id = self.current_key_id
                self.current_key_id += 1
                feature_key = f"{feature_path}:{row_num}"
                self.feature_mapping[key_id] = feature_key

                yield key_id, {
                    "coordinates": [c1, x1, x2, c2, y1, y2],
                    "numpy_bytes": numpy_bytes,
                    "viewing_vmax": vmax,
                    "true_max": true_max,
                    "hist_rel_bytes": hrel,
                    "hist_true_bytes": htrue,
                    "feature_source": feature_key,
                }

    def extract_sequences(self, coords: List[Any], genome: str) -> Tuple[str, str]:
        genome_path = self.config.get(f"{genome.upper()}_PATH")
        if not genome_path or not os.path.exists(genome_path):
            return "", ""

        tb = self._twobit_cache.setdefault(genome_path, py2bit.open(genome_path))
        try:
            c1, x1, x2, c2, y1, y2 = coords
            return tb.sequence(c1, x1, x2), tb.sequence(c2, y1, y2)
        except Exception:
            return "", ""

    def create_database(self, output_path: str):
        conn = sqlite3.connect(output_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS imag_with_seqs (
                key_id INTEGER PRIMARY KEY,
                name TEXT,
                dataset TEXT,
                condition TEXT,
                coordinates TEXT,
                numpyarr BLOB,
                viewing_vmax REAL,
                true_max REAL,
                hist_rel BLOB,
                hist_true BLOB,
                dimensions INTEGER,
                hic_path TEXT,
                resolution INTEGER,
                norm TEXT,
                seqA TEXT,
                seqB TEXT,
                toolsource TEXT,
                featuretype TEXT,
                labels TEXT DEFAULT '',
                meta TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_mapping (
                key_id INTEGER PRIMARY KEY,
                source_file TEXT,
                source_row INTEGER
            )
            """
        )
        conn.commit()
        conn.close()

    def run(self, output_db: str):
        print("Starting unified Hi-C image pipeline.")  # :contentReference[oaicite:2]{index=2}
        print(f"Config has {len(self.config.get('datasets', []))} datasets")
        for dataset in self.config.get("datasets", []):
            print(f"  Dataset: {dataset.get('name', 'UNKNOWN')}")
            print(f"    Hi-C: {dataset.get('hic_path', 'MISSING')}")
            print(f"    Features: {dataset.get('feature_path', 'MISSING')}")

        self.create_database(output_db)
        conn = sqlite3.connect(output_db)
        cursor = conn.cursor()

        batch_rows: List[Tuple[Any, ...]] = []
        batch_maps: List[Tuple[Any, ...]] = []
        BATCH = 1000

        def flush_with_logging():
            nonlocal batch_rows, batch_maps
            if not batch_rows:
                return
            try:
                cursor.executemany(
                    "INSERT INTO imag_with_seqs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    batch_rows,
                )
                cursor.executemany(
                    "INSERT INTO feature_mapping VALUES (?,?,?)",
                    batch_maps,
                )
                conn.commit()
                print(f"    Wrote batch of {len(batch_rows)} records")  # :contentReference[oaicite:3]{index=3}
            except Exception as e:
                print(f"    Error writing batch: {e}")
                for record in batch_rows:
                    try:
                        cursor.execute(
                            "INSERT INTO imag_with_seqs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            record,
                        )
                    except Exception as e2:
                        print(f"      Failed to write record {record[0]}: {e2}")
                # Feature mappings are best-effort in fallback path too
                for fmap in batch_maps:
                    try:
                        cursor.execute(
                            "INSERT INTO feature_mapping VALUES (?,?,?)",
                            fmap,
                        )
                    except Exception:
                        pass
                conn.commit()
            finally:
                batch_rows.clear()
                batch_maps.clear()

        try:
            for ds in self.config["datasets"]:
                print(f"\nProcessing dataset: {ds.get('name', 'UNKNOWN')}")  # :contentReference[oaicite:5]{index=5}
                dims = self.validate_dimensions(ds.get("resolutions", DEFAULT_RESOLUTIONS))

                for res, dim in dims.items():
                    print(f"  Processing resolution: {res}bp")  # :contentReference[oaicite:6]{index=6}
                    norm = ds.get("options", {}).get("norm", "NONE")

                    for key_id, rec in self.process_hic_file(
                        ds["hic_path"],
                        ds["feature_path"],
                        res,
                        dim,
                        norm,
                    ):
                        coords = rec["coordinates"]
                        coord_str = ",".join(map(str, coords))
                        seqA, seqB = self.extract_sequences(coords, ds.get("genome", "hg38"))

                        batch_rows.append(
                            (
                                key_id,
                                f"{ds['name']}_{res}_{key_id}",
                                ds.get("dataset", ds["name"]),
                                ds.get("condition", ""),
                                coord_str,
                                rec["numpy_bytes"],
                                rec["viewing_vmax"],
                                rec["true_max"],
                                rec["hist_rel_bytes"],
                                rec["hist_true_bytes"],
                                dim,
                                ds["hic_path"],
                                res,
                                norm,
                                seqA,
                                seqB,
                                ds.get("options", {}).get("toolsource", "unknown"),
                                ds.get("options", {}).get("featuretype", "unknown"),
                                "",
                                json.dumps({"feature_source": rec["feature_source"]}),
                            )
                        )

                        src, row = rec["feature_source"].rsplit(":", 1)
                        batch_maps.append((key_id, src, int(row)))

                        if len(batch_rows) >= BATCH:
                            flush_with_logging()

            if batch_rows:
                try:
                    cursor.executemany(
                        "INSERT INTO imag_with_seqs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        batch_rows,
                    )
                    cursor.executemany(
                        "INSERT INTO feature_mapping VALUES (?,?,?)",
                        batch_maps,
                    )
                    conn.commit()
                    print(f"  Wrote final batch of {len(batch_rows)} records")  # :contentReference[oaicite:7]{index=7}
                except Exception as e:
                    print(f"  Error writing final batch: {e}")  # :contentReference[oaicite:8]{index=8}
                    for record in batch_rows:
                        try:
                            cursor.execute(
                                "INSERT INTO imag_with_seqs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                record,
                            )
                        except Exception as e2:
                            print(f"    Failed to write record {record[0]}: {e2}")
                    conn.commit()

        finally:
            conn.close()
            for tb in self._twobit_cache.values():
                try:
                    tb.close()
                except Exception:
                    pass

        mapping_path = Path(output_db).with_suffix(".mapping.json")
        with open(mapping_path, "w") as f:
            json.dump(self.feature_mapping, f, indent=2)

        print("\nPipeline complete!")  # :contentReference[oaicite:9]{index=9}
        print(f"Database saved to: {output_db}")
        print(f"Feature mapping saved to: {mapping_path}")
        print(f"Total images processed: {self.current_key_id - 1}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python strainer.py <config.json/yaml> [output.db]")
        sys.exit(1)

    config_path = sys.argv[1]
    output_db = sys.argv[2] if len(sys.argv) > 2 else "output.db"
    UnifiedHiCPipeline(config_path).run(output_db)


if __name__ == "__main__":
    main()
