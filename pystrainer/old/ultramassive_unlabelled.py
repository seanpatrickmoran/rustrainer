#!/usr/bin/env python3

import sqlite3
from pathlib import Path
import json
import datetime
from typing import List, Dict, Tuple, Optional

import rstrainer


def call(path: str, timeout: float):
    conn = sqlite3.connect(path, timeout=timeout)
    cur = conn.cursor()
    return conn, cur

def collate_unlabeled_data(
    cursor_s,
    limit: int,
    offset: int,
    image_path: str,
    dataset_store: List[Dict],
    running_key_id: int,
    resolution_filter: str = "5000",
    max_n_ratio: float = 0.15,
    min_seq_length: int = 100,
) -> Tuple[List[Dict], int]:

    resolution_filter = str(resolution_filter)

    skipped_n_content = 0
    skipped_short = 0
    skipped_invalid = 0
    skipped_image = 0
    processed = 0

    cursor_s.execute(
        """
        SELECT key_id, resolution, viewing_vmax, numpyarr, seqA, seqB, dimensions, labels
        FROM imag_with_seqs
        WHERE (labels IS NULL OR labels = '') AND resolution = ?
        LIMIT ? OFFSET ?
        """,
        (resolution_filter, limit, offset),
    )

    rows = cursor_s.fetchall()
    for en in rows:
        try:
            original_key_id = en[0]
            resolution = en[1]
            viewing_vmax = float(en[2]) if en[2] is not None else float("nan")
            numpyarr = en[3]
            seqA = en[4] or ""
            seqB = en[5] or ""
            dim_hint = int(en[6]) if en[6] is not None else 0

            if numpyarr is None or dim_hint <= 0:
                skipped_invalid += 1
                continue

            ok, cleanedA, cleanedB, nra, nrb = rstrainer.clean_pair_and_n_ratio(seqA, seqB)
            if not ok:
                skipped_invalid += 1
                continue

            # Apply same filtering logic as your Python version
            if nra > max_n_ratio or (seqB and nrb > max_n_ratio):
                skipped_n_content += 1
                continue

            if len(cleanedA) < min_seq_length:
                skipped_short += 1
                continue

            running_key_id += 1
            out_name = f"{image_path}/unlabeled_{running_key_id}_image.png"

            ok_img, used_dim = rstrainer.make_png_from_f32_blob(
                numpyarr, int(dim_hint), float(viewing_vmax), out_name
            )
            if not ok_img:
                skipped_image += 1
                continue

            dataset_store.append(
                {
                    "key_id": running_key_id,
                    "original_key_id": original_key_id,
                    "resolution": resolution,
                    "image": out_name,
                    "sequenceA": cleanedA,
                    "sequenceB": cleanedB,
                    "predicted_label": "unlabeled",
                    "dimensions": int(used_dim),
                }
            )
            processed += 1

        except Exception as e:
            print(f"Error processing record {en[0]}: {e}")
            skipped_invalid += 1

    total_attempted = processed + skipped_n_content + skipped_short + skipped_invalid + skipped_image
    print(
        f"  Batch: {processed} processed | Skipped: {skipped_n_content} high-N, "
        f"{skipped_short} short, {skipped_invalid} invalid, {skipped_image} image "
        f"(total: {total_attempted})"
    )

    return dataset_store, running_key_id


def extract_unlabeled_data() -> str:
    dbSOURCE = [
        ("/mnt/data/vitc_data/SourceSummer/SEQ_base_database_36_bin.db", 199627),
        ("/mnt/data/vitc_data/SourceSummer/SEQ_negSynth_database_36_bin.db", 152827),
        ("/home/sean/git/strainer/vitc_preproc/harris2.db", 124161),
        ("/home/sean/git/strainer/vitc_preproc/harris_synNeg.db", 47358),
        ("/home/sean/git/strainer/vitc_preproc/scale_harris.db", 67849),
        # ("/home/sean/git/strainer/vitc_preproc/pancancer.db", 128101),
        # ("/home/sean/git/strainer/vitc_preproc/SynthNeg_pancancer.db", 118723),
    ]

    config = {
        "limit": 10240,
        "image_path": "/home/sean/git/proj_vitc/TwoModal_DDP/data/ultramassive_unlabelled_images_5k",
        "resolution_filter": "5000",
        "output_path": "/home/sean/git/proj_vitc/TwoModal_DDP/data/120825_ultramassive_unlabeled_data.json",
        "max_n_ratio": 0.15,
        "min_seq_length": 100,
        "timeout": 5.0,
    }

    Path(config["image_path"]).mkdir(parents=True, exist_ok=True)
    Path(config["output_path"]).parent.mkdir(parents=True, exist_ok=True)

    dataset_store: List[Dict] = []
    running_key_id = 0

    for db_index, (db_path, db_limit) in enumerate(dbSOURCE):
        conn = None
        cur = None
        try:
            print(f"\n{'='*60}")
            print(f"Processing UNLABELED data from database {db_index + 1}/{len(dbSOURCE)}")
            print(f"Path: {db_path}")
            print(f"{'='*60}")

            conn, cur = call(db_path, config["timeout"])

            cur.execute("PRAGMA journal_mode=OFF;")
            cur.execute("PRAGMA synchronous=OFF;")
            cur.execute("PRAGMA temp_store=MEMORY;")

            cur.execute(
                """
                SELECT COUNT(*) FROM imag_with_seqs
                WHERE (labels IS NULL OR labels = '') AND resolution = ?
                """,
                (config["resolution_filter"],),
            )
            unlabeled_count = cur.fetchone()[0]
            print(f"Found {unlabeled_count} unlabeled samples in this database")

            actual_limit = min(db_limit, unlabeled_count)
            print(f"Will process up to {actual_limit} samples")

            offset = 0
            while offset < actual_limit:
                batch_start = datetime.datetime.now()

                dataset_store, running_key_id = collate_unlabeled_data(
                    cursor_s=cur,
                    limit=config["limit"],
                    offset=offset,
                    image_path=config["image_path"],
                    dataset_store=dataset_store,
                    running_key_id=running_key_id,
                    resolution_filter=config["resolution_filter"],
                    max_n_ratio=config["max_n_ratio"],
                    min_seq_length=config["min_seq_length"],
                )

                offset += config["limit"]

                batch_time = datetime.datetime.now() - batch_start
                print(
                    f"  Progress: {min(offset, actual_limit)}/{actual_limit} | "
                    f"Total collected: {len(dataset_store)} | "
                    f"Batch time: {batch_time}"
                )

        except Exception as e:
            print(f"Error processing database {db_path}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                conn.close()

    print(f"\n{'='*60}")
    print(f"Saving {len(dataset_store)} unlabeled records...")
    print(f"Output: {config['output_path']}")
    print(f"{'='*60}")

    with open(config["output_path"], "w") as f:
        json.dump(dataset_store, f, indent=2)

    print("✓ Unlabeled data saved successfully!")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total records: {len(dataset_store)}")
    print(f"Image directory: {config['image_path']}")
    print(f"JSON output: {config['output_path']}")
    print("\nCleaning settings:")
    print(f"  Max N ratio: {config['max_n_ratio']*100:.0f}%")
    print(f"  Min sequence length: {config['min_seq_length']} bp")
    print(f"  Resolution filter: {config['resolution_filter']}")

    return config["output_path"]

def count_unlabeled_samples(db_path: str, resolution: str = "5000") -> int:
    conn, cur = call(db_path, 5.0)
    try:
        cur.execute(
            """
            SELECT COUNT(*) FROM imag_with_seqs
            WHERE (labels IS NULL OR labels = '') AND resolution = ?
            """,
            (str(resolution),),
        )
        return cur.fetchone()[0]
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    now = datetime.datetime.now()
    unlabeled_json_path = extract_unlabeled_data()

    print(f"\n{'='*60}")
    print(f"Extraction completed in: {datetime.datetime.now() - now}")
    print(f"{'='*60}")
    print("\nYou can now use:")
    print(f"  --data_json {unlabeled_json_path}")
