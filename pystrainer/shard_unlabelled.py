#!/usr/bin/env python3

import os
import sqlite3
from pathlib import Path
import json
import datetime
from typing import Optional, Tuple

import rstrainer  #pyo3 

OUT_C = 3
OUT_H = 224
OUT_W = 224
REC_BYTES = OUT_C * OUT_H * OUT_W  # uint8 CHW


def call(path: str, timeout: float) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
    conn = sqlite3.connect(path, timeout=timeout)
    cur = conn.cursor()
    return conn, cur


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def export_unlabeled_shard_for_db(
    db_path: str,
    out_dir: str,
    db_limit: int,
    resolution_filter: str = "5000",
    limit: int = 10240,
    timeout: float = 5.0,
    max_n_ratio: float = 0.15,
    min_seq_length: int = 100,
) -> None:
    """
      - reads unlabeled rows at given resolution
      - writes fixed-size uint8 CHW records into a .bin shard (append)
      - append write JSONL metadata
    """
    resolution_filter = str(resolution_filter)

    out_dir = str(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    shard_bin = os.path.join(out_dir, f"unlabeled_res{resolution_filter}.bin")
    shard_jsonl = os.path.join(out_dir, f"unlabeled_res{resolution_filter}.jsonl")

    conn = None
    cur = None

    skipped_n_content = 0
    skipped_short = 0
    skipped_invalid = 0
    skipped_render = 0
    written = 0
    attempted = 0

    try:
        print(f"\nDB: {db_path}")
        print(f"Resolution: {resolution_filter}")
        print(f"Shard BIN: {shard_bin}")
        print(f"Shard JSONL: {shard_jsonl}")
        conn, cur = call(db_path, timeout)
        cur.execute("PRAGMA journal_mode=OFF;")
        cur.execute("PRAGMA synchronous=OFF;")
        cur.execute("PRAGMA temp_store=MEMORY;")

        cur.execute(
            """
            SELECT COUNT(*) FROM imag_with_seqs
            WHERE (labels IS NULL OR labels = '') AND resolution = ?
            """,
            (resolution_filter,),
        )
        unlabeled_count = cur.fetchone()[0]
        actual_limit = min(int(db_limit), int(unlabeled_count))
        print(f"Found {unlabeled_count} unlabeled; will process up to {actual_limit}")

        Path(shard_bin).parent.mkdir(parents=True, exist_ok=True)
        bin_f = open(shard_bin, "ab", buffering=1024 * 1024)
        jsonl_f = open(shard_jsonl, "a", buffering=1024 * 1024)
        bin_f.seek(0, os.SEEK_END)
        base_offset = bin_f.tell()

        offset = 0
        while offset < actual_limit:
            batch_start = datetime.datetime.now()

            cur.execute(
                """
                SELECT key_id, resolution, viewing_vmax, numpyarr, seqA, seqB, dimensions
                FROM imag_with_seqs
                WHERE (labels IS NULL OR labels = '') AND resolution = ?
                LIMIT ? OFFSET ?
                """,
                (resolution_filter, limit, offset),
            )
            rows = cur.fetchall()
            if not rows:
                break

            for en in rows:
                attempted += 1
                try:
                    original_key_id = en[0]
                    resolution = en[1]
                    viewing_vmax = float(en[2]) if en[2] is not None else float("nan")
                    numpyarr = en[3]
                    seqA = en[4] or ""
                    seqB = en[5] or ""
                    dim_hint = _safe_int(en[6], 0)

                    if numpyarr is None or dim_hint <= 0 or (not (viewing_vmax == viewing_vmax)):  # NaN check
                        skipped_invalid += 1
                        continue

                    ok, cleanedA, cleanedB, nra, nrb = rstrainer.clean_pair_and_n_ratio(seqA, seqB)
                    if not ok:
                        skipped_invalid += 1
                        continue

                    if nra > max_n_ratio or (seqB and nrb > max_n_ratio):
                        skipped_n_content += 1
                        continue

                    if len(cleanedA) < min_seq_length:
                        skipped_short += 1
                        continue

                    rgb_bytes, used_dim = rstrainer.render_rgb_u8_chw_from_f32_blob(
                        numpyarr, int(dim_hint), float(viewing_vmax), OUT_H, OUT_W
                    )

                    if rgb_bytes is None or len(rgb_bytes) != REC_BYTES:
                        skipped_render += 1
                        continue

                    rec_offset = base_offset + (written * REC_BYTES)
                    bin_f.write(rgb_bytes)

                    meta = {
                        "original_key_id": int(original_key_id),
                        "resolution": str(resolution),
                        "offset": int(rec_offset),
                        "length": int(REC_BYTES),
                        "shape": [OUT_C, OUT_H, OUT_W],  # CHW
                        "dtype": "uint8",
                        "dimensions_used": int(used_dim),
                        "sequenceA": cleanedA,
                        "sequenceB": cleanedB,
                        "predicted_label": "unlabeled",
                        "source_db": db_path,
                    }
                    jsonl_f.write(json.dumps(meta) + "\n")

                    written += 1

                except Exception as e:
                    skipped_invalid += 1
                    print(f"  row error (key_id={en[0]}): {e}")

            offset += limit

            batch_time = datetime.datetime.now() - batch_start
            total_skipped = skipped_n_content + skipped_short + skipped_invalid + skipped_render
            print(
                f"  Batch done: offset={min(offset, actual_limit)}/{actual_limit} | "
                f"written={written} | skipped={total_skipped} | "
                f"time={batch_time}"
            )

        bin_f.flush()
        jsonl_f.flush()
        bin_f.close()
        jsonl_f.close()

        print("\nDB summary:")
        print(f"  attempted: {attempted}")
        print(f"  written:   {written}")
        print(f"  skipped_n: {skipped_n_content}")
        print(f"  skipped_short: {skipped_short}")
        print(f"  skipped_invalid: {skipped_invalid}")
        print(f"  skipped_render: {skipped_render}")

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


def main():
    #change to user
    dbSOURCE = [
        ("/mnt/data/vitc_data/SourceSummer/SEQ_base_database_36_bin.db", 199627),
        ("/mnt/data/vitc_data/SourceSummer/SEQ_negSynth_database_36_bin.db", 152827),
        ("/home/sean/git/strainer/vitc_preproc/harris2.db", 124161),
        ("/home/sean/git/strainer/vitc_preproc/harris_synNeg.db", 47358),
        ("/home/sean/git/strainer/vitc_preproc/scale_harris.db", 67849),
    ]

    config = {
        "limit": 10240,
        "resolution_filter": "5000",
        "out_dir": "/home/sean/git/proj_vitc/TwoModal_DDP/data/shards_unlabeled_5k_res",
        "max_n_ratio": 0.15,
        "min_seq_length": 100,
        "timeout": 5.0,
    }

    Path(config["out_dir"]).mkdir(parents=True, exist_ok=True)

    start = datetime.datetime.now()
    for i, (db_path, db_limit) in enumerate(dbSOURCE):
        print(f"\n{'='*70}")
        print(f"DB {i+1}/{len(dbSOURCE)}")
        print(f"{'='*70}")

        export_unlabeled_shard_for_db(
            db_path=db_path,
            out_dir=config["out_dir"],
            db_limit=db_limit,
            resolution_filter=config["resolution_filter"],
            limit=config["limit"],
            timeout=config["timeout"],
            max_n_ratio=config["max_n_ratio"],
            min_seq_length=config["min_seq_length"],
        )

    print(f"\nAll done in: {datetime.datetime.now() - start}")
    print(f"Shard dir: {config['out_dir']}")
    print(f"Expected record bytes: {REC_BYTES} (uint8 CHW 3×224×224)")


if __name__ == "__main__":
    main()
