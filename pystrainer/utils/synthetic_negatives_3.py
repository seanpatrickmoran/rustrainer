#!/usr/bin/env python3
import argparse
import os
import random
import sqlite3
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# Data extraction + stats
# -----------------------------

def calculate_genomic_distances_with_metadata(sqlite_path: str, resolution_filter: int | None = None):
    """
    Calculate genomic distances from Hi-C coordinate data with dataset/hic_path metadata.

    Returns:
        (distance_dict, chromosome_data, dataset_hicpath_data)
    """
    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        print(f"Connected to database: {sqlite_path}")
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return {}, {}, {}

    distance_dict = {}
    chromosome_data = defaultdict(list)          # chromosome -> distances
    dataset_hicpath_data = defaultdict(list)     # (dataset, hic_path) -> distances

    try:
        if resolution_filter is not None:
            query = "SELECT key_id, coordinates, dataset, hic_path FROM imag_with_seqs WHERE resolution = ?;"
            cursor.execute(query, (str(resolution_filter),))
            print(f"Filtering for resolution = {resolution_filter}")
        else:
            query = "SELECT key_id, coordinates, dataset, hic_path FROM imag_with_seqs;"
            cursor.execute(query)
            print("No resolution filter applied")

        for row_num, (key_id, coordinates, dataset, hic_path) in enumerate(cursor.fetchall(), 1):
            dataset = dataset or "unknown"
            hic_path = hic_path or "unknown"

            try:
                parts = coordinates.split(",")
                if len(parts) != 6:
                    print(f"Warning: Row {row_num} has {len(parts)} parts, expected 6. Skipping.")
                    continue

                chr_a = parts[0]
                x1 = int(parts[1])
                x2 = int(parts[2])
                chr_b = parts[3]
                y1 = int(parts[4])
                y2 = int(parts[5])

                if chr_a != chr_b:
                    print(f"Warning: Row {row_num} has mismatched chromosomes ({chr_a} != {chr_b}). Skipping.")
                    continue

                distance = y2 - x1  # original logic

                distance_dict[key_id] = {
                    "distance": distance,
                    "chromosome": chr_a,
                    "dataset": dataset,
                    "hic_path": hic_path,
                    "coordinates": coordinates,
                }
                chromosome_data[chr_a].append(distance)
                dataset_hicpath_data[(dataset, hic_path)].append(distance)

            except (ValueError, IndexError) as e:
                print(f"Error parsing row {row_num} coordinates '{coordinates}': {e}")
                continue

    except sqlite3.Error as e:
        print(f"Database query error: {e}")
    finally:
        conn.close()

    return distance_dict, chromosome_data, dataset_hicpath_data


def calculate_enhanced_statistics(distance_dict, chromosome_data, dataset_hicpath_data):
    """Returns (overall_stats, chromosome_stats, dataset_hicpath_stats)."""
    all_distances = np.array([d["distance"] for d in distance_dict.values()], dtype=float)

    overall_stats = {
        "count": int(all_distances.size),
        "mean": float(np.mean(all_distances)),
        "median": float(np.median(all_distances)),
        "std": float(np.std(all_distances)),
        "var": float(np.var(all_distances)),
        "min": float(np.min(all_distances)),
        "max": float(np.max(all_distances)),
        "q25": float(np.percentile(all_distances, 25)),
        "q75": float(np.percentile(all_distances, 75)),
    }

    chromosome_stats = {}
    for chromosome, distances in chromosome_data.items():
        arr = np.array(distances, dtype=float)
        chromosome_stats[chromosome] = {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "var": float(np.var(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
        }

    dataset_hicpath_stats = {}
    for (dataset, hic_path), distances in dataset_hicpath_data.items():
        if not distances:
            continue
        arr = np.array(distances, dtype=float)
        dataset_hicpath_stats[(dataset, hic_path)] = {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "var": float(np.var(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
        }

    return overall_stats, chromosome_stats, dataset_hicpath_stats


def print_enhanced_statistics(overall_stats, chromosome_stats, dataset_hicpath_stats):
    print("\n" + "=" * 80)
    print("GENOMIC DISTANCE STATISTICS REPORT - ENHANCED VERSION")
    print("=" * 80)

    print("\nOVERALL STATISTICS")
    print("-" * 40)
    print(f"Total Samples: {overall_stats['count']:,}")
    print(f"Mean Distance: {overall_stats['mean']:,.0f} bp")
    print(f"Median Distance: {overall_stats['median']:,.0f} bp")
    print(f"Standard Deviation: {overall_stats['std']:,.0f} bp")
    print(f"Variance: {overall_stats['var']:,.0f}")
    print(f"Min Distance: {overall_stats['min']:,.0f} bp")
    print(f"Max Distance: {overall_stats['max']:,.0f} bp")
    print(f"25th Percentile: {overall_stats['q25']:,.0f} bp")
    print(f"75th Percentile: {overall_stats['q75']:,.0f} bp")

    print("\nPER-CHROMOSOME STATISTICS")
    print("-" * 40)
    print(f"{'Chromosome':<12} {'Count':<8} {'Mean':<12} {'Median':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 90)
    for chromosome, stats in sorted(chromosome_stats.items()):
        print(
            f"{chromosome:<12} {stats['count']:<8} {stats['mean']:<12,.0f} {stats['median']:<12,.0f} "
            f"{stats['std']:<12,.0f} {stats['min']:<12,.0f} {stats['max']:<12,.0f}"
        )

    print("\nPER DATASET/HIC_PATH STATISTICS")
    print("-" * 80)
    print(f"{'Dataset/HIC Path':<40} {'Count':<8} {'Mean':<12} {'Median':<12} {'Std':<12}")
    print("-" * 80)
    for (dataset, hic_path), stats in sorted(dataset_hicpath_stats.items()):
        nickname = os.path.basename(hic_path).removesuffix(".hic") if hic_path else "unknown"
        label = f"{dataset}/{nickname}"[:39]
        print(
            f"{label:<40} {stats['count']:<8} {stats['mean']:<12,.0f} "
            f"{stats['median']:<12,.0f} {stats['std']:<12,.0f}"
        )


def save_statistics_report(overall_stats, chromosome_stats, dataset_hicpath_stats, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"statistics_report_{timestamp}.txt")

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GENOMIC DISTANCE STATISTICS REPORT - ENHANCED VERSION\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")

        f.write("\nOVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Samples: {overall_stats['count']:,}\n")
        f.write(f"Mean Distance: {overall_stats['mean']:,.0f} bp\n")
        f.write(f"Median Distance: {overall_stats['median']:,.0f} bp\n")
        f.write(f"Standard Deviation: {overall_stats['std']:,.0f} bp\n")
        f.write(f"Variance: {overall_stats['var']:,.0f}\n")
        f.write(f"Min Distance: {overall_stats['min']:,.0f} bp\n")
        f.write(f"Max Distance: {overall_stats['max']:,.0f} bp\n")
        f.write(f"25th Percentile: {overall_stats['q25']:,.0f} bp\n")
        f.write(f"75th Percentile: {overall_stats['q75']:,.0f} bp\n")

        f.write("\nPER-CHROMOSOME STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"{'Chromosome':<12} {'Count':<8} {'Mean':<12} {'Median':<12} {'Std':<12} {'Min':<12} {'Max':<12}\n"
        )
        f.write("-" * 90 + "\n")
        for chromosome, stats in sorted(chromosome_stats.items()):
            f.write(
                f"{chromosome:<12} {stats['count']:<8} {stats['mean']:<12,.0f} {stats['median']:<12,.0f} "
                f"{stats['std']:<12,.0f} {stats['min']:<12,.0f} {stats['max']:<12,.0f}\n"
            )

        f.write("\nPER DATASET/HIC_PATH STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Dataset/HIC Path':<40} {'Count':<8} {'Mean':<12} {'Median':<12} {'Std':<12}\n")
        f.write("-" * 80 + "\n")
        for (dataset, hic_path), stats in sorted(dataset_hicpath_stats.items()):
            nickname = os.path.basename(hic_path).removesuffix(".hic") if hic_path else "unknown"
            label = f"{dataset}/{nickname}"[:39]
            f.write(
                f"{label:<40} {stats['count']:<8} {stats['mean']:<12,.0f} "
                f"{stats['median']:<12,.0f} {stats['std']:<12,.0f}\n"
            )

    print(f"  Saved statistics report: {report_path}")
    return report_path


def get_available_resolutions(sqlite_path: str):
    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT resolution FROM imag_with_seqs ORDER BY resolution;")
        resolutions = [row[0] for row in cursor.fetchall()]
        conn.close()
        print(f"Found {len(resolutions)} unique resolutions in database: {resolutions}")
        return resolutions
    except sqlite3.Error as e:
        print(f"Error getting resolutions from database: {e}")
        return []


# -----------------------------
# Plotting
# -----------------------------

def create_enhanced_distance_histograms(distance_dict, chromosome_data, dataset_hicpath_data,
                                       output_dir="analysis_outputs", save_format="png", dpi=300):
    os.makedirs(output_dir, exist_ok=True)

    all_distances = [d["distance"] for d in distance_dict.values()]
    if not all_distances:
        print("No distances to plot.")
        return

    fig = plt.figure(figsize=(20, 16))

    # 1) all distances
    ax1 = plt.subplot(2, 2, 1)
    ax1.hist(all_distances, bins=50, alpha=0.7, edgecolor="black")
    ax1.set_title("All Loop Span Distances", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Genomic Distance (bp)")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)
    mean_dist = np.mean(all_distances)
    median_dist = np.median(all_distances)
    std_dist = np.std(all_distances)
    ax1.text(
        0.7, 0.9,
        f"Mean: {mean_dist:,.0f}\nMedian: {median_dist:,.0f}\nStd: {std_dist:,.0f}",
        transform=ax1.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # 2) by chromosome
    ax2 = plt.subplot(2, 2, 2)
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(chromosome_data))))
    for i, (chromosome, distances) in enumerate(sorted(chromosome_data.items())):
        ax2.hist(distances, bins=30, alpha=0.6, label=chromosome,
                 color=colors[i % len(colors)], edgecolor="black", linewidth=0.5)
    ax2.set_title("Loop Span Distances by Chromosome", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Genomic Distance (bp)")
    ax2.set_ylabel("Frequency")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)
    ax2.grid(True, alpha=0.3)

    # 3) by dataset/hic path (box)
    ax3 = plt.subplot(2, 2, 3)
    dataset_labels, dataset_distances = [], []
    for (dataset, hic_path), distances in sorted(dataset_hicpath_data.items()):
        nickname = os.path.basename(hic_path).removesuffix(".hic") if hic_path else "unknown"
        label = f"{dataset}/{nickname}"
        dataset_labels.append(label[:20])
        dataset_distances.append(distances)

    if dataset_distances:
        bp = ax3.boxplot(dataset_distances, labels=dataset_labels, patch_artist=True)
        colors_box = plt.cm.Set3(np.linspace(0, 1, len(dataset_distances)))
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax3.set_title("Loop Span Distribution by Dataset/HIC Path", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Dataset/HIC Path")
        ax3.set_ylabel("Genomic Distance (bp)")
        ax3.grid(True, alpha=0.3, axis="y")
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")
    else:
        ax3.text(0.5, 0.5, "No dataset/hic_path groups", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_axis_off()

    # 4) summary table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("tight")
    ax4.axis("off")

    table_data = [["Dataset/HIC Path", "Count", "Mean", "Std", "Min", "Max"]]
    for (dataset, hic_path), distances in sorted(dataset_hicpath_data.items())[:10]:
        nickname = os.path.basename(hic_path).removesuffix(".hic") if hic_path else "unknown"
        label = f"{dataset}/{nickname}"[:30]
        arr = np.array(distances, dtype=float)
        table_data.append([
            label,
            f"{len(distances)}",
            f"{np.mean(arr):,.0f}",
            f"{np.std(arr):,.0f}",
            f"{np.min(arr):,.0f}",
            f"{np.max(arr):,.0f}",
        ])

    table = ax4.table(
        cellText=table_data,
        cellLoc="left",
        loc="center",
        colWidths=[0.35, 0.1, 0.15, 0.15, 0.125, 0.125],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    for i in range(6):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    for i in range(1, len(table_data)):
        if i % 2 == 0:
            for j in range(6):
                table[(i, j)].set_facecolor("#f1f1f2")

    ax4.set_title("Summary Statistics (Top 10)", fontsize=14, fontweight="bold", pad=20)

    plt.suptitle("Hi-C Loop Span Distance Analysis", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"hic_distance_analysis_combined_{timestamp}.{save_format}")
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"  Saved combined analysis plot: {filepath}")
    plt.close()


# -----------------------------
# Blacklist
# -----------------------------

def load_blacklist_with_metadata(sqlite_path: str, resolution_filter: int | None = None):
    """
    Returns:
        (blacklist_set, dataset_hicpath_coords)
    """
    blacklist = set()
    dataset_hicpath_coords = defaultdict(set)

    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        print(f"Loading blacklist and metadata from database: {sqlite_path}")

        if resolution_filter is not None:
            cursor.execute(
                "SELECT coordinates, dataset, hic_path FROM imag_with_seqs WHERE resolution = ?;",
                (str(resolution_filter),),
            )
            print(f"Loading blacklist for resolution = {resolution_filter}")
        else:
            cursor.execute("SELECT coordinates, dataset, hic_path FROM imag_with_seqs;")
            print("Loading blacklist for all resolutions")

        for (coordinates, dataset, hic_path) in cursor.fetchall():
            dataset = dataset or "unknown"
            hic_path = hic_path or "unknown"

            try:
                parts = coordinates.split(",")
                if len(parts) != 6:
                    continue
                chr_a = parts[0]
                x1, x2 = int(parts[1]), int(parts[2])
                chr_b = parts[3]
                y1, y2 = int(parts[4]), int(parts[5])

                if chr_a != chr_b:
                    continue

                coord1 = (chr_a, x1, x2, y1, y2)
                coord2 = (chr_a, y1, y2, x1, x2)

                blacklist.add(coord1)
                blacklist.add(coord2)
                dataset_hicpath_coords[(dataset, hic_path)].add(coord1)
                dataset_hicpath_coords[(dataset, hic_path)].add(coord2)

            except (ValueError, IndexError):
                continue

        conn.close()
        print(f"Loaded {len(blacklist)} coordinate pairs into blacklist")
        print(f"Found {len(dataset_hicpath_coords)} unique dataset/hic_path combinations")

    except sqlite3.Error as e:
        print(f"Warning: Could not load blacklist from database: {e}")
        print("Proceeding without blacklist...")

    return blacklist, dataset_hicpath_coords


# -----------------------------
# Synthetic generation
# -----------------------------

def generate_stratified_synthetic_coordinates(
    dataset_hicpath_stats,
    chromosome_stats,
    chromosome_sizes,
    sqlite_path=None,
    target_per_dataset: int | None = None,
    snap_interval=5000,
    default_bin_size=5000,
    output_dir="synthetic_outputs",
    limiter=100,
    resolution_filter=None,
):
    """
    Generate synthetic Hi-C coordinates stratified by dataset/hic_path.

    - If target_per_dataset is None: match original counts per dataset/hic_path.
    - If target_per_dataset is set: try to generate that many per dataset/hic_path.
      If collisions/constraints prevent it, may generate fewer.

    limiter: max attempts per sample before skipping that sample.
    """
    os.makedirs(output_dir, exist_ok=True)

    blacklist = set()
    if sqlite_path:
        blacklist, _ = load_blacklist_with_metadata(sqlite_path, resolution_filter)

    chr_sizes = dict(chromosome_sizes)
    print("\nGenerating stratified synthetic coordinates")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    all_generated_coords = {}

    total_chr_samples = sum(stats["count"] for stats in chromosome_stats.values()) or 1

    for (dataset, hic_path), dh_stats in dataset_hicpath_stats.items():
        nickname = os.path.basename(hic_path).removesuffix(".hic") if hic_path else "unknown"
        output_filename = f"SynNeg_{nickname}_{snap_interval//1000}k.tsv"
        output_path = os.path.join(output_dir, output_filename)

        target_samples = int(target_per_dataset) if target_per_dataset is not None else int(dh_stats["count"])

        print(f"\nProcessing: {dataset}/{nickname}")
        print(f"  Target samples: {target_samples}")
        print(f"  Output file: {output_filename}")

        generated = []
        collisions = 0
        failed_samples = 0

        for chr_name, chr_stat in chromosome_stats.items():
            if chr_name not in chr_sizes:
                continue

            chr_proportion = chr_stat["count"] / total_chr_samples
            chr_samples = int(round(target_samples * chr_proportion))
            if chr_samples <= 0:
                continue

            chr_size = chr_sizes[chr_name]
            # use dataset stats if enough points, else chromosome stats
            stats_to_use = dh_stats if dh_stats["count"] > 30 else chr_stat

            for _ in range(chr_samples):
                success = False
                for _attempt in range(int(limiter)):
                    # distance ~ N(mean,std), snapped, min snap_interval
                    distance = max(
                        snap_interval,
                        int(np.random.normal(stats_to_use["mean"], max(1.0, stats_to_use["std"]))),
                    )
                    distance = (distance // snap_interval) * snap_interval

                    max_start_pos = chr_size - distance - default_bin_size
                    if max_start_pos <= 0:
                        continue

                    start_pos = random.randint(0, max_start_pos // snap_interval) * snap_interval
                    bin1_start = start_pos
                    bin1_end = bin1_start + default_bin_size
                    bin2_start = bin1_start + distance
                    bin2_end = bin2_start + default_bin_size

                    if bin2_end > chr_size:
                        continue

                    coord = (chr_name, bin1_start, bin1_end, bin2_start, bin2_end)
                    coord_rev = (chr_name, bin2_start, bin2_end, bin1_start, bin1_end)

                    if coord in blacklist or coord_rev in blacklist:
                        collisions += 1
                        continue

                    generated.append({
                        "chr": chr_name,
                        "bin1_start": bin1_start,
                        "bin1_end": bin1_end,
                        "bin2_start": bin2_start,
                        "bin2_end": bin2_end,
                        "fdr": random.uniform(0.001, 0.05),
                    })
                    success = True
                    break

                if not success:
                    failed_samples += 1

        # If rounding by chromosome proportions overshot/undershot, we won't force-correct here.
        # The key guarantee: we never exceed constraints; we try hard but may produce fewer.
        with open(output_path, "w") as out_file:
            out_file.write(
                "BIN1_CHR\tBIN1_START\tBIN1_END\tBIN2_CHROMOSOME\tBIN2_START\tBIN2_END\tFDR\tDETECTION_SCALE\n"
            )
            for coord in generated:
                out_file.write(
                    f"{coord['chr']}\t{coord['bin1_start']}\t{coord['bin1_end']}\t"
                    f"{coord['chr']}\t{coord['bin2_start']}\t{coord['bin2_end']}\t"
                    f"{coord['fdr']:.6f}\t0.01\n"
                )

        all_generated_coords[(dataset, hic_path)] = generated

        print(f"  Generated: {len(generated)} coordinates")
        print(f"  Collisions avoided: {collisions}")
        if failed_samples:
            print(f"  Skipped (hit limiter={limiter}): {failed_samples}")
        if target_samples is not None and len(generated) < target_samples:
            print(f"  NOTE: Could not reach target {target_samples}; produced {len(generated)}.")

        print(f"  Saved to: {output_path}")

    return all_generated_coords


# -----------------------------
# Main pipeline
# -----------------------------

def main_enhanced_synthetic_data(sqlite_path: str, resolution: int | None,
                                generate: int | None, limiter: int,
                                output_root: str | None = None):
    # Chromosome sizes
    chromosome_sizes = {
        "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
        "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chrX": 156040895,
        "chr8": 145138636, "chr9": 138394717, "chr11": 135086622, "chr10": 133797422,
        "chr12": 133275309, "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
        "chr16": 90338345, "chr17": 83257441, "chr18": 80373285, "chr20": 64444167,
        "chr19": 58617616, "chrY": 57227415, "chr22": 50818468, "chr21": 46709983,
        "chrM": 16569,
    }

    if resolution is not None:
        resolutions_to_process = [resolution]
        print(f"Processing specified resolution: {resolution}")
    else:
        resolutions_to_process = get_available_resolutions(sqlite_path)
        if not resolutions_to_process:
            print("No resolutions found in database. Exiting.")
            return {}
        print(f"\nWill process {len(resolutions_to_process)} resolutions: {resolutions_to_process}")

    all_results = {}

    for res in resolutions_to_process:
        if res == None:
            continue
        res = int(res)
        print(f"\n{'=' * 80}")
        print(f"PROCESSING RESOLUTION: {res} bp ({res // 1000}kb)")
        print(f"{'=' * 80}")

        # outputs (same style as your original, but optionally under an output root)
        analysis_dir = f"analysis_outputs_{res // 1000}k"
        synthetic_dir = f"synthetic_outputs_{res // 1000}k"
        if output_root:
            analysis_dir = os.path.join(output_root, analysis_dir)
            synthetic_dir = os.path.join(output_root, synthetic_dir)

        print(f"\nStep 1: Calculating genomic distances with metadata (resolution={res})...")
        distance_dict, chromosome_data, dataset_hicpath_data = calculate_genomic_distances_with_metadata(
            sqlite_path, resolution_filter=res
        )
        if not distance_dict:
            print(f"No data found in database for resolution={res}. Skipping.")
            continue

        print(f"\nStep 2: Calculating enhanced statistics for resolution={res}...")
        overall_stats, chromosome_stats, dataset_hicpath_stats = calculate_enhanced_statistics(
            distance_dict, chromosome_data, dataset_hicpath_data
        )

        print(f"\nStep 3: Generating statistics report for resolution={res}...")
        print_enhanced_statistics(overall_stats, chromosome_stats, dataset_hicpath_stats)
        report_path = save_statistics_report(overall_stats, chromosome_stats, dataset_hicpath_stats, output_dir=analysis_dir)

        print(f"\nStep 4: Creating enhanced visualizations for resolution={res}...")
        create_enhanced_distance_histograms(
            distance_dict, chromosome_data, dataset_hicpath_data,
            output_dir=analysis_dir, save_format="png", dpi=300
        )

        print(f"\nStep 5: Generating stratified synthetic coordinates (resolution={res})...")
        generated_coords = generate_stratified_synthetic_coordinates(
            dataset_hicpath_stats=dataset_hicpath_stats,
            chromosome_stats=chromosome_stats,
            chromosome_sizes=chromosome_sizes,
            sqlite_path=sqlite_path,
            target_per_dataset=generate,        # <-- your --generate knob
            snap_interval=res,
            default_bin_size=res,
            output_dir=synthetic_dir,
            limiter=limiter,                    # <-- your --limiter knob
            resolution_filter=res,
        )

        all_results[res] = {
            "distance_dict": distance_dict,
            "overall_stats": overall_stats,
            "chromosome_stats": chromosome_stats,
            "dataset_hicpath_stats": dataset_hicpath_stats,
            "generated_coords": generated_coords,
            "analysis_dir": analysis_dir,
            "synthetic_dir": synthetic_dir,
            "report_path": report_path,
        }

        print(f"\n{'-' * 60}")
        print(f"SUMMARY FOR RESOLUTION {res} bp ({res // 1000}kb)")
        print(f"{'-' * 60}")
        print(f"Total dataset/hic_path combinations: {len(generated_coords)}")
        total_generated = sum(len(coords) for coords in generated_coords.values())
        print(f"Total synthetic coordinates generated: {total_generated:,}")
        print(f"Analysis outputs directory: {analysis_dir}/")
        print(f"Synthetic outputs directory: {synthetic_dir}/")

    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY - ALL RESOLUTIONS PROCESSED")
    print(f"{'=' * 80}")
    print(f"Resolutions processed: {len(all_results)}")

    for res in sorted(all_results.keys()):
        res_data = all_results[res]
        total_coords = sum(len(coords) for coords in res_data["generated_coords"].values())
        print(f"\nResolution {res} bp ({res // 1000}kb):")
        print(f"  - Original data points: {res_data['overall_stats']['count']:,}")
        print(f"  - Dataset/HIC combinations: {len(res_data['dataset_hicpath_stats'])}")
        print(f"  - Synthetic coordinates generated: {total_coords:,}")
        print(f"  - Analysis directory: {res_data['analysis_dir']}/")
        print(f"  - Synthetic directory: {res_data['synthetic_dir']}/")

    print(f"\n{'=' * 80}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 80}")
    return all_results


def build_argparser():
    p = argparse.ArgumentParser(
        description="Hi-C loop span distance analysis + stratified synthetic negative generation."
    )
    p.add_argument("--sqlite_path", help="Path to SQLite database containing imag_with_seqs.")
    p.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Process only this resolution (bp). Default: process all resolutions found in DB.",
    )
    p.add_argument(
        "--generate",
        type=int,
        default=None,
        help=(
            "Generate this many synthetic coordinates per dataset/hic_path (tries hard; may output fewer). "
            "Default: match original per dataset/hic_path counts."
        ),
    )
    p.add_argument(
        "--limiter",
        type=int,
        default=100,
        help="Max attempts per sample before skipping that sample. Default: 100.",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional root directory to place analysis_outputs_* and synthetic_outputs_* under.",
    )
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main_enhanced_synthetic_data(
        sqlite_path=args.sqlite_path,
        resolution=args.resolution,
        generate=args.generate,
        limiter=args.limiter,
        output_root=args.output_root,
    )
