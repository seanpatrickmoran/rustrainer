import numpy as np
import random
from collections import defaultdict
import os
import sqlite3
import sys
import matplotlib.pyplot as plt
import numpy as np


def calculate_genomic_distances(sqlite_path):
    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        print(f"Connected to database: {sqlite_path}")
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return {}, {}
    
    distance_dict = {}
    chromosome_data = defaultdict(list)
    
    try:
        query = "SELECT key_id, coordinates FROM imag_with_seqs;"
        cursor.execute(query)
        
        for row_num, row in enumerate(cursor.fetchall(), 1):
            key_id = row[0]
            coordinates = row[1]
            
            try:
                parts = coordinates.split(',')
                
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
                
                distance = y2 - x1
                distance_dict[key_id] = distance
                chromosome_data[chr_a].append(distance)
                
            except (ValueError, IndexError) as e:
                print(f"Error parsing row {row_num} coordinates '{coordinates}': {e}")
                continue
    
    except sqlite3.Error as e:
        print(f"Database query error: {e}")
    finally:
        conn.close()
    
    return distance_dict, chromosome_data


def create_distance_histograms(distance_dict, chromosome_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: All distances combined
    all_distances = list(distance_dict.values())
    ax1.hist(all_distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('All Loop Span Distances')
    ax1.set_xlabel('Genomic Distance (bp)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    mean_dist = np.mean(all_distances)
    median_dist = np.median(all_distances)
    std_dist = np.std(all_distances)
    ax1.text(0.7, 0.9, f'Mean: {mean_dist:,.0f}\nMedian: {median_dist:,.0f}\nStd: {std_dist:,.0f}',
             transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    # Plot 2: Distances by chromosome
    colors = plt.cm.tab20(np.linspace(0, 1, len(chromosome_data)))
    
    for i, (chromosome, distances) in enumerate(chromosome_data.items()):
        ax2.hist(distances, bins=30, alpha=0.6, label=chromosome, 
                color=colors[i], edgecolor='black', linewidth=0.5)
    
    ax2.set_title('Loop Span Distances by Chromosome')
    ax2.set_xlabel('Genomic Distance (bp)')
    ax2.set_ylabel('Frequency')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def calculate_statistics(distance_dict, chromosome_data):
    all_distances = np.array(list(distance_dict.values()))
    
    # Overall statistics
    overall_stats = {
        'count': len(all_distances),
        'mean': np.mean(all_distances),
        'median': np.median(all_distances),
        'std': np.std(all_distances),
        'var': np.var(all_distances),
        'min': np.min(all_distances),
        'max': np.max(all_distances),
        'q25': np.percentile(all_distances, 25),
        'q75': np.percentile(all_distances, 75)
    }
    
    # Per-chromosome statistics
    chromosome_stats = {}
    for chromosome, distances in chromosome_data.items():
        distances_array = np.array(distances)
        chromosome_stats[chromosome] = {
            'count': len(distances_array),
            'mean': np.mean(distances_array),
            'median': np.median(distances_array),
            'std': np.std(distances_array),
            'var': np.var(distances_array),
            'min': np.min(distances_array),
            'max': np.max(distances_array),
            'q25': np.percentile(distances_array, 25),
            'q75': np.percentile(distances_array, 75)
        }
    
    return overall_stats, chromosome_stats

def print_statistics(overall_stats, chromosome_stats):
    """Print formatted statistics report"""
    print("-" * 30)
    print(f"Total Samples: {overall_stats['count']:,}")
    print(f"Mean Distance: {overall_stats['mean']:,.0f} bp")
    print(f"Median Distance: {overall_stats['median']:,.0f} bp")
    print(f"Standard Deviation: {overall_stats['std']:,.0f} bp")
    print(f"Variance: {overall_stats['var']:,.0f}")
    print(f"Min Distance: {overall_stats['min']:,.0f} bp")
    print(f"Max Distance: {overall_stats['max']:,.0f} bp")
    print(f"25th Percentile: {overall_stats['q25']:,.0f} bp")
    print(f"75th Percentile: {overall_stats['q75']:,.0f} bp")
    
    # Per-chromosome statistics
    print(f"{'Chromosome':<12} {'Count':<8} {'Mean':<12} {'Median':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    
    for chromosome, stats in chromosome_stats.items():
        print(f"{chromosome:<12} {stats['count']:<8} {stats['mean']:<12,.0f} {stats['median']:<12,.0f} "
              f"{stats['std']:<12,.0f} {stats['min']:<12,.0f} {stats['max']:<12,.0f}")
    

    
    for chromosome, stats in chromosome_stats.items():
        print(f"\n{chromosome}:")
        print(f"  Count: {stats['count']:,}")
        print(f"  Mean: {stats['mean']:,.0f} bp")
        print(f"  Median: {stats['median']:,.0f} bp")
        print(f"  Std Dev: {stats['std']:,.0f} bp")
        print(f"  Variance: {stats['var']:,.0f}")
        print(f"  Range: {stats['min']:,.0f} - {stats['max']:,.0f} bp")
        print(f"  IQR: {stats['q25']:,.0f} - {stats['q75']:,.0f} bp")


def load_blacklist_coordinates(sqlite_path):

    blacklist = set()
    
    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        print(f"Loading blacklist from database: {sqlite_path}")
        
        query = "SELECT coordinates FROM imag_with_seqs;"
        cursor.execute(query)
        
        for row in cursor.fetchall():
            coordinates = row[0]
            try:
                parts = coordinates.split(',')
                if len(parts) == 6:
                    chr_a = parts[0]
                    x1, x2 = int(parts[1]), int(parts[2])
                    chr_b = parts[3] 
                    y1, y2 = int(parts[4]), int(parts[5])
                    
                    if chr_a == chr_b:
                        blacklist.add((chr_a, x1, x2, y1, y2))
                        blacklist.add((chr_a, y1, y2, x1, x2))
                        
            except (ValueError, IndexError) as e:
                continue
                
        conn.close()
        print(f"Loaded {len(blacklist)} coordinate pairs into blacklist")
        
    except sqlite3.Error as e:
        print(f"Warning: Could not load blacklist from database: {e}")
        print("Proceeding without blacklist...")
        
    return blacklist

def check_coordinate_collision(chr_name, bin1_start, bin1_end, bin2_start, bin2_end, blacklist):
    coord1 = (chr_name, bin1_start, bin1_end, bin2_start, bin2_end)
    coord2 = (chr_name, bin2_start, bin2_end, bin1_start, bin1_end)
    
    return coord1 in blacklist or coord2 in blacklist





def generate_synthetic_hic_coordinates(chromosome_stats, chromosome_sizes, sqlite_path=None,
                                     num_samples=1000, snap_interval=5000, 
                                     default_bin_size=5000, output_file="synthetic_hic_coordinates.txt",
                                     max_attempts_per_sample=100):
    blacklist = set()
    if sqlite_path:
        blacklist = load_blacklist_coordinates(sqlite_path)
    else:
        print("No SQLite path provided - proceeding without blacklist")
    
    chr_sizes = {}
    for chr_name, size in chromosome_sizes.items():
        chr_sizes[chr_name] = size
    
    print(f"Chromosome sizes loaded: {len(chr_sizes)} chromosomes")
    print(f"Available chromosomes in stats: {list(chromosome_stats.keys())}")
    
    total_original_samples = sum(stats['count'] for stats in chromosome_stats.values())
    chr_proportions = {}
    
    for chr_name, stats in chromosome_stats.items():
        if chr_name in chr_sizes:  # Only include chromosomes we have size data for
            chr_proportions[chr_name] = stats['count'] / total_original_samples
    
    print(f"\nChromosome proportions:")
    for chr_name, prop in chr_proportions.items():
        print(f"  {chr_name}: {prop:.3f} ({chromosome_stats[chr_name]['count']} samples)")
    
    generated_coordinates = []
    collision_count = 0
    total_attempts = 0
    
    for chr_name, proportion in chr_proportions.items():
        chr_samples = int(num_samples * proportion)
        if chr_samples == 0:
            chr_samples = 1
            
        chr_size = chr_sizes[chr_name]
        chr_stats = chromosome_stats[chr_name]
        
        print(f"\nGenerating {chr_samples} samples for {chr_name} (size: {chr_size:,} bp)")
        print(f"  Original stats - Mean: {chr_stats['mean']:.0f}, Std: {chr_stats['std']:.0f}")
        
        successful_samples = 0
        
        while successful_samples < chr_samples:
            attempts_for_this_sample = 0
            
            while attempts_for_this_sample < max_attempts_per_sample:
                total_attempts += 1
                attempts_for_this_sample += 1

                distance = max(snap_interval, 
                              int(np.random.normal(chr_stats['mean'], chr_stats['std'])))
                
                distance = int(distance // snap_interval) * snap_interval
                
                max_start_pos = chr_size - distance - default_bin_size
                if max_start_pos <= 0:
                    continue  # Skip if distance too large for chromosome
                    
                start_pos = random.randint(0, max_start_pos // snap_interval) * snap_interval
                
                bin1_start = start_pos
                bin1_end = bin1_start + default_bin_size
                
                bin2_start = bin1_start + distance
                bin2_end = bin2_start + default_bin_size
                
                if bin2_end > chr_size:
                    continue
                
                if blacklist and check_coordinate_collision(chr_name, bin1_start, bin1_end, 
                                                          bin2_start, bin2_end, blacklist):
                    collision_count += 1
                    continue  # Try again
                    
                coord_string = f"{chr_name},{bin1_start},{bin1_end},{chr_name},{bin2_start},{bin2_end}"
                generated_coordinates.append(coord_string)
                successful_samples += 1
                break
            
            if attempts_for_this_sample >= max_attempts_per_sample:
                print(f"  Warning: Could not generate non-colliding coordinate for {chr_name} after {max_attempts_per_sample} attempts")
                break
    
    with open(output_file, 'w') as out_file:
        out_file.write("BIN1_CHR\tBIN1_START\tBIN1_END\tBIN2_CHROMOSOME\tBIN2_START\tBIN2_END\tFDR\tDETECTION_SCALE\n")
    
        for coord_string in generated_coordinates:
            parts = coord_string.split(',')
            chr_a, x1, x2, chr_b, y1, y2 = parts
            fdr_value = random.uniform(0.001, 0.05)
            out_file.write(f"{chr_a}\t{x1}\t{x2}\t{chr_b}\t{y1}\t{y2}\t{fdr_value:.6f}\t0.01\n")
    
    print(f"\nGenerated {len(generated_coordinates)} coordinate pairs")
    print(f"Total attempts: {total_attempts}")
    print(f"Collisions avoided: {collision_count}")
    if blacklist:
        print(f"Collision rate: {collision_count/total_attempts*100:.2f}%")
    print(f"Results saved to: {output_file}")
    
    return generated_coordinates

def verify_generated_distances(coordinate_strings):
    distances_by_chr = defaultdict(list)
    
    for coord_string in coordinate_strings:
        parts = coord_string.split(',')
        chr_a, x1, x2, chr_b, y1, y2 = parts[0], int(parts[1]), int(parts[2]), parts[3], int(parts[4]), int(parts[5])
        
        if chr_a == chr_b:
            distance = y2 - x1  # Same calculation as original
            distances_by_chr[chr_a].append(distance)
    
    for chr_name, distances in distances_by_chr.items():
        distances = np.array(distances)
        print(f"{chr_name}: Count={len(distances)}, Mean={np.mean(distances):.0f}, "
              f"Std={np.std(distances):.0f}, Range={np.min(distances):.0f}-{np.max(distances):.0f}")

def main_generate_synthetic_data(SQLITE_PATH, num_samples=80000):
    chromosome_sizes = {
        'chr1': 248956422,
        'chr2': 242193529,
        'chr3': 198295559,
        'chr4': 190214555,
        'chr5': 181538259,
        'chr6': 170805979,
        'chr7': 159345973,
        'chrX': 156040895,
        'chr8': 145138636,
        'chr9': 138394717,
        'chr11': 135086622,
        'chr10': 133797422,
        'chr12': 133275309,
        'chr13': 114364328,
        'chr14': 107043718,
        'chr15': 101991189,
        'chr16': 90338345,
        'chr17': 83257441,
        'chr18': 80373285,
        'chr20': 64444167,
        'chr19': 58617616,
        'chrY': 57227415,
        'chr22': 50818468,
        'chr21': 46709983,
        'chrM': 16569,
        '1': 248956422,
        '2': 242193529,
        '3': 198295559,
        '4': 190214555,
        '5': 181538259,
        '6': 170805979,
        '7': 159345973,
        'X': 156040895,
        '8': 145138636,
        '9': 138394717,
        '11': 135086622,
        '10': 133797422,
        '12': 133275309,
        '13': 114364328,
        '14': 107043718,
        '15': 101991189,
        '16': 90338345,
        '17': 83257441,
        '18': 80373285,
        '20': 64444167,
        '19': 58617616,
        'Y': 57227415,
        '22': 50818468,
        '21': 46709983,
        'M': 16569
    }
    
    # Example chromosome statistics (you need to replace this with your actual data)
    # This should come from your calculate_statistics() function output
    distance_dict, chromosome_data = calculate_genomic_distances(SQLITE_PATH)
    overall_stats, chromosome_stats = calculate_statistics(distance_dict, chromosome_data)
    


    print("IMPORTANT: Replace 'example_chromosome_stats' with your actual")
    print("chromosome statistics from calculate_statistics() function!")
    print("\nTo use this function properly:")
    print("1. Run your calculate_genomic_distances() and calculate_statistics() functions")
    print("2. Pass the chromosome_stats dictionary to this function")
    print("3. Adjust num_samples, snap_interval, and default_bin_size as needed")
    
    # Uncomment and modify this line when you have your actual chromosome_stats
    generated_coords = generate_synthetic_hic_coordinates(
        chromosome_stats=chromosome_stats,  # Replace with actual data
        chromosome_sizes=chromosome_sizes,
        sqlite_path=SQLITE_PATH,  # This enables blacklisting
        num_samples=500000,
        snap_interval=5000,
        default_bin_size=5000,
        output_file="synthetic_hic_coordinates.txt",
        max_attempts_per_sample=100  # Adjust if you get too many collision warnings
    )
    
    verify_generated_distances(generated_coords)


if __name__ == "__main__":
    # You can modify the database path here
    if len(sys.argv) > 1:
        SQLITE_PATH = sys.argv[1]
    else:
        SQLITE_PATH = "your_database.db"  # Default path

    main_generate_synthetic_data(SQLITE_PATH)
