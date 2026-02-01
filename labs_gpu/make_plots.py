import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
METRICS_FILE = "variant_10steps_benchmark_metrics.csv"
GPU_MAX_N = 512  # The fixed width used in your export_ready_for_cuda function
OUTPUT_PLOT = "10step_benchmark.png"

def get_hamming_stats(file_path, n, target_bitstring):
    """
    Returns (min_hamming, avg_hamming) from the binary population file.
    """
    if not os.path.exists(file_path) or pd.isna(target_bitstring) or target_bitstring == "N/A":
        return None, None
    
    try:
        # 1. Standardize the target bitstring (handle strings/ints/floats)
        if isinstance(target_bitstring, (float, int, np.number)):
            target_str = str(int(target_bitstring)).zfill(n)
        else:
            target_str = str(target_bitstring).strip()

        target = np.array([int(b) for b in target_str], dtype=np.int8)

        # 2. Load and Reshape Binary Data
        raw_data = np.fromfile(file_path, dtype=np.int8)
        actual_pop_size = len(raw_data) // GPU_MAX_N
        if actual_pop_size == 0:
            return None, None

        population = raw_data.reshape((actual_pop_size, GPU_MAX_N))
        
        # 3. Slice to N and Convert [-1, 1] to [0, 1]
        samples = (population[:, :n] + 1) // 2
        
        # 4. Calculate Hamming Distances for the whole population
        # (normalized by N so 0.0 is perfect and 1.0 is opposite)
        distances = np.sum(samples != target, axis=1) / n
        
        return np.min(distances), np.mean(distances)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def main():
    if not os.path.exists(METRICS_FILE):
        print(f"Error: {METRICS_FILE} not found.")
        return

    # Load CSV with bitstrings as strings
    df = pd.read_csv(METRICS_FILE, dtype={'best_bitstring': str})

    # Numeric conversion
    for col in ['total_time', 'equiv_evals', 'N']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print("Calculating population Hamming statistics...")
    
    # Apply the function and expand results into two columns
    hamming_results = df.apply(
        lambda row: get_hamming_stats(row['warm_start_file'], int(row['N']), row['best_bitstring']), 
        axis=1
    )
    df[['min_hamming', 'avg_hamming']] = pd.DataFrame(hamming_results.tolist(), index=df.index)

    # Drop failures
    df = df.dropna(subset=['min_hamming', 'avg_hamming'])

    # --- Plotting ---
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Quantum Population Quality & Performance Scaling", fontsize=20)

    # Plot 1: Average Hamming Distance (Top Left)
    # This shows the "center of mass" of the quantum state
    sns.lineplot(ax=axes[0, 0], data=df, x="N", y="avg_hamming", hue="variant", style="steps", markers=True)
    axes[0, 0].set_title("Average Population Quality (Mean Hamming Dist)")
    axes[0, 0].set_ylabel("Avg Distance (0.5 = Random)")
    axes[0, 0].axhline(0.5, ls='--', color='red', alpha=0.4, label="Random Level")
    axes[0, 0].set_ylim(0, 0.6)

    # Plot 2: Minimum Hamming Distance (Top Right)
    # This shows the best single seed provided to the optimizer
    sns.lineplot(ax=axes[0, 1], data=df, x="N", y="min_hamming", hue="variant", style="steps", markers=True)
    axes[0, 1].set_title("Best Seed Quality (Min Hamming Dist)")
    axes[0, 1].set_ylabel("Min Distance (Lower is Better)")
    axes[0, 1].axhline(0.5, ls='--', color='red', alpha=0.4)
    axes[0, 1].set_ylim(0, 0.6)

    # Plot 3: Time Scaling (Bottom Left)
    # df = df[df['variant'] != "beyblade"]
    sns.lineplot(ax=axes[1, 0], data=df, x="N", y="total_time", hue="variant", style="steps", markers=True)
    axes[1, 0].set_title("Total Execution Time (Seconds)")

    # Plot 4: Complexity (Bottom Right)
    sns.lineplot(ax=axes[1, 1], data=df, x="N", y="equiv_evals", hue="variant", style="steps", markers=True)
    axes[1, 1].set_title("Equivalent Evaluations (RunOpt Effort)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_PLOT)
    print(f"Analysis complete. Results saved to {OUTPUT_PLOT}")
    plt.show()

if __name__ == "__main__":
    main()