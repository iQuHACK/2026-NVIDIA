import time
import numpy as np
import matplotlib.pyplot as plt
from enhanced_labs_implementation import get_interactions
from utils import calculate_labs_energy
import json


def benchmark_scaling(N_values, n_trials=10):
    """Benchmark CPU performance vs N."""
    results = {
        'N': [],
        'time_cpu': [],
        'time_cpu_std': [],
        'num_interactions_G2': [],
        'num_interactions_G4': []
    }

    for N in N_values:
        print(f"Benchmarking N={N}...")

        # Generate interactions
        G2, G4 = get_interactions(N)
        results['num_interactions_G2'].append(len(G2))
        results['num_interactions_G4'].append(len(G4))

        # Benchmark CPU
        times = []
        for _ in range(n_trials):
            seq = [np.random.randint(0, 2) for _ in range(N)]
            start = time.time()
            energy = calculate_labs_energy(seq, N)
            times.append(time.time() - start)

        results['N'].append(N)
        results['time_cpu'].append(np.mean(times))
        results['time_cpu_std'].append(np.std(times))

    return results


def plot_results(results):
    """Generate publication-quality plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Time vs N
    ax1 = axes[0]
    ax1.errorbar(results['N'], results['time_cpu'],
                 yerr=results['time_cpu_std'],
                 marker='o', label='CPU', capsize=5)
    ax1.set_xlabel('Problem Size (N)', fontsize=12)
    ax1.set_ylabel('Computation Time (s)', fontsize=12)
    ax1.set_title('LABS Energy Computation Scaling', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Interactions vs N
    ax2 = axes[1]
    ax2.plot(results['N'], results['num_interactions_G2'],
             marker='s', label='G2 (2-body)')
    ax2.plot(results['N'], results['num_interactions_G4'],
             marker='^', label='G4 (4-body)')
    ax2.set_xlabel('Problem Size (N)', fontsize=12)
    ax2.set_ylabel('Number of Interactions', fontsize=12)
    ax2.set_title('Interaction Complexity Growth', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    N_values = [10, 15, 20, 25, 30, 35, 40]
    results = benchmark_scaling(N_values)

    # Save data
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    plot_results(results)