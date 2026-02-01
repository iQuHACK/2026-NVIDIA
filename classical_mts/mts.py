# slower one

import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import cupy as cp

class LabsMTS_GPU:
    def __init__(self, n_bits, n_agents, max_iter):
        """
        Initialize the GPU-accelerated MTS solver for the LABS problem.
        """
        self.n_bits = n_bits
        self.n_agents = n_agents
        self.max_iter = max_iter

    def calculate_labs_energy_batch(self, population):
        """
        GPU Optimized Batch Energy Calculation.
        Computes the LABS energy (sum of squared aperiodic autocorrelations) 
        for the entire population in parallel.
        
        Input: population (n_agents, n_bits) [0, 1] on GPU
        Output: energy (n_agents,)
        """
        # Map binary sequence (0, 1) to bipolar sequence (-1, +1)
        # cp.array operations are automatically parallelized on the GPU
        s = 2 * population - 1 
        
        n_agents, n_bits = s.shape
        energies = cp.zeros(n_agents, dtype=cp.float32)
        
        # Vectorized calculation over lags 'k'
        # We iterate through lags, but the calculation for each lag 
        # is performed simultaneously across all agents (N_AGENTS parallel).
        for k in range(1, n_bits):
            # Efficient slicing to compute autocorrelation:
            # s[:, :-k] -> The sequence from index 0 to N-k-1
            # s[:, k:]  -> The sequence from index k to N-1
            # axis=1    -> Sum along the bit sequence dimension to get C_k
            c_k = cp.sum(s[:, :-k] * s[:, k:], axis=1)
            
            # Accumulate the square of the autocorrelation
            energies += c_k**2
            
        return energies

    def get_neighbor_batch(self, population, perturbation_strength=1):
        """
        Generate neighbor solutions for the entire population simultaneously on GPU.
        """
        n_agents, n_bits = population.shape
        neighbor = population.copy()
        
        if perturbation_strength == 1:
            # Strategy: Flip exactly 1 random bit per agent
            # Generate one random index per agent
            flip_indices = cp.random.randint(0, n_bits, size=(n_agents, 1))
            
            # Use advanced indexing to flip the bits at selected indices
            rows = cp.arange(n_agents).reshape(-1, 1)
            neighbor[rows, flip_indices] = 1 - neighbor[rows, flip_indices]
            
        else:
            # Strategy: Flip 'k' unique bits per agent (Wide Search)
            # Since random.choice without replacement is hard to vectorize efficiently,
            # we use the argsort trick on a random noise matrix.
            
            # 1. Generate a random noise matrix of the same shape as population
            noise = cp.random.rand(n_agents, n_bits)
            
            # 2. Get indices of the top 'k' values in the noise matrix
            # This guarantees 'perturbation_strength' unique indices per row
            flip_indices = cp.argsort(noise, axis=1)[:, -perturbation_strength:]
            
            # 3. Flip the bits at these indices using advanced indexing
            rows = cp.arange(n_agents).reshape(-1, 1)
            neighbor[rows, flip_indices] = 1 - neighbor[rows, flip_indices]

        return neighbor

    def run(self):
        # --- 1. Initialization Phase ---
        print(f"Starting MTS (GPU-CuPy) for LABS (N={self.n_bits}) with {self.n_agents} agents...")
        
        # Generate random initial population directly on GPU memory
        population = cp.random.randint(2, size=(self.n_agents, self.n_bits), dtype=cp.int8)
        
        # Calculate initial energies for the entire batch
        energies = self.calculate_labs_energy_batch(population)
        
        # Track the Global Best Solution
        best_idx = cp.argmin(energies)
        global_best_solution = population[best_idx].copy()
        global_best_energy = energies[best_idx]
        
        history_best_energy = []
        
        # Transfer scalar value to CPU for logging (inexpensive)
        current_global_best_cpu = float(global_best_energy)
        print(f"Initial Best Energy: {current_global_best_cpu}")

        # --- 2. Main Optimization Loop ---
        start_time = time.time()
        
        # Pre-calculate flip count for Strategy 2 (approx 5% of bits)
        strength_2 = max(2, int(self.n_bits * 0.05))

        for it in range(self.max_iter):
            # --- MTS Search Strategy (Fully Vectorized) ---
            
            # Strategy 1: Local Fine Search
            # Apply a 1-bit flip to ALL agents
            candidate_1 = self.get_neighbor_batch(population, perturbation_strength=1)
            e1 = self.calculate_labs_energy_batch(candidate_1)
            
            # Create a boolean mask identifying agents that improved
            improved_mask = e1 < energies
            
            # Update population and energies ONLY for agents that improved
            # Boolean indexing allows updating specific rows in the GPU matrix
            population[improved_mask] = candidate_1[improved_mask]
            energies[improved_mask] = e1[improved_mask]
            
            # Strategy 2: Wide Search (Escaping Local Optima)
            # Identify agents that did NOT improve in Strategy 1
            not_improved_mask = ~improved_mask
            
            # If there are any agents stuck, apply Strategy 2
            if cp.any(not_improved_mask):
                # Generate candidates with larger perturbation
                # Note: We generate for all, but only use the ones needed to keep matrix shapes consistent
                candidate_2 = self.get_neighbor_batch(population, perturbation_strength=strength_2)
                e2 = self.calculate_labs_energy_batch(candidate_2)
                
                # Update only if:
                # 1. The new candidate is better than the current state
                # 2. The agent was in the "not improved" group
                improved_s2_mask = (e2 < energies) & not_improved_mask
                
                population[improved_s2_mask] = candidate_2[improved_s2_mask]
                energies[improved_s2_mask] = e2[improved_s2_mask]

            # --- Update Global Best ---
            # Find the best agent in the current generation
            current_best_idx = cp.argmin(energies)
            current_min_energy = energies[current_best_idx]
            
            if current_min_energy < global_best_energy:
                global_best_energy = current_min_energy
                global_best_solution = population[current_best_idx].copy()
                current_global_best_cpu = float(global_best_energy) # Scalar copy to CPU
                print(f"Iter {it}: New Best Energy found: {current_global_best_cpu}")
            
            history_best_energy.append(current_global_best_cpu)

        # Ensure all GPU tasks are finished before stopping the timer
        cp.cuda.Stream.null.synchronize()
        total_time = time.time() - start_time
        print(f"Optimization finished in {total_time:.2f} seconds.")
        
        # --- 3. Final Data Transfer ---
        # Transfer results back to CPU (NumPy) for visualization
        return (cp.asnumpy(global_best_solution), 
                float(global_best_energy), 
                cp.asnumpy(energies), 
                history_best_energy)

# Visualization code (Runs on CPU)
def visualize_results(final_energies, history, best_sol, best_energy):
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)

    # Plot 1: Energy Distribution of Final Population
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(final_energies, bins=15, kde=True, ax=ax1, color='teal')
    ax1.set_title('Final Population Energy Distribution')
    ax1.set_xlabel('LABS Energy')
    
    # Plot 2: Convergence History
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history, linewidth=2, color='firebrick')
    ax2.set_title('Optimization Convergence')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best Energy')
    ax2.grid(True)
    
    # Plot 3: Autocorrelation Side-lobes
    ax3 = fig.add_subplot(gs[1, :])
    s = 2 * best_sol - 1
    N = len(s)
    lags = np.arange(1, N)
    correlations = []
    for k in lags:
        correlations.append(np.sum(s[:-k] * s[k:]))
        
    ax3.bar(lags, correlations, color='navy')
    ax3.set_title(f'Autocorrelation Side-lobes (Energy={best_energy:.0f})')
    ax3.set_xlabel('Lag (k)')
    ax3.axhline(0, color='black', linewidth=1)

    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Configuration for LABS Problem
    # With GPU parallelization, we can handle a massive number of agents efficiently.
    N_BITS = 38        
    N_AGENTS = 10000000      # 10 Million agents to leverage GPU parallelism
    MAX_ITER = 100    

    mts_gpu = LabsMTS_GPU(N_BITS, N_AGENTS, MAX_ITER)
    
    # Run the optimization and retrieve results as NumPy arrays
    best_sol, best_energy, final_pop_energies, history = mts_gpu.run()

    # Output Results
    print("-" * 30)
    print(f"LABS Solution (N={N_BITS}) - GPU Result")
    print(f"Best Energy Found: {best_energy}")
    
    display_seq = 2 * best_sol - 1
    seq_str = ''.join(['+' if x == 1 else '-' for x in display_seq])
    print(f"Sequence Pattern: {seq_str}")
    
    visualize_results(final_pop_energies, history, best_sol, best_energy)