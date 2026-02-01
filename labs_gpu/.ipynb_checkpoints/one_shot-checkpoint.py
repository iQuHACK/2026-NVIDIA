import argparse
import numpy as np
import cudaq
import sys
import os

# -----------------------------------------------------------------------------
# 1. Imports & Dependency Check
# -----------------------------------------------------------------------------
try:
    # Try importing if file is in the same directory
    import labs_utils as utils
except ImportError:
    try:
        # Try importing if file is in the original folder structure
        import auxiliary_files.labs_utils as utils
    except ImportError:
        print("\n[Error] 'labs_utils.py' not found.")
        print("Please ensure 'labs_utils.py' is in the same directory or 'auxiliary_files/'.")
        print("This file is required for the 'compute_theta' physics calculations.\n")
        sys.exit(1)

# -----------------------------------------------------------------------------
# 2. CUDA-Q Kernel Definition (The Quantum Circuit)
# -----------------------------------------------------------------------------
@cudaq.kernel
def qc(n: int, indices: list[int], indices2: list[int], theta_list: list[float]):
    qubits = cudaq.qvector(n)
    
    # Initialize in superposition
    for i in range(n):
        h(qubits[i])

    # Apply Trotter Steps
    for t in range(len(theta_list)):
        # Apply 2-Body Terms
        for i in range(len(indices) // 2):
            i1 = indices[2*i]
            i0 = indices[2*i + 1]
    
            rx(np.pi/2, qubits[i0])
            cx(qubits[i0], qubits[i1])
            rz(theta_list[t], qubits[i1])
            cx(qubits[i0], qubits[i1])
            rx(-np.pi/2, qubits[i0])
            rx(np.pi/2, qubits[i1])
            cx(qubits[i0], qubits[i1])
            rz(theta_list[t], qubits[i1])
            cx(qubits[i0], qubits[i1])
            rx(-np.pi/2, qubits[i1])
    
        # Apply 4-Body Terms
        for i in range(len(indices2) // 4):
            i0 = indices2[4*i]
            i1 = indices2[4*i + 1]
            i2 = indices2[4*i + 2]
            i3 = indices2[4*i + 3]
    
            rx(-np.pi/2, qubits[i0])
            ry(np.pi/2, qubits[i1])
            ry(-np.pi/2, qubits[i2])
            cx(qubits[i0], qubits[i1])
            rz(-np.pi/2, qubits[i1])
            cx(qubits[i0], qubits[i1])
            cx(qubits[i2], qubits[i3])
            rz(-np.pi/2, qubits[i3])
            cx(qubits[i2], qubits[i3])
            rx(np.pi/2, qubits[i0])
            ry(-np.pi/2, qubits[i1])
            ry(np.pi/2, qubits[i2])
            rx(-np.pi/2, qubits[i3])
            rx(-np.pi/2, qubits[i1])
            rx(-np.pi/2, qubits[i2])
            cx(qubits[i1], qubits[i2])
            rz(theta_list[t], qubits[i2])
            cx(qubits[i1], qubits[i2])
            rx(np.pi/2, qubits[i1])
            rx(np.pi, qubits[i2])
            ry(np.pi/2, qubits[i1])
            cx(qubits[i0], qubits[i1])
            rz(np.pi/2, qubits[i1])
            cx(qubits[i0], qubits[i1])
            rx(np.pi/2, qubits[i0])
            ry(-np.pi/2, qubits[i1])
            cx(qubits[i1], qubits[i2])
            rz(-theta_list[t], qubits[i2])
            cx(qubits[i1], qubits[i2])
            rx(np.pi/2, qubits[i1])
            rx(-np.pi, qubits[i2])
            cx(qubits[i1], qubits[i2])
            rz(-theta_list[t], qubits[i2])
            cx(qubits[i1], qubits[i2])
            rx(-np.pi, qubits[i1])
            ry(np.pi/2, qubits[i2])
            cx(qubits[i2], qubits[i3])
            rz(-np.pi/2, qubits[i3])
            cx(qubits[i2], qubits[i3])
            ry(-np.pi/2, qubits[i2])
            rx(-np.pi/2, qubits[i3])
            rx(-np.pi/2, qubits[i2])
            cx(qubits[i1], qubits[i2])
            rz(theta_list[t], qubits[i2])
            cx(qubits[i1], qubits[i2])
            rx(np.pi/2, qubits[i1])
            rx(np.pi/2, qubits[i2])
            ry(-np.pi/2, qubits[i1])
            ry(np.pi/2, qubits[i2])
            cx(qubits[i0], qubits[i1])
            rz(np.pi/2, qubits[i1])
            cx(qubits[i0], qubits[i1])
            cx(qubits[i2], qubits[i3])
            rz(np.pi/2, qubits[i3])
            cx(qubits[i2], qubits[i3])
            ry(np.pi/2, qubits[i1])
            ry(-np.pi/2, qubits[i2])
            rx(np.pi/2, qubits[i3])

    mz(qubits)

# -----------------------------------------------------------------------------
# 3. Helper Functions (Interactions & Exports)
# -----------------------------------------------------------------------------
def get_interactions(n):
    """Generates the interaction sets G2 and G4 based on problem size N."""
    pairs = []
    for i in range(1, n):
        square = [(i+j, j) for j in range(n-i)]
        pairs.append(square)

    full = []
    for x in range(len(pairs)):
        test = pairs[x]
        for i in range(len(test)):
            for j in range(len(test)):
                list1 = [test[i][0], test[i][1], test[j][0], test[j][1]]
                set1 = {w for w in list1 if list1.count(w) == 1}
                if set1:
                    full.append(set1)

    unique = [set(s) for s in set(frozenset(s) for s in full)]
    list_pairs = [list(i) for i in unique]

    list_2, list_4, G2, G4 = [], [], [], []
    for i in list_pairs:
        if len(i) == 2:
            list_2 += [i[0], i[1]]
            G2.append(i)
        else:
            list_4 += [i[0], i[1], i[2], i[3]]
            G4.append(i)
            
    return G2, G4, list_2, list_4

def export_ready_for_cuda(quantum_population, n, filename, gpu_pop_size=8192, gpu_max_n=512):
    """Exports data to a binary file formatted for the C++ Solver."""
    print(f"--> Formatting data for C++ (Pop: {gpu_pop_size}, MaxN: {gpu_max_n})...")
    host_buffer = np.random.choice([-1, 1], size=(gpu_pop_size, gpu_max_n)).astype(np.int8)
    
    num_samples = quantum_population.shape[0]
    rows_to_fill = min(num_samples, gpu_pop_size)
    
    # Inject quantum data into the buffer
    host_buffer[:rows_to_fill, :n] = quantum_population[:rows_to_fill].astype(np.int8)
    
    host_buffer.tofile(filename)
    print(f"--> Saved '{filename}' ({host_buffer.nbytes/1024**2:.2f} MB)")

# -----------------------------------------------------------------------------
# 4. Main Simulation Logic
# -----------------------------------------------------------------------------
def run_simulation(n, shots, runs, pop_size, output_file):
    print(f"\n=== Starting Quantum LABS Simulation ===")
    print(f"Parameters: N={n}, Shots={shots}, Runs={runs}, Target Pop={pop_size}")
    
    # 1. Setup Physics
    G2, G4, list_2, list_4 = get_interactions(n)
    
    T = 1
    n_steps = 1
    dt = T / n_steps
    thetas = []
    
    print("Computing annealing angles...")
    for step in range(1, n_steps + 1):
        t = step * dt
        # Using the imported utils file
        theta_val = utils.compute_theta(t, dt, T, n, G2, G4)
        thetas.append(theta_val)

    # 2. Run Quantum Sampling
    print("Running CUDA-Q sampling...")
    aggregated_strings = []
    
    for i in range(runs):
        result = cudaq.sample(qc, n, list_2, list_4, thetas, shots_count=shots)
        
        # We take the top results to fill our population needs
        # We grab enough to fill the population if possible
        samples_needed = int(pop_size / runs) + 50 
        
        top_strings = [
            bs for bs, count in sorted(result.items(), key=lambda x: x[1], reverse=True)[:samples_needed]
        ]
        aggregated_strings.extend(top_strings)
        print(f"  Run {i+1}/{runs}: Collected {len(top_strings)} candidates.")

    # 3. Process Data
    print(f"Processing {len(aggregated_strings)} total candidates...")
    
    # Convert "01" strings to +/- 1 integers
    # Assumption: "1" maps to 1, "0" maps to -1 (or vice versa, LABS is symmetric)
    # We use: 1 -> 1, 0 -> -1
    pop_list = []
    for s in aggregated_strings:
        arr = np.array([int(b) for b in s])
        pm1 = (2 * arr - 1).astype(np.int8)
        pop_list.append(pm1)
        
    initial_pop = np.array(pop_list)

    # 4. Export
    export_ready_for_cuda(initial_pop, n, output_file, gpu_pop_size=pop_size)
    print("=== Generation Complete ===\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Quantum Warm Start for LABS")
    parser.add_argument("--n", type=int, required=True, help="Problem size (Sequence Length)")
    parser.add_argument("--shots", type=int, default=100000, help="Shots per quantum run")
    parser.add_argument("--runs", type=int, default=10, help="Number of independent quantum runs")
    parser.add_argument("--popsize", type=int, default=8192, help="Target population size for C++")
    parser.add_argument("--output", type=str, default="warm_start.bin", help="Output filename")
    
    args = parser.parse_args()
    
    run_simulation(args.n, args.shots, args.runs, args.popsize, args.output)