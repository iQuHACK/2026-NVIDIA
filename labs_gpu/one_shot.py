import argparse
import numpy as np
import cudaq
import sys
import os

# -----------------------------------------------------------------------------
# 1. Imports & Dependency Check
# -----------------------------------------------------------------------------
try:
    import labs_utils as utils
except ImportError:
    try:
        import auxiliary_files.labs_utils as utils
    except ImportError:
        print("\n[Error] 'labs_utils.py' not found.")
        sys.exit(1)

# -----------------------------------------------------------------------------
# 2. Kernel Definitions
# -----------------------------------------------------------------------------

# --- Variant 1: Jenga (Pure Counter-Diabatic Impulse) ---
@cudaq.kernel
def kernel_jenga(n: int, indices: list[int], indices2: list[int], theta_list: list[float]):
    qubits = cudaq.qvector(n)
    for i in range(n):
        h(qubits[i])

    for t in range(len(theta_list)):
        # 2-Body CD Terms
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

        # 4-Body CD Terms
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

# --- Variant 2: DNA (Digitized Adiabatic + CD) ---
@cudaq.kernel
def kernel_dna(n: int, dt: float, indices: list[int], indices2: list[int], theta_list_CD: list[float], lamb_list_AD: list[float], CD: list[bool], AD: list[bool]):
    qubits = cudaq.qvector(n)
    for i in range(n):
        h(qubits[i])

    for t in range(len(theta_list_CD)):
        if CD[t] == True:
            # 2-Body CD
            for i in range(len(indices) // 2):
                i1 = indices[2*i]; i0 = indices[2*i + 1]
                rx(np.pi/2, qubits[i0]); cx(qubits[i0], qubits[i1]); rz(theta_list_CD[t], qubits[i1]); cx(qubits[i0], qubits[i1])
                rx(-np.pi/2, qubits[i0]); rx(np.pi/2, qubits[i1]); cx(qubits[i0], qubits[i1]); rz(theta_list_CD[t], qubits[i1]); cx(qubits[i0], qubits[i1]); rx(-np.pi/2, qubits[i1])
            # 4-Body CD
            for i in range(len(indices2) // 4):
                i0 = indices2[4*i]; i1 = indices2[4*i + 1]; i2 = indices2[4*i + 2]; i3 = indices2[4*i + 3]
                rx(-np.pi/2, qubits[i0]); ry(np.pi/2, qubits[i1]); ry(-np.pi/2, qubits[i2]); cx(qubits[i0], qubits[i1]); rz(-np.pi/2, qubits[i1]); cx(qubits[i0], qubits[i1])
                cx(qubits[i2], qubits[i3]); rz(-np.pi/2, qubits[i3]); cx(qubits[i2], qubits[i3]); rx(np.pi/2, qubits[i0]); ry(-np.pi/2, qubits[i1]); ry(np.pi/2, qubits[i2]); rx(-np.pi/2, qubits[i3])
                rx(-np.pi/2, qubits[i1]); rx(-np.pi/2, qubits[i2]); cx(qubits[i1], qubits[i2]); rz(theta_list_CD[t], qubits[i2]); cx(qubits[i1], qubits[i2])
                rx(np.pi/2, qubits[i1]); rx(np.pi, qubits[i2]); ry(np.pi/2, qubits[i1]); cx(qubits[i0], qubits[i1]); rz(np.pi/2, qubits[i1]); cx(qubits[i0], qubits[i1])
                rx(np.pi/2, qubits[i0]); ry(-np.pi/2, qubits[i1]); cx(qubits[i1], qubits[i2]); rz(-theta_list_CD[t], qubits[i2]); cx(qubits[i1], qubits[i2])
                rx(np.pi/2, qubits[i1]); rx(-np.pi, qubits[i2]); cx(qubits[i1], qubits[i2]); rz(-theta_list_CD[t], qubits[i2]); cx(qubits[i1], qubits[i2])
                rx(-np.pi, qubits[i1]); ry(np.pi/2, qubits[i2]); cx(qubits[i2], qubits[i3]); rz(-np.pi/2, qubits[i3]); cx(qubits[i2], qubits[i3])
                ry(-np.pi/2, qubits[i2]); rx(-np.pi/2, qubits[i3]); rx(-np.pi/2, qubits[i2]); cx(qubits[i1], qubits[i2]); rz(theta_list_CD[t], qubits[i2]); cx(qubits[i1], qubits[i2])
                rx(np.pi/2, qubits[i1]); rx(np.pi/2, qubits[i2]); ry(-np.pi/2, qubits[i1]); ry(np.pi/2, qubits[i2]); cx(qubits[i0], qubits[i1]); rz(np.pi/2, qubits[i1]); cx(qubits[i0], qubits[i1])
                cx(qubits[i2], qubits[i3]); rz(np.pi/2, qubits[i3]); cx(qubits[i2], qubits[i3]); ry(np.pi/2, qubits[i1]); ry(-np.pi/2, qubits[i2]); rx(np.pi/2, qubits[i3])

        if AD[t] == True:
            # Adiabatic Evolution (Mixer + Problem)
            for i in range(n):
                rx(2*dt - 2*lamb_list_AD[t]*dt, qubits[i])
            for i in range(len(indices) // 2):
                i1 = indices[2*i]; i0 = indices[2*i + 1]
                cx(qubits[i0], qubits[i1]); rz(4*lamb_list_AD[t]*dt, qubits[i1]); cx(qubits[i0], qubits[i1])
            for i in range(len(indices2) // 4):
                i0 = indices2[4*i]; i1 = indices2[4*i + 1]; i2 = indices2[4*i + 2]; i3 = indices2[4*i + 3]
                cx(qubits[i0],qubits[i1]); cx(qubits[i1],qubits[i2]); cx(qubits[i2],qubits[i3])
                rz(8*lamb_list_AD[t]*dt, qubits[i3])
                cx(qubits[i2],qubits[i3]); cx(qubits[i1],qubits[i2]); cx(qubits[i0],qubits[i1])

    for i in range(n//2):
        rx.ctrl(0.0, qubits[2*i+1], qubits[2*i])
    mz(qubits)

# --- Variant 3: Beyblade (Interleaved AD/CD) ---
@cudaq.kernel
def kernel_beyblade(n: int, dt: float, indices: list[int], indices2: list[int], theta_list_CD: list[float], lamb_list_AD: list[float], CD: list[bool], AD: list[bool]):
    qubits = cudaq.qvector(n)
    for i in range(n):
        h(qubits[i])

    for t in range(len(theta_list_CD)):
        # 1. Adiabatic Mixer First
        if AD[t] == True:
            for i in range(n):
                rx(2*dt - 2*lamb_list_AD[t]*dt, qubits[i])
        
        # 2. 2-Body Terms (Interleaved)
        for i in range(len(indices) // 2):
            i1 = indices[2*i]; i0 = indices[2*i + 1]
            if CD[t] == True:
                rx(np.pi/2, qubits[i0]); cx(qubits[i0], qubits[i1]); rz(theta_list_CD[t], qubits[i1]); cx(qubits[i0], qubits[i1])
                rx(-np.pi/2, qubits[i0]); rx(np.pi/2, qubits[i1]); cx(qubits[i0], qubits[i1]); rz(theta_list_CD[t], qubits[i1]); cx(qubits[i0], qubits[i1]); rx(-np.pi/2, qubits[i1])
            if AD[t] == True:
                cx(qubits[i0], qubits[i1]); rz(4*lamb_list_AD[t]*dt, qubits[i1]); cx(qubits[i0], qubits[i1])

        # 3. 4-Body Terms (Interleaved)
        for i in range(len(indices2) // 4):
            i0 = indices2[4*i]; i1 = indices2[4*i + 1]; i2 = indices2[4*i + 2]; i3 = indices2[4*i + 3]
            if CD[t] == True:
                rx(-np.pi/2, qubits[i0]); ry(np.pi/2, qubits[i1]); ry(-np.pi/2, qubits[i2]); cx(qubits[i0], qubits[i1]); rz(-np.pi/2, qubits[i1]); cx(qubits[i0], qubits[i1])
                cx(qubits[i2], qubits[i3]); rz(-np.pi/2, qubits[i3]); cx(qubits[i2], qubits[i3]); rx(np.pi/2, qubits[i0]); ry(-np.pi/2, qubits[i1]); ry(np.pi/2, qubits[i2]); rx(-np.pi/2, qubits[i3])
                rx(-np.pi/2, qubits[i1]); rx(-np.pi/2, qubits[i2]); cx(qubits[i1], qubits[i2]); rz(theta_list_CD[t], qubits[i2]); cx(qubits[i1], qubits[i2])
                rx(np.pi/2, qubits[i1]); rx(np.pi, qubits[i2]); ry(np.pi/2, qubits[i1]); cx(qubits[i0], qubits[i1]); rz(np.pi/2, qubits[i1]); cx(qubits[i0], qubits[i1])
                rx(np.pi/2, qubits[i0]); ry(-np.pi/2, qubits[i1]); cx(qubits[i1], qubits[i2]); rz(-theta_list_CD[t], qubits[i2]); cx(qubits[i1], qubits[i2])
                rx(np.pi/2, qubits[i1]); rx(-np.pi, qubits[i2]); cx(qubits[i1], qubits[i2]); rz(-theta_list_CD[t], qubits[i2]); cx(qubits[i1], qubits[i2])
                rx(-np.pi, qubits[i1]); ry(np.pi/2, qubits[i2]); cx(qubits[i2], qubits[i3]); rz(-np.pi/2, qubits[i3]); cx(qubits[i2], qubits[i3])
                ry(-np.pi/2, qubits[i2]); rx(-np.pi/2, qubits[i3]); rx(-np.pi/2, qubits[i2]); cx(qubits[i1], qubits[i2]); rz(theta_list_CD[t], qubits[i2]); cx(qubits[i1], qubits[i2])
                rx(np.pi/2, qubits[i1]); rx(np.pi/2, qubits[i2]); ry(-np.pi/2, qubits[i1]); ry(np.pi/2, qubits[i2]); cx(qubits[i0], qubits[i1]); rz(np.pi/2, qubits[i1]); cx(qubits[i0], qubits[i1])
                cx(qubits[i2], qubits[i3]); rz(np.pi/2, qubits[i3]); cx(qubits[i2], qubits[i3]); ry(np.pi/2, qubits[i1]); ry(-np.pi/2, qubits[i2]); rx(np.pi/2, qubits[i3])
            if AD[t] == True:
                cx(qubits[i0],qubits[i1]); cx(qubits[i1],qubits[i2]); cx(qubits[i2],qubits[i3])
                rz(8*lamb_list_AD[t]*dt, qubits[i3])
                cx(qubits[i2],qubits[i3]); cx(qubits[i1],qubits[i2]); cx(qubits[i0],qubits[i1])

    for i in range(n//2):
        rx.ctrl(0.0, qubits[2*i+1], qubits[2*i])
    mz(qubits)

# -----------------------------------------------------------------------------
# 3. Helpers (Interactions, Schedules, Export)
# -----------------------------------------------------------------------------
def get_interactions(n):
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
                if set1: full.append(set1)
    unique = [set(s) for s in set(frozenset(s) for s in full)]
    list_pairs = [list(i) for i in unique]
    G2, G4, list_2, list_4 = [], [], [], []
    for i in list_pairs:
        if len(i) == 2:
            list_2 += [i[0], i[1]]; G2.append(i)
        else:
            list_4 += [i[0], i[1], i[2], i[3]]; G4.append(i)
    return G2, G4, list_2, list_4

def compute_adiabatic_schedule(n_steps):
    """Generates the lambda schedule for adiabatic terms."""
    # Standard Sin^2 schedule often used in CD/QAOA literature
    t_vals = np.linspace(0, 1, n_steps + 1)[1:] # Evaluate at end of step
    lambdas = np.sin((np.pi/2) * (np.sin(np.pi * t_vals / 2)**2))**2
    return lambdas.tolist()

def export_ready_for_cuda(quantum_population, n, filename, gpu_pop_size=8192, gpu_max_n=512):
    print(f"--> Exporting {len(quantum_population)} samples to '{filename}' (Target Size: {gpu_pop_size})...")
    host_buffer = np.random.choice([-1, 1], size=(gpu_pop_size, gpu_max_n)).astype(np.int8)
    num_samples = quantum_population.shape[0]
    rows_to_fill = min(num_samples, gpu_pop_size)
    host_buffer[:rows_to_fill, :n] = quantum_population[:rows_to_fill].astype(np.int8)
    host_buffer.tofile(filename)

# -----------------------------------------------------------------------------
# 4. Simulation Controller (SINGLE RUN ONLY)
# -----------------------------------------------------------------------------
def run_simulation(n, shots, pop_size, output_file, variant, steps):
    print(f"\n=== Quantum LABS Simulation ({variant.upper()}) ===")
    print(f"N={n}, Shots={shots}, Steps={steps}")

    G2, G4, list_2, list_4 = get_interactions(n)
    T = 1.0
    dt = T / steps
    
    # 1. Physics Calculations
    thetas_CD = []
    print("Computing Counter-Diabatic (CD) angles...")
    for step in range(1, steps + 1):
        t = step * dt
        theta_val = utils.compute_theta(t, dt, T, n, G2, G4)
        thetas_CD.append(theta_val)

    lambdas_AD = compute_adiabatic_schedule(steps)
    
    # Default toggles (Enable everything)
    CD_toggle = [True] * steps
    AD_toggle = [True] * steps

    # 2. Kernel Selection & Single Shot Execution
    print(f"Running Kernel: {variant} (Single Run)...")
    
    if variant == 'jenga':
        result = cudaq.sample(kernel_jenga, n, list_2, list_4, thetas_CD, shots_count=shots)
    elif variant == 'dna':
        result = cudaq.sample(kernel_dna, n, dt, list_2, list_4, thetas_CD, lambdas_AD, CD_toggle, AD_toggle, shots_count=shots)
    elif variant == 'beyblade':
        result = cudaq.sample(kernel_beyblade, n, dt, list_2, list_4, thetas_CD, lambdas_AD, CD_toggle, AD_toggle, shots_count=shots)
    
    # 3. Extract Top K Bitstrings
    # Sort dictionary items by count (descending) and take top `pop_size`
    top_strings = [bs for bs, count in sorted(result.items(), key=lambda x: x[1], reverse=True)[:pop_size]]
    
    print(f"Collected {len(top_strings)} elite candidates from {shots} shots.")

    # 4. Convert & Export
    pop_list = []
    for s in top_strings:
        arr = np.array([int(b) for b in s])
        pm1 = (2 * arr - 1).astype(np.int8)
        pop_list.append(pm1)
        
    initial_pop = np.array(pop_list)
    export_ready_for_cuda(initial_pop, n, output_file, gpu_pop_size=pop_size)
    print("=== Complete ===\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Quantum Warm Start")
    parser.add_argument("--n", type=int, required=True, help="Sequence Length")
    parser.add_argument("--shots", type=int, default=100000)
    parser.add_argument("--popsize", type=int, default=8192, help="Number of elites to extract")
    parser.add_argument("--steps", type=int, default=1, help="Trotter steps")
    parser.add_argument("--output", type=str, default="warm_start.bin")
    parser.add_argument("--variant", type=str, choices=['jenga', 'dna', 'beyblade'], default='jenga', 
                        help="Quantum Circuit Variant: 'jenga' (Pure CD), 'dna' (Digitized Adiabatic), 'beyblade' (Interleaved)")
    
    args = parser.parse_args()
    
    # Notice: 'runs' argument is gone.
    run_simulation(args.n, args.shots, args.popsize, args.output, args.variant, args.steps)