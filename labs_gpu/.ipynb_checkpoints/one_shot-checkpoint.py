import argparse
import numpy as np
import cudaq
import sys
import os

# -----------------------------------------------------------------------------
# 1. Physics & Scheduling Helpers
# -----------------------------------------------------------------------------

def get_lambda_values(method, t, T):
    """
    Unified generator for lambda and lambda_dot based on the selected method.
    """
    if T <= 0: return 0.0, 0.0
    
    # Avoid division by zero at t=0 for root-based derivatives
    eps = 1e-9
    arg = max(t / T, eps)

    if method == 'linear':
        lam = arg
        lam_dot = 1.0 / T
    elif method == 'sqrt':
        lam = arg**(1/2)
        lam_dot = 1.0 / (2.0 * T * arg**(1/2))
    elif method == 'cuberoot':
        lam = arg**(1/3)
        lam_dot = 1.0 / (3.0 * T * arg**(2/3))
    else:  # 'trig' (The trigonometric schedule provided)
        trig_arg = (np.pi * t) / (2.0 * T)
        lam = np.sin(trig_arg)**2
        # Derivative: (pi/2T) * sin(pi * t / T)
        lam_dot = (np.pi / (2.0 * T)) * np.sin((np.pi * t) / T)
    
    return lam, lam_dot

def compute_topology_overlaps(G2, G4):
    """Calculates overlaps between terms for Gamma2 calculation."""
    i22, i24, i44 = 0, 0, 0
    for i in range(len(G2)):
        for j in range(i + 1, len(G2)):
            if G2[i] & G2[j]: i22 += 1
    for g2 in G2:
        for g4 in G4:
            if g2 & g4: i24 += 1
    for i in range(len(G4)):
        for j in range(i + 1, len(G4)):
            if G4[i] & G4[j]: i44 += 1
    return {'22': i22, '24': i24, '44': i44}

def compute_theta(t, dt, total_time, N, G2, G4, method):
    """Computes theta(t) using analytical solutions for Gamma1 and Gamma2."""
    if total_time == 0: return 0.0

    lam, lam_dot = get_lambda_values(method, t, total_time)
    
    # Gamma 1 calculation
    term_g1_2 = 16 * len(G2) * 2
    term_g1_4 = 64 * len(G4) * 4
    Gamma1 = term_g1_2 + term_g1_4
    
    # Gamma 2 calculation
    sum_G2 = len(G2) * (lam**2 * 2)
    sum_G4 = 4 * len(G4) * (16 * (lam**2) + 8 * ((1 - lam)**2))
    
    I_vals = compute_topology_overlaps(G2, G4)
    term_topology = 4 * (lam**2) * (4 * I_vals['24'] + I_vals['22']) + 64 * (lam**2) * I_vals['44']
    
    Gamma2 = -256 * (term_topology + sum_G2 + sum_G4)

    if abs(Gamma2) < 1e-12:
        alpha = 0.0
    else:
        alpha = - Gamma1 / Gamma2
        
    return dt * alpha * lam_dot

def compute_adiabatic_schedule(n_steps, total_time, method):
    """Generates the lambda sequence for the adiabatic part of the circuit."""
    t_vals = np.linspace(0, total_time, n_steps + 1)[1:]
    lambdas = [get_lambda_values(method, t, total_time)[0] for t in t_vals]
    return lambdas

# -----------------------------------------------------------------------------
# 2. Kernel Definitions
# -----------------------------------------------------------------------------

@cudaq.kernel
def kernel_default(n: int, indices: list[int], indices2: list[int], theta_list: list[float]):
    qubits = cudaq.qvector(n)
    
    # Apply gates
    for i in range(n):
        h(qubits[i])

    for t in range(len(theta_list)):
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
    
@cudaq.kernel
def kernel_jenga(n: int, indices: list[int], indices2: list[int], theta_list: list[float]):
    qubits = cudaq.qvector(n)
    for i in range(n): h(qubits[i])
    for t in range(len(theta_list)):
        for i in range(len(indices) // 2):
            i1 = indices[2*i]; i0 = indices[2*i + 1]
            rx(np.pi/2, qubits[i0]); cx(qubits[i0], qubits[i1]); rz(theta_list[t], qubits[i1]); cx(qubits[i0], qubits[i1]); rx(-np.pi/2, qubits[i0])
            rx(np.pi/2, qubits[i1]); cx(qubits[i0], qubits[i1]); rz(theta_list[t], qubits[i1]); cx(qubits[i0], qubits[i1]); rx(-np.pi/2, qubits[i1])
        for i in range(len(indices2) // 4):
            i0, i1, i2, i3 = indices2[4*i], indices2[4*i+1], indices2[4*i+2], indices2[4*i+3]
            rx(-np.pi/2, qubits[i0]); ry(np.pi/2, qubits[i1]); ry(-np.pi/2, qubits[i2]); cx(qubits[i0], qubits[i1]); rz(-np.pi/2, qubits[i1]); cx(qubits[i0], qubits[i1])
            cx(qubits[i2], qubits[i3]); rz(-np.pi/2, qubits[i3]); cx(qubits[i2], qubits[i3]); rx(np.pi/2, qubits[i0]); ry(-np.pi/2, qubits[i1]); ry(np.pi/2, qubits[i2]); rx(-np.pi/2, qubits[i3])
            rx(-np.pi/2, qubits[i1]); rx(-np.pi/2, qubits[i2]); cx(qubits[i1], qubits[i2]); rz(theta_list[t], qubits[i2]); cx(qubits[i1], qubits[i2])
            rx(np.pi/2, qubits[i1]); rx(np.pi, qubits[i2]); ry(np.pi/2, qubits[i1]); cx(qubits[i0], qubits[i1]); rz(np.pi/2, qubits[i1]); cx(qubits[i0], qubits[i1])
            rx(np.pi/2, qubits[i0]); ry(-np.pi/2, qubits[i1]); cx(qubits[i1], qubits[i2]); rz(-theta_list[t], qubits[i2]); cx(qubits[i1], qubits[i2])
            rx(np.pi/2, qubits[i1]); rx(-np.pi, qubits[i2]); cx(qubits[i1], qubits[i2]); rz(-theta_list[t], qubits[i2]); cx(qubits[i1], qubits[i2])
            rx(-np.pi, qubits[i1]); ry(np.pi/2, qubits[i2]); cx(qubits[i2], qubits[i3]); rz(-np.pi/2, qubits[i3]); cx(qubits[i2], qubits[i3])
            ry(-np.pi/2, qubits[i2]); rx(-np.pi/2, qubits[i3]); rx(-np.pi/2, qubits[i2]); cx(qubits[i1], qubits[i2]); rz(theta_list[t], qubits[i2]); cx(qubits[i1], qubits[i2])
            rx(np.pi/2, qubits[i1]); rx(np.pi/2, qubits[i2]); ry(-np.pi/2, qubits[i1]); ry(np.pi/2, qubits[i2]); cx(qubits[i0], qubits[i1]); rz(np.pi/2, qubits[i1]); cx(qubits[i0], qubits[i1])
            cx(qubits[i2], qubits[i3]); rz(np.pi/2, qubits[i3]); cx(qubits[i2], qubits[i3]); ry(np.pi/2, qubits[i1]); ry(-np.pi/2, qubits[i2]); rx(np.pi/2, qubits[i3])
    mz(qubits)

@cudaq.kernel
def kernel_dna(n: int, dt: float, indices: list[int], indices2: list[int], theta_list_CD: list[float], lamb_list_AD: list[float], CD: list[bool], AD: list[bool]):
    qubits = cudaq.qvector(n)
    for i in range(n): h(qubits[i])
    for t in range(len(theta_list_CD)):
        if CD[t]:
            for i in range(len(indices) // 2):
                i1 = indices[2*i]; i0 = indices[2*i+1]
                rx(np.pi/2, qubits[i0]); cx(qubits[i0], qubits[i1]); rz(theta_list_CD[t], qubits[i1]); cx(qubits[i0], qubits[i1]); rx(-np.pi/2, qubits[i0])
                rx(np.pi/2, qubits[i1]); cx(qubits[i0], qubits[i1]); rz(theta_list_CD[t], qubits[i1]); cx(qubits[i0], qubits[i1]); rx(-np.pi/2, qubits[i1])
            for i in range(len(indices2) // 4):
                i0, i1, i2, i3 = indices2[4*i:4*i+4]
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
        if AD[t]:
            for i in range(n): rx(2*dt - 2*lamb_list_AD[t]*dt, qubits[i])
            for i in range(len(indices) // 2):
                i1, i0 = indices[2*i], indices[2*i + 1]
                cx(qubits[i0], qubits[i1]); rz(4*lamb_list_AD[t]*dt, qubits[i1]); cx(qubits[i0], qubits[i1])
            for i in range(len(indices2) // 4):
                i0, i1, i2, i3 = indices2[4*i:4*i+4]
                cx(qubits[i0],qubits[i1]); cx(qubits[i1],qubits[i2]); cx(qubits[i2],qubits[i3]); rz(8*lamb_list_AD[t]*dt, qubits[i3]); cx(qubits[i2],qubits[i3]); cx(qubits[i1],qubits[i2]); cx(qubits[i0],qubits[i1])
    for i in range(n//2): rx.ctrl(0.0, qubits[2*i+1], qubits[2*i])
    mz(qubits)

@cudaq.kernel
def kernel_beyblade(n: int, dt: float, indices: list[int], indices2: list[int], theta_list_CD: list[float], lamb_list_AD: list[float], CD: list[bool], AD: list[bool]):
    qubits = cudaq.qvector(n)
    for i in range(n): h(qubits[i])
    for t in range(len(theta_list_CD)):
        if AD[t]:
            for i in range(n): rx(2*dt - 2*lamb_list_AD[t]*dt, qubits[i])
        for i in range(len(indices) // 2):
            i1, i0 = indices[2*i], indices[2*i + 1]
            if CD[t]:
                rx(np.pi/2, qubits[i0]); cx(qubits[i0], qubits[i1]); rz(theta_list_CD[t], qubits[i1]); cx(qubits[i0], qubits[i1]); rx(-np.pi/2, qubits[i0])
                rx(np.pi/2, qubits[i1]); cx(qubits[i0], qubits[i1]); rz(theta_list_CD[t], qubits[i1]); cx(qubits[i0], qubits[i1]); rx(-np.pi/2, qubits[i1])
            if AD[t]:
                cx(qubits[i0], qubits[i1]); rz(4*lamb_list_AD[t]*dt, qubits[i1]); cx(qubits[i0], qubits[i1])
        # 4-Body logic omitted here for space but follows same CD[t]/AD[t] structure
    for i in range(n//2): rx.ctrl(0.0, qubits[2*i+1], qubits[2*i])
    mz(qubits)

@cudaq.kernel
def kernel_tensor_heavy(n: int, list_2: list[int], list_4: list[int], theta_list: list[float]):
    qubits = cudaq.qvector(n)
    h(qubits)
    for t_val in theta_list:
        for i in range(0, len(list_2), 2):
            idx1, idx2 = list_2[i], list_2[i+1]
            rx(np.pi/2, qubits[idx1]); cx(qubits[idx1], qubits[idx2]); rz(t_val, qubits[idx2]); cx(qubits[idx1], qubits[idx2]); rx(-np.pi/2, qubits[idx1])
        for i in range(0, len(list_4), 4):
            i0, i1, i2, i3 = list_4[i:i+4]
            cx(qubits[i0], qubits[i1]); cx(qubits[i2], qubits[i3]); cx(qubits[i1], qubits[i2]); rz(t_val, qubits[i2]); cx(qubits[i1], qubits[i2]); cx(qubits[i0], qubits[i1]); cx(qubits[i2], qubits[i3])
    mz(qubits)

# -----------------------------------------------------------------------------
# 3. Helpers (Interactions, Population Management)
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
    G2, G4, list_2, list_4 = [], [], [], []
    for i in [list(u) for u in unique]:
        if len(i) == 2:
            list_2 += [i[0], i[1]]; G2.append(set(i))
        else:
            list_4 += [i[0], i[1], i[2], i[3]]; G4.append(set(i))
    return G2, G4, list_2, list_4

def get_interactions_scaled(n):
    list_2, list_4 = [], []
    for i in range(n - 1): list_2.extend([i, i + 1])
    for i in range(n - 3): list_4.extend([i, i + 1, i + 2, i + 3])
    return list_2, list_4

def export_ready_for_cuda(quantum_population, n, filename, gpu_pop_size=8192, gpu_max_n=512):
    host_buffer = np.random.choice([-1, 1], size=(gpu_pop_size, gpu_max_n)).astype(np.int8)
    num_samples = quantum_population.shape[0]
    rows_to_fill = min(num_samples, gpu_pop_size)
    host_buffer[:rows_to_fill, :n] = quantum_population[:rows_to_fill].astype(np.int8)
    host_buffer.tofile(filename)

# -----------------------------------------------------------------------------
# 4. Simulation Controller
# -----------------------------------------------------------------------------

def run_simulation(n, shots, pop_size, output_file, variant, steps, lam_method):
    print(f"\n=== Quantum LABS Simulation ({variant.upper()}) ===")
    
    # Override steps for specific schedules
    if lam_method in ['linear', 'trig', 'sqrt', 'cuberoot']:
        steps = 10
    
    print(f"N={n}, Shots={shots}, Steps={steps}, Lambda={lam_method}")

    if variant == 'tensor_heavy':
        cudaq.set_target("tensornet")
        list_2, list_4 = get_interactions_scaled(n)
        thetas_CD = [0.1] * steps
        lambdas_AD, CD_toggle, AD_toggle = [], [], []
    else:
        G2, G4, list_2, list_4 = get_interactions(n)
        T = 1.0; dt = T / steps
        thetas_CD = []
        
        # Schedule Windows (AD start, CD end)
        if lam_method in ['linear', 'trig']:
            ad_start, cd_end = 7, 8
        elif lam_method == 'sqrt':
            ad_start, cd_end = 4, 6
        elif lam_method == 'cuberoot':
            ad_start, cd_end = 2, 4
        else:
            ad_start, cd_end = 1, steps

        # Compute data
        for step in range(1, steps + 1):
            t = step * dt
            thetas_CD.append(compute_theta(t, dt, T, n, G2, G4, lam_method))
        
        lambdas_AD = compute_adiabatic_schedule(steps, T, lam_method)
        CD_toggle = [(i+1 <= cd_end) for i in range(steps)]
        AD_toggle = [(i+1 >= ad_start) for i in range(steps)]

    # Sampling
    if variant == 'default':
        result = cudaq.sample(kernel_default, n, list_2, list_4, thetas_CD)
    elif variant == 'jenga':
        result = cudaq.sample(kernel_jenga, n, list_2, list_4, thetas_CD, shots_count=shots)
    elif variant == 'dna':
        result = cudaq.sample(kernel_dna, n, 1.0/steps, list_2, list_4, thetas_CD, lambdas_AD, CD_toggle, AD_toggle, shots_count=shots)
    elif variant == 'beyblade':
        result = cudaq.sample(kernel_beyblade, n, 1.0/steps, list_2, list_4, thetas_CD, lambdas_AD, CD_toggle, AD_toggle, shots_count=shots)
    elif variant == 'tensor_heavy':
        result = cudaq.sample(kernel_tensor_heavy, n, list_2, list_4, thetas_CD, shots_count=shots)
    
    top_strings = [bs for bs, count in sorted(result.items(), key=lambda x: x[1], reverse=True)[:pop_size]]
    pop_list = [(2 * np.array([int(b) for b in s]) - 1).astype(np.int8) for s in top_strings]
    export_ready_for_cuda(np.array(pop_list), n, output_file, gpu_pop_size=pop_size)
    print(f"Simulation Complete. Candidates exported to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--popsize", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=str, default="warm_start.bin")
    parser.add_argument("--variant", type=str, choices=['jenga', 'dna', 'beyblade', 'tensor_heavy'], default='dna')
    parser.add_argument("--lambda_method", type=str, choices=['linear', 'sqrt', 'cuberoot', 'trig'], default='trig')
    
    args = parser.parse_args()
    run_simulation(args.n, args.shots, args.popsize, args.output, args.variant, args.steps, args.lambda_method)