import cudaq
import numpy as np
from math import floor
import tutorial_notebook.auxiliary_files.labs_utils as utils
import time

# TODO FIX LATER IMPORT
from ... import get_interactions
from ... import energy
from classical.mts import MTS

def get_interactions(N):
    """
    Generates the interaction sets G2 and G4 based on the loop limits in Eq. 15.
    Returns standard 0-based indices as lists of lists of ints.
   
    Args:
        N (int): Sequence length.
       
    Returns:
        G2: List of lists containing two body term indices
        G4: List of lists containing four body term indices
        
    Note: Reused from Phase 1.
    """
   
    G2 = []
    G4 = []
   
    for i in range(N - 2):
        max_k = (N - i) // 2          # depends on i
        for k in range(1, max_k + 1): # starts at 1 to avoid (i,i)
            G2.append([i, i + k])

    # --- Four-body terms ---
    for i in range(N - 3):
        max_t = (N - i - 1) // 2
        for t in range(1, max_t + 1):
            for k in range(t + 1, N - i - t):
                quad = [i, i + t, i + k, i + k + t]
                if quad[3] < N:
                    G4.append(quad)

    return G2, G4

def bias_angle(hb: float) -> float:
    """
    Ground-state rotation angle for H = σ_x + hb σ_z.
    Returns angle for Ry rotation to prepare ground state.
    From paper: θ_i = 2 tan^{-1}((h_i^b + λ_min) / h_i^b)
    Simplified for h_x = -1 case.
    """
    if hb == 0:
        return np.pi/2  # Ry(pi/2)|0⟩ = |+⟩
    # For H = -σ_x + hb σ_z, ground state rotation
    return 2.0 * np.arctan(hb + np.sqrt(hb**2 + 1.0))

@cudaq.kernel
def biased_trotter_circuit(N: int, angles: list[float], G2: list[list[int]], 
                          G4: list[list[int]], steps: int, dt: float, 
                          theta_cutoff: float,
                          T: float, thetas: list[float]):
    """
    Trotterized circuit with bias-field initialization.
    Combines your existing circuit with bias preparation.
    """
    # theta_cutoff = 0.05 # TODO: change?
    q = cudaq.qvector(N)
    
    # Initialize with bias-field rotations (Eq. 9 in paper)
    for i in range(N):
        ry(angles[i], q[i])
    
    # Apply trotterized counterdiabatic evolution
    for s in range(steps):
        theta = thetas[s]

        # discard an angle if it's too small
        if theta < theta_cutoff:
            continue
        
        # 2-body block
        for pair in G2:
            i = pair[0]
            j = pair[1]
            # hx=-1
            r_yz(q[i], q[j], -4.0 * theta)
            r_zy(q[i], q[j], -4.0 * theta)
            
        # 4-body block
        for quad in G4:
            a, b, c, d = quad[0], quad[1], quad[2], quad[3]
            # hx=-1
            r_yzzz(q[a], q[b], q[c], q[d], -8.0 * theta)
            r_zyzz(q[a], q[b], q[c], q[d], -8.0 * theta)
            r_zzyz(q[a], q[b], q[c], q[d], -8.0 * theta)
            r_zzzy(q[a], q[b], q[c], q[d], -8.0 * theta)

def run_biased_circuit(N: int, angles: np.ndarray, T: float, 
                       shots: int, n_steps: int,
                       theta_cutoff: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the biased trotterized circuit and return samples.
    
    Args:
        N: Problem size
        angles: Ry rotation angles for each qubit (from bias fields)
        T: Total evolution time
        n_steps: Number of Trotter steps
        shots: Number of measurements
        theta_cutoff: Discard gates with angles below this threshold
    
    Returns:
        bitstrings: Array of shape (shots, N) in {0, 1}
        energies: Array of shape (shots,) with LABS energies
    """
    # Get interactions
    G2, G4 = get_interactions(N)
    dt = T / n_steps
    
    # Compute theta values for each Trotter step
    thetas = []
    for step in range(1, n_steps + 1):
        t = step * dt
        # Using utils.compute_theta as in your notebook
        theta_val = utils.compute_theta(t, dt, T, N, G2, G4)
        thetas.append(theta_val)
    
    # Sample from circuit
    counts = cudaq.sample(
        biased_trotter_circuit, 
        N, angles.tolist(), G2, G4, n_steps, dt, T, thetas, 
        theta_cutoff=theta_cutoff,
        shots_count=shots
    )
    
    # Convert to numpy arrays
    bitstrings = []
    energies = []
    
    for bitstr, count in counts.items():
        # Convert bitstring to array
        bits = np.array([int(b) for b in bitstr], dtype=int)
        # Convert to spins {+1, -1}
        spins = 1 - 2 * bits
        # Compute energy
        E = energy(spins)
        
        for _ in range(count):
            bitstrings.append(bits)
            energies.append(E)
            
            if len(bitstrings) >= shots:
                break
        if len(bitstrings) >= shots:
            break
    
    # Pad if needed
    while len(bitstrings) < shots:
        bitstrings.append(np.zeros(N, dtype=int))
        energies.append(float('inf'))
    
    return np.array(bitstrings[:shots]), np.array(energies[:shots])

def bf_dcqo_sampler(N: int, n_iter: int, n_shots: int, 
                   alpha: float, kappa: float,
                   T: float, n_steps: int,
                   theta_cutoff: float) -> Tuple[np.ndarray, List[float]]:
    """
    Bias-Field Digitized Counterdiabatic Quantum Optimization sampler.
    Based on Algorithm 1 in the paper.
    
    Args:
        N: Problem size
        n_iter: Number of BF-DCQO iterations
        n_shots: Number of measurements per iteration
        alpha: CVaR fraction (e.g., 0.01 = best 1%)
        kappa: Strength of final signed bias
        T: Total evolution time
        n_steps: Number of Trotter steps
        theta_cutoff: Gate cutoff threshold
        
    Returns:
        samples: Best samples from final iteration (spins in {+1, -1})
        energy_history: Best energy at each iteration
    """
    # Initialize bias fields
    h_b = np.zeros(N)
    energy_history = []
    all_samples = []
    
    for iteration in range(n_iter):
        print(f"[BF-DCQO] Iteration {iteration+1}/{n_iter}")
        
        # 1. Prepare biased initial state
        angles = np.array([bias_angle(hb) for hb in h_b])
        
        # 2. Run quantum circuit
        bitstrings, energies = run_biased_circuit(
            N, angles, T=T, n_steps=n_steps, shots=n_shots, 
            theta_cutoff=theta_cutoff
        )

        # 3. Convert to -1, 1
        samples = 1 - 2 * bitstrings
        
        # 4. CVaR filtering (keep best alpha fraction)
        k = max(1, int(alpha * n_shots))
        elite_idx = np.argsort(energies)[:k]
        elite_samples = samples[elite_idx]
        elite_energies = energies[elite_idx]
        
        # 5. Compute ⟨σ_z⟩ per qubit from elite samples
        mean_z = elite_samples.mean(axis=0)  # shape (N,)

        # DEBUG
        # print("MEAN", mean_z)
        
        # 6. Update bias fields (different strategies)
        if iteration < n_iter - 1:
            # Unsigned bias for exploration (learning phase)
            h_b = mean_z
            strategy = "unsigned"
        else:
            # Final iteration: strong signed bias for convergence
            h_b = kappa * np.sign(mean_z)
            strategy = "signed (κ={})".format(kappa)
        
        # Track best energy
        best_energy = np.min(energies)
        energy_history.append(best_energy)
        all_samples.append(samples)
        
        print(f"  Strategy: {strategy}")
        print(f"  Best energy: {best_energy}")
        print(f"  Mean |h_b|: {np.mean(np.abs(h_b)):.3f}")
        print(f"  CVaR fraction: {alpha} ({k} samples)")
    
    # Return samples from final iteration
    return all_samples[-1], energy_history

def quantum_enhanced_mts(N: int, pop_size: int, 
                        bf_dcqo_iter: int, mts_iter: int,
                        quantum_shots: int, alpha: float,
                        kappa: float, T: float, theta_cutoff: float,
                        use_cvar: bool = True) -> dict:
    """
    Complete quantum-enhanced MTS workflow.
    
    1. Generate initial population with BF-DCQO
    2. Refine with MTS
    3. Return comprehensive results
    
    Returns dictionary with timing and performance metrics.
    """
    results = {
        'N': N,
        'pop_size': pop_size,
        'timing': {},
        'energies': {},
        'solution': None
    }

    # print("sanity check of input")
    # print(N, pop_size, bf_dcqo_iter, mts_iter, quantum_shots, alpha, kappa, T)
    
    # Phase 1: BF-DCQO for initial population
    print("\n" + "="*60)
    print("PHASE 1: BF-DCQO Quantum Sampling")
    print("="*60)
    
    start_time = time.perf_counter()
    quantum_samples, bf_dcqo_energies = bf_dcqo_sampler(
        N, n_iter=bf_dcqo_iter, n_shots=quantum_shots,
        alpha=alpha, kappa=kappa, T=T, 
        n_steps=100, theta_cutoff=theta_cutoff
    )
    end_time = time.perf_counter()
    bf_dcqo_time = end_time - start_time
    
    # Select diverse samples for population
    population = []
    if len(quantum_samples) > pop_size:
        # Select based on energy and diversity
        energies = np.array([energy(s) for s in quantum_samples])
        elite_idx = np.argsort(energies)[:pop_size]
        population = quantum_samples[elite_idx]
    else:
        population = quantum_samples

    # print("DEBUG POPULATION", population)
    
    results['timing']['bf_dcqo'] = bf_dcqo_time
    results['energies']['bf_dcqo'] = bf_dcqo_energies
    results['bf_dcqo_best'] = min(bf_dcqo_energies)
    
    # Phase 2: MTS refinement
    print("\n" + "="*60)
    print("PHASE 2: Memetic Tabu Search Refinement")
    print("="*60)
    
    start_time = time.perf_counter()
    best_s, best_E, final_pop, final_energies, mts_history = MTS(
        k=len(population), N=N, max_iter=mts_iter, population0=population
    )
    end_time = time.perf_counter()
    mts_time = end_time - start_time
    # print("DEBUG ENERGIES", best_E)
    # print("DEBUG S", best_s)
    
    results['timing']['mts'] = mts_time
    results['timing']['total'] = bf_dcqo_time + mts_time
    results['energies']['mts'] = mts_history
    results['solution'] = {
        'bitstring': best_s,
        'energy': best_E
    }
    results['population'] = final_pop
    results['population_energies'] = final_energies
    
    # Calculate speedup metrics
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Problem size: N = {N}")
    print(f"Population size: {pop_size}")
    print(f"BF-DCQO iterations: {bf_dcqo_iter}")
    print(f"MTS iterations: {mts_iter}")
    print(f"\nTiming:")
    print(f"  BF-DCQO: {bf_dcqo_time:.2f} seconds")
    print(f"  MTS: {mts_time:.2f} seconds")
    print(f"  Total: {results['timing']['total']:.2f} seconds")
    print(f"\nEnergies:")
    print(f"  BF-DCQO best: {min(bf_dcqo_energies)}")
    print(f"  MTS final best: {best_E}")
    print(f"  Improvement: {(min(bf_dcqo_energies) - best_E)/abs(min(bf_dcqo_energies))*100:.1f}%")
    
    return results