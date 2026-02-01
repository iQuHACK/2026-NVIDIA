from numba import cuda, int32, int8
import numpy as np

# Constants for the kernel
TPB = 128  # Threads Per Block (Adjust based on GPU, usually 128 or 256)


@cuda.jit
def labs_memetic_kernel(
    rng_states,  # Random number generator states
    population_out,  # Output: Final best sequences (Blocks x N)
    energies_out,  # Output: Final energies (Blocks)
    N,  # Sequence length
    max_steps,  # Tabu iterations
    tabu_tenure,  # How long a move stays banned
):
    # -----------------------------------------------------------
    # 1. SHARED MEMORY ALLOCATION
    #    Each Block is one "Walker". All threads in the block share this data.
    # -----------------------------------------------------------
    # Current sequence s[i]
    s_sh = cuda.shared.array(shape=(256,), dtype=int8)
    # Best sequence found so far for this walker
    best_s_sh = cuda.shared.array(shape=(256,), dtype=int8)
    # Correlation array C[k]
    C_sh = cuda.shared.array(shape=(256,), dtype=int32)
    # Tabu list: stores the step number when a bit is free to flip again
    tabu_list_sh = cuda.shared.array(shape=(256,), dtype=int32)

    # Scalar shared state (visible to all threads in the block)
    current_E_sh = cuda.shared.array(shape=(1,), dtype=int32)
    best_E_sh = cuda.shared.array(shape=(1,), dtype=int32)

    # Reduction arrays for finding the best move
    # We store the delta_E and the bit_index for every thread
    scratch_val = cuda.shared.array(shape=(TPB,), dtype=int32)
    scratch_idx = cuda.shared.array(shape=(TPB,), dtype=int32)

    # Thread and Block indices
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x

    # -----------------------------------------------------------
    # 2. INITIALIZATION (Parallelized)
    # -----------------------------------------------------------
    # Initialize Random Sequence for this block
    # Each thread initializes a chunk of the sequence
    for i in range(tx, N, bw):
        # Simple RNG: Use Numba's XORWOW or similar (omitted for brevity, assume random start)
        # For simplicity here, we map blockID + threadID to a pseudo-random start
        val = 1 if ((bx * N + i) % 2 == 0) else -1
        s_sh[i] = val
        best_s_sh[i] = val
        tabu_list_sh[i] = 0  # Not tabu initially

    cuda.syncthreads()  # Wait for s_sh to be ready

    # Calculate initial Correlation C[] and Energy E
    # This is O(N^2), but we only do it once.
    # Parallelize the calculation of C[k]
    for k in range(tx + 1, N, bw):  # k goes 1..N-1
        dot_prod = 0
        for i in range(N - k):
            dot_prod += s_sh[i] * s_sh[i + k]
        C_sh[k] = dot_prod

    cuda.syncthreads()

    # Calculate initial Energy E (Thread 0 does the sum for simplicity)
    # (In production, use a parallel reduction here too)
    if tx == 0:
        current_E = 0
        for k in range(1, N):
            current_E += C_sh[k] * C_sh[k]

        current_E_sh[0] = current_E
        best_E_sh[0] = current_E

    cuda.syncthreads()

    # -----------------------------------------------------------
    # 3. MAIN TABU SEARCH LOOP
    # -----------------------------------------------------------
    for step in range(1, max_steps + 1):
        current_E = current_E_sh[0]
        best_E_walker = best_E_sh[0]

        # --- A. EVALUATE NEIGHBORS (The Bottleneck) ---
        # Each thread checks a specific bit flip 'j'
        # Or if N > Threads, use a stride loop.

        my_best_delta = 2147483647  # Max Int
        my_best_bit = -1

        for j in range(tx, N, bw):
            # Is this move Tabu?
            is_tabu = tabu_list_sh[j] > step

            # Compute Delta E efficiently
            # dE = -4 * s[j] * sum(C[k] * s[j+k] + C[k]*s[j-k]) ... roughly
            # Implementing the exact Delta function from your python code:
            sj = s_sh[j]
            delta = 0

            # This inner loop is O(N)
            for k in range(1, N):
                dCk = 0
                if j < N - k:
                    dCk += -2 * sj * s_sh[j + k]
                if j >= k:
                    dCk += -2 * s_sh[j - k] * sj

                if dCk != 0:
                    delta += 2 * C_sh[k] * dCk + dCk * dCk

            # Aspiration Criteria: If (current_E + delta) < best_E_walker, ignore tabu
            if is_tabu and (current_E + delta >= best_E_walker):
                continue  # Skip tabu move

            # Track best seen by this thread
            if delta < my_best_delta:
                my_best_delta = delta
                my_best_bit = j

        # Store result in shared memory for reduction
        scratch_val[tx] = my_best_delta
        scratch_idx[tx] = my_best_bit

        cuda.syncthreads()

        # --- B. REDUCTION (Find Best Move in Block) ---
        # Simple iterative reduction by Thread 0 (Fast enough for N < 256)
        # For larger N, use tree reduction.
        if tx == 0:
            winner_delta = 2147483647
            winner_bit = -1

            for i in range(min(N, bw)):  # Only check active threads
                val = scratch_val[i]
                if val < winner_delta:
                    winner_delta = val
                    winner_bit = scratch_idx[i]

            # Store winner in scratch 0 to broadcast
            scratch_val[0] = winner_delta
            scratch_idx[0] = winner_bit

        cuda.syncthreads()

        best_delta = scratch_val[0]
        best_bit = scratch_idx[0]

        if best_bit == -1:
            cuda.syncthreads()
            continue

        # --- C. UPDATE STATE (Incremental) ---
        # Parallel Update of C (Optimization from Paper)
        # Instead of one thread updating all C, split it up!
        sj_old = s_sh[best_bit]

        # Update C array in parallel
        for k in range(tx + 1, N, bw):
            if best_bit < N - k:
                C_sh[k] += -2 * sj_old * s_sh[best_bit + k]
            if best_bit >= k:
                C_sh[k] += -2 * s_sh[best_bit - k] * sj_old

        cuda.syncthreads()

        # Thread 0 updates scalar values and Tabu list
        if tx == 0:
            s_sh[best_bit] = -s_sh[best_bit]  # Flip bit
            current_E = current_E_sh[0] + best_delta
            current_E_sh[0] = current_E

            # Update Tabu List
            tabu_list_sh[best_bit] = step + tabu_tenure

            # Update Best Global (+ snapshot best sequence)
            if current_E < best_E_sh[0]:
                best_E_sh[0] = current_E
                for i in range(N):
                    best_s_sh[i] = s_sh[i]

        cuda.syncthreads()

    # -----------------------------------------------------------
    # 4. EXPORT RESULTS
    # -----------------------------------------------------------
    if tx == 0:
        energies_out[bx] = best_E_sh[0]
        # Copy best_s_sh to global memory
        for i in range(N):
            population_out[bx, i] = best_s_sh[i]


def run_gpu_labs(N, num_walkers=1024, max_steps=500, tabu_tenure=15):
    """
    N: Sequence length
    num_walkers: Size of population (Number of GPU Blocks)
    """
    if N > 256:
        raise ValueError("N must be <= 256 (shared-memory arrays are size 256)")
    # 1. Prepare Memory
    # Output arrays
    d_population = cuda.device_array((num_walkers, N), dtype=np.int8)
    d_energies = cuda.device_array(num_walkers, dtype=np.int32)

    # RNG states (needed for numba rng, omitted here for brevity)
    rng_states = 0

    # 2. Kernel Configuration
    threads_per_block = 128
    if threads_per_block != TPB:
        raise ValueError("threads_per_block must equal TPB for current kernel")
    blocks = num_walkers

    # 3. Launch
    print(f"Launching GPU Search: {blocks} Walkers, {N} bits...")
    labs_memetic_kernel[blocks, threads_per_block](
        rng_states, d_population, d_energies, N, max_steps, tabu_tenure=tabu_tenure
    )
    cuda.synchronize()

    # 4. Retrieve Results
    final_energies = d_energies.copy_to_host()
    final_pop = d_population.copy_to_host()

    best_idx = np.argmin(final_energies)
    print(f"Best Energy Found: {final_energies[best_idx]}")
    return final_pop[best_idx], final_energies[best_idx]
