#include "kernels.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define POP_SIZE 100 // Fixed population size from paper
#define MAX_SHARED_INTS                                                        \
  5000 // Adjust based on N (N=200 -> ~5KB needs carefully packing)

// Helper: Get bit at pos
__device__ inline int get_bit(const uint32_t *bits, int pos) {
  return (bits[pos / 32] >> (pos % 32)) & 1;
}

// Helper: Flip bit at pos
__device__ inline void flip_bit(uint32_t *bits, int pos) {
  atomicXor(&bits[pos / 32], (1u << (pos % 32)));
}

// Helper: Set bit at pos to val (0 or 1) - Non-atomic for initialization/copy
__device__ inline void set_bit(uint32_t *bits, int pos, int val) {
  uint32_t mask = (1u << (pos % 32));
  if (val)
    bits[pos / 32] |= mask;
  else
    bits[pos / 32] &= ~mask;
}

// Helper: Compute full energy of a sequence
// Returns E = sum(C_k^2)
// Parallelized across block threads for a single sequence
__device__ int compute_energy_parallel(int N, const uint32_t *seq,
                                       int *shared_vectorC, int tid,
                                       int block_dim) {
  // 1. Compute vectorC[k] for k in 1..N-1
  // C_k = sum_{i=1}^{N-k} s_i * s_{i+k}
  // s_i is {-1, 1}. We store bits {0, 1}. Mapping 0->-1, 1->1 (or vice versa).
  // s_i * s_j = (2*b_i - 1) * (2*b_j - 1)
  //           = 4*b_i*b_j - 2*b_i - 2*b_j + 1
  // Actually simpler: XNOR (equal -> 1, diff -> -1).
  // s_i * s_j = 1 if b_i == b_j else -1.
  //           = 1 - 2 * (b_i ^ b_j)

  // Each thread computes subset of C_k's or we split the dot product?
  // Parallelizing C_k calculation:
  // With 128 threads and N ~ 100, each thread can compute one C_k?
  // N-1 coeffs.

  for (int k = 1 + tid; k < N; k += block_dim) {
    int ck = 0;
    for (int i = 0; i < N - k; ++i) {
      int b1 = get_bit(seq, i);
      int b2 = get_bit(seq, i + k);
      ck += (b1 == b2) ? 1 : -1;
    }
    shared_vectorC[k] = ck;
  }
  __syncthreads();

  // 2. Reduce sum of squares
  int local_sum = 0;
  for (int k = 1 + tid; k < N; k += block_dim) {
    local_sum += shared_vectorC[k] * shared_vectorC[k];
  }

  // We need a block reduction for local_sum
  // Use warp shuffle or shared mem
  // Simple shared mem reduction
  // Assume we have space for reduction buffer or use atomicAdd (slow but OK for
  // init) Or just one atomicAdd to a result variable

  // For simplicity using atomicAdd to a shared var
  // Need a shared variable passed in or allocated
  // We'll return partial sum and caller aggregates? No, "parallel" implies
  // collective result.

  // Hack: use shared_vectorC[0] as accumulator since C_0 is N (const) or unused
  // in metric? Metric is sum k=1..N-1 C_k^2. C_0 is usually N. Let's use
  // atomicAdd on a single shared var.
  return local_sum; // Caller must reduce
}

// Compute delta energy for flipping bit p
// Delta E = sum_k (C_k' ^ 2 - C_k ^ 2)
// C_k' = C_k - 2 * s_p * (s_{p-k} + s_{p+k})
// s_p is old value.
// Terms in parens:
// s_{p-k}: valid if p-k >= 0 => k <= p
// s_{p+k}: valid if p+k < N  => k < N - p
__device__ int compute_delta_energy(int N, int p, const uint32_t *seq,
                                    const int *vectorC) {
  int sp = get_bit(seq, p) ? 1 : -1;
  int delta = 0;

  // Iterate k from 1 to N-1
  // This function is called PER THREAD for checking ONE neighbor (flip p)
  // So this must be O(N) sequential.

  for (int k = 1; k < N; ++k) {
    int term = 0;
    if (p - k >= 0) {
      term += (get_bit(seq, p - k) ? 1 : -1);
    }
    if (p + k < N) {
      term += (get_bit(seq, p + k) ? 1 : -1);
    }

    if (term != 0) {
      // change in C_k is dCk = -2 * sp * term
      // new C_k' = C_k + dCk
      // d(C_k^2) = (C_k + dCk)^2 - C_k^2
      //          = 2 * C_k * dCk + dCk^2

      int dCk = -2 * sp * term;
      delta += 2 * vectorC[k] * dCk + dCk * dCk;
    }
  }
  return delta;
}

// Update vectorC after flipping bit p
__device__ void update_vectorC(int N, int p, int sp_old_val,
                               const uint32_t *seq, int *vectorC, int tid,
                               int block_dim) {
  // sp_old is pass in value (+1 or -1)
  // This is run in parallel by all threads.

  for (int k = 1 + tid; k < N; k += block_dim) {
    int term = 0;
    if (p - k >= 0) {
      term += (get_bit(seq, p - k) ? 1 : -1);
    }
    if (p + k < N) {
      term += (get_bit(seq, p + k) ? 1 : -1);
    }

    if (term != 0) {
      int dCk = -2 * sp_old_val * term;
      vectorC[k] += dCk;
    }
  }
}

__global__ void memetic_search_kernel(int N, int target_energy, int *stop_flag,
                                      int *global_best_energy, uint64_t *seeds,
                                      uint32_t *global_best_seq,
                                      long long *log_time, int *log_energy,
                                      int *log_count, int *d_lock) {
  // Dynamic shared memory
  extern __shared__ int shared_mem[];
  // Memory Layout:
  // [Sequence (32-bit aligned)] - Size depending on N and Population
  // For Population K=100: K * (N/32 + 1) ints
  // For Working Child + Temp: 2 * (N/32 + 1) ints
  // VectorC: N ints
  // TabuList: N ints
  // Population Energies: K ints

  int ints_per_seq = (N + 31) / 32;
  uint32_t *pop_seqs = (uint32_t *)shared_mem;
  int pop_storage_size = POP_SIZE * ints_per_seq;

  int *pop_energies = (int *)&pop_seqs[pop_storage_size];

  uint32_t *child_seq = (uint32_t *)&pop_energies[POP_SIZE];
  uint32_t *best_tabu_seq = (uint32_t *)&child_seq[ints_per_seq];

  int *vectorC = (int *)&best_tabu_seq[ints_per_seq];
  int *tabu_list = (int *)&vectorC[N];

  // Helper pointers
  int *shared_reduction = (int *)&tabu_list[N]; // Reuse space or specific slot

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int bdim = blockDim.x;

  curandState state;
  curand_init(seeds[bid], tid, 0, &state);

  // 1. Initialize Population
  // Parallel Initialization: Each thread inits some bits of some sequences or
  // full sequences? Let's do: Loop over population, init each. Parallelize bits
  // init for speed? N is small (100-200), K=100. K*N ~ 10k ops. Fast. Init
  // population
  for (int i = 0; i < POP_SIZE; ++i) {
    // Init bits
    for (int j = tid; j < ints_per_seq; j += bdim) {
      pop_seqs[i * ints_per_seq + j] = curand(&state);
      // Mask excess bits if needed? Not critical if logic respects N.
    }
  }
  __syncthreads();

  // Compute Energies
  // Loop over Pop
  for (int i = 0; i < POP_SIZE; ++i) {
    // Compute energy for pop[i] using shared vectorC as buffer?
    // Need to serialize this or use different buffers?
    // We only have one VectorC buffer.
    // So compute sequentially for population (init is one-off).

    // Compute C_k for pop[i]
    int e_part = compute_energy_parallel(N, &pop_seqs[i * ints_per_seq],
                                         vectorC, tid, bdim);

    // Reduce e_part
    // Use atomicAdd to a shared var?
    if (tid == 0)
      shared_reduction[0] = 0;
    __syncthreads();
    atomicAdd(&shared_reduction[0], e_part);
    __syncthreads();

    if (tid == 0) {
      pop_energies[i] = shared_reduction[0];

      // Update global best with lock
      int current_pop_e = shared_reduction[0];
      if (current_pop_e < *global_best_energy) {
        while (atomicCAS(d_lock, 0, 1) != 0)
          ; // Lock
        if (current_pop_e < *global_best_energy) {
          atomicMin(global_best_energy, current_pop_e);
          // Copy Sequence
          for (int j = 0; j < ints_per_seq; ++j) {
            global_best_seq[j] = pop_seqs[i * ints_per_seq + j];
          }
          // Log
          int log_idx = *log_count;
          if (log_idx < 1000) {
            log_time[log_idx] = clock64();
            log_energy[log_idx] = current_pop_e;
            *log_count = log_idx + 1;
          }
        }
        atomicExch(d_lock, 0); // Unlock
      }
    }
    __syncthreads();
  }

  // Memetic Loop
  int generations = 0;
  while (!(*stop_flag) && generations < 20000) { // Safety break
    generations++;

    // 2. Selection (Thread 0)
    // Pick 2 parents random
    // Copy to child
    // 2. Selection (Thread 0) & Crossover
    __syncthreads();
    if (tid == 0) {
      // Select 2 parents randomly
      int p1 = curand(&state) % POP_SIZE;
      int p2 = curand(&state) % POP_SIZE;

      // Crossover: Uniform crossover (p_comb = 0.9)
      // For each bit, take from p1 or p2
      // Mutation: p_mutate = 1/N

      // We copy to child_seq
      for (int j = 0; j < N; ++j) {
        // Handle bit level
        int b1 = get_bit(&pop_seqs[p1 * ints_per_seq], j);
        int b2 = get_bit(&pop_seqs[p2 * ints_per_seq], j);
        int child_bit = (curand_uniform(&state) < 0.5f) ? b1 : b2;

        // Mutation
        if (curand_uniform(&state) < (1.0f / N)) {
          child_bit = 1 - child_bit;
        }

        // Set bit in child_seq
        // This is sequential by thread 0, might be slow but N is small
        // Ideally parallelize this
        set_bit(child_seq, j, child_bit);
      }
    }
    __syncthreads();
    // Since N is small and thread 0 did it, we are safe.
    // For larger N, parallelize:
    // e.g. "if (tid < N) { int p1=...; int b1 = ... }"
    // But need synchronized random state or pretrace choice.
    // Keeping it simple for Thread 0 as N~200 is tiny.

    // 3. Tabu Search on `child_seq`
    // Initialize Tabu Search State
    // Init vectorC for child (Parallel)
    // Re-compute energy from scratch for child
    int local_E = compute_energy_parallel(N, child_seq, vectorC, tid, bdim);

    // Reduce to get full energy
    if (tid == 0)
      shared_reduction[0] = 0;
    __syncthreads();
    atomicAdd(&shared_reduction[0], local_E);
    __syncthreads();

    int current_energy = shared_reduction[0]; // All threads get full energy
    int best_child_energy = current_energy;   // Track best in this LS

    // Copy child to best_tabu_seq (Thread 0)
    // Parallel copy
    for (int j = tid; j < ints_per_seq; ++j)
      best_tabu_seq[j] = child_seq[j];

    // Clear Tabu List
    for (int j = tid; j < N; j += bdim)
      tabu_list[j] = 0;

    __syncthreads();

    int iter = 0;
    int max_iter = N; // Paper: [N/2, 3N/2]. Fixed N for simplicity or use rand
    if (tid == 0) {
      // Random max iter
      max_iter = N / 2 + (curand(&state) % (N + 1));
    }
    // Broadcast max_iter? It's in register of thread 0?
    // Use shared mem to broadcast
    if (tid == 0)
      shared_reduction[0] = max_iter;
    __syncthreads();
    max_iter = shared_reduction[0];

    // Tabu Loop
    for (; iter < max_iter; ++iter) {
      // Evaluate all neighbors parallely
      // Each thread finds its local best
      int local_best_delta = 9999999;
      int local_best_p = -1;

      for (int p = tid; p < N; p += bdim) {
        int delta = compute_delta_energy(N, p, child_seq, vectorC);

        // Check Tabu
        bool is_tabu = (tabu_list[p] > iter);

        // Aspiration: if current_energy + delta < best_child_energy
        int new_energy = current_energy + delta;
        if (is_tabu && new_energy >= best_child_energy) {
          continue; // Skip if tabu and not aspiring
        }

        // Update local best
        // We want Lowest Energy => Lowest Delta (most negative)
        if (local_best_p == -1 || delta < local_best_delta) {
          local_best_delta = delta;
          local_best_p = p;
        }
      }

      // Reduction to find global best for this iteration
      // We need minimum delta across threads
      // Warp shuffle reduction
      int best_delta_val = local_best_delta;
      int best_p_val = local_best_p;

      for (int offset = 16; offset > 0; offset /= 2) {
        int other_delta = __shfl_down_sync(0xFFFFFFFF, best_delta_val, offset);
        int other_p = __shfl_down_sync(0xFFFFFFFF, best_p_val, offset);
        if (other_p != -1 &&
            (other_delta < best_delta_val || best_p_val == -1)) {
          best_delta_val = other_delta;
          best_p_val = other_p;
        }
      }

      // Shared memory for block reduction (across warps)
      __shared__ int warp_deltas[32]; // Max 32 warps for 1024 threads
      __shared__ int warp_ps[32];

      int warp_id = tid / 32;
      int lane_id = tid % 32;
      if (lane_id == 0) {
        warp_deltas[warp_id] = best_delta_val;
        warp_ps[warp_id] = best_p_val;
      }
      __syncthreads();

      if (warp_id == 0) {
        // First warp reduces the warp_deltas
        int num_warps = (bdim + 31) / 32;
        if (tid < num_warps) {
          best_delta_val = warp_deltas[tid];
          best_p_val = warp_ps[tid];
        } else {
          best_p_val = -1;
          best_delta_val = 9999999;
        }

        for (int offset = 16; offset > 0; offset /= 2) {
          int other_delta =
              __shfl_down_sync(0xFFFFFFFF, best_delta_val, offset);
          int other_p = __shfl_down_sync(0xFFFFFFFF, best_p_val, offset);
          if (other_p != -1 &&
              (other_delta < best_delta_val || best_p_val == -1)) {
            best_delta_val = other_delta;
            best_p_val = other_p;
          }
        }

        if (tid == 0) {
          // best_p_val is the winner
          shared_reduction[0] = best_p_val;
          shared_reduction[1] = best_delta_val;
        }
      }
      __syncthreads();

      int move_p = shared_reduction[0];
      int move_delta = shared_reduction[1];

      if (move_p != -1) {
        // Update Structures (Thread 0 or Parallel?)
        // update_vectorC is parallel
        int old_bit_val = get_bit(child_seq, move_p) ? 1 : -1;
        update_vectorC(N, move_p, old_bit_val, child_seq, vectorC, tid, bdim);
      }
      // Barrier ensures vectorC update is done before we flip bit
      __syncthreads();

      if (move_p != -1) {
        // Update state on ALL threads
        current_energy += move_delta;

        bool improved = false;
        if (current_energy < best_child_energy) {
          best_child_energy = current_energy;
          improved = true;
        }

        if (tid == 0) {
          flip_bit(child_seq, move_p);

          // Update Tabu Tenure
          // Paper: [0.1 L, 0.12 L] + current_iter
          int tenure = (int)(0.1 * max_iter); // Simplified
          tabu_list[move_p] = iter + tenure;  // Use logic from paper

          // Check Best
          if (improved) {
            // Update best_tabu_seq
            for (int j = 0; j < ints_per_seq; ++j)
              best_tabu_seq[j] = child_seq[j];
          }
        }
      }
      __syncthreads();
    } // End Tabu Loop

    // 4. Update Global Best & Population
    if (tid == 0) {
      // Optimistic check first
      if (best_child_energy < *global_best_energy) {
        // Acquire lock (simple spinlock)
        while (atomicCAS(d_lock, 0, 1) != 0)
          ;

        // Double check inside lock
        if (best_child_energy < *global_best_energy) {
          atomicMin(global_best_energy, best_child_energy);

          // Copy Sequence
          // Note: best_tabu_seq is in shared memory.
          for (int j = 0; j < ints_per_seq; ++j) {
            global_best_seq[j] = best_tabu_seq[j];
          }

          // Log
          int log_idx = *log_count;
          if (log_idx < 1000) { // Limit log size to prevent overflow
            log_time[log_idx] = clock64();
            log_energy[log_idx] = best_child_energy;
            *log_count = log_idx + 1;
          }

          // Check Target
          if (best_child_energy <= target_energy) {
            *stop_flag = 1;
          }
        }

        // Release lock
        atomicExch(d_lock, 0);
      }

      // Replacement: Replace random individual
      int vic_idx = curand(&state) % POP_SIZE;
      for (int j = 0; j < ints_per_seq; ++j)
        pop_seqs[vic_idx * ints_per_seq + j] = best_tabu_seq[j];
      pop_energies[vic_idx] = best_child_energy;
    }
    __syncthreads();

  } // End Memetic Loop
}
