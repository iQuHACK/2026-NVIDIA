#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>

// A100-80GB Optimized Constants
#define MAX_N 128            // Increased limit for future-proofing
#define THREADS_PER_BLOCK 128 
#define POP_SIZE 16384       // Increased population to saturate A100
#define MEMETIC_FREQ 500     
#define ELITE_COUNT 256      
#define TABU_MIN 10
#define TABU_VAR 20

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

const int TARGETS[] = {
    0, 0, 0, 1, 2, 2, 7, 3, 8, 12, 13, 5, 10, 6, 19, 15, 24, 32, 25, 29, 26, 
    26, 39, 47, 36, 36, 45, 37, 50, 62, 59, 67, 64, 64, 65, 73, 82, 86, 87, 99, 
    108, 112, 101, 109, 122, 118, 131, 135, 140, 136, 153, 153, 166, 170, 175, 
    171, 192, 188, 197, 205, 218, 226, 235, 207, 208, 240, 257
};

// -------------------------------------------------------------------------
// GPU Kernels
// -------------------------------------------------------------------------

__global__ void init_kernel(int N, int8_t* d_pop, curandState* states, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < POP_SIZE) {
        curand_init(seed, id, 0, &states[id]);
        curandState local_rng = states[id];
        for (int i = 0; i < N; i++) {
            d_pop[id * MAX_N + i] = (curand_uniform(&local_rng) > 0.5f) ? 1 : -1;
        }
        states[id] = local_rng;
    }
}

// Optimized Tabu Search with Warp Shuffles
__global__ void tabu_kernel(int N, int iterations, int8_t* d_pop, long long* d_energies, curandState* states) {
    // Dynamic shared memory layout
    extern __shared__ int8_t shared_mem[];
    int8_t* s_seq = &shared_mem[0];           // N bytes
    int8_t* s_best_seq = &shared_mem[MAX_N];  // N bytes
    int* s_corr = (int*)&shared_mem[MAX_N*2]; // N*4 bytes
    int* s_tabu = (int*)&shared_mem[MAX_N*6]; // N*4 bytes

    __shared__ long long s_cur_e;
    __shared__ long long s_best_e;

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    curandState rng = states[bid];

    // 1. Initialize local sequence and correlation table
    if (tid < N) {
        s_seq[tid] = d_pop[bid * MAX_N + tid];
        s_best_seq[tid] = s_seq[tid];
        s_tabu[tid] = 0;
    }
    __syncthreads();

    // Calculate initial correlations Ck
    if (tid > 0 && tid < N) {
        int ck = 0;
        for (int i = 0; i < N - tid; i++) ck += s_seq[i] * s_seq[i + tid];
        s_corr[tid] = ck;
    }
    __syncthreads();

    // Calculate initial energy
    if (tid == 0) {
        long long e = 0;
        for (int k = 1; k < N; k++) e += (long long)s_corr[k] * s_corr[k];
        s_cur_e = e; s_best_e = e;
    }
    __syncthreads();

    // 2. Tabu Iterations
    for (int iter = 0; iter < iterations; iter++) {
        long long best_local_dE = 2e18; // Large value
        int best_local_idx = -1;

        if (tid < N) {
            long long dE = 0;
            int s_p = s_seq[tid];
            // Delta calculation: O(N) simplified
            for (int k = 1; k < N; k++) {
                int nb = 0;
                if (tid + k < N) nb += s_seq[tid + k];
                if (tid - k >= 0) nb += s_seq[tid - k];
                int dCk = -2 * s_p * nb;
                dE += (long long)dCk * (2 * s_corr[k] + dCk);
            }

            // Tabu aspiration criteria
            if (iter >= s_tabu[tid] || (s_cur_e + dE < s_best_e)) {
                best_local_dE = dE;
                best_local_idx = tid;
            }
        }

        // Warp reduction to find best move
        for (int offset = 16; offset > 0; offset /= 2) {
            long long other_dE = __shfl_down_sync(0xFFFFFFFF, best_local_dE, offset);
            int other_idx = __shfl_down_sync(0xFFFFFFFF, best_local_idx, offset);
            if (other_dE < best_local_dE) {
                best_local_dE = other_dE;
                best_local_idx = other_idx;
            }
        }

        // Intra-block reduction using shared memory for the winners of each warp
        __shared__ long long block_dE[4]; 
        __shared__ int block_idx[4];
        if ((tid & 31) == 0) {
            block_dE[tid/32] = best_local_dE;
            block_idx[tid/32] = best_local_idx;
        }
        __syncthreads();

        if (tid == 0) {
            long long final_dE = block_dE[0];
            int final_idx = block_idx[0];
            for (int i = 1; i < 4; i++) {
                if (block_dE[i] < final_dE) { final_dE = block_dE[i]; final_idx = block_idx[i]; }
            }
            
            if (final_idx != -1) {
                block_idx[0] = final_idx; // communicate winner
                block_dE[0] = (long long)s_seq[final_idx]; // store old sign for update
                s_cur_e += final_dE;
                s_seq[final_idx] = -s_seq[final_idx];
                s_tabu[final_idx] = iter + TABU_MIN + (curand(&rng) % TABU_VAR);
                if (s_cur_e < s_best_e) {
                    s_best_e = s_cur_e;
                    block_idx[1] = 1; // Flag update
                } else block_idx[1] = 0;
            } else block_idx[0] = -1;
        }
        __syncthreads();

        int move_idx = block_idx[0];
        if (move_idx == -1) break;

        // Update correlations and best sequence in parallel
        if (block_idx[1] == 1 && tid < N) s_best_seq[tid] = s_seq[tid];
        
        int old_sign = (int)block_dE[0];
        if (tid > 0 && tid < N) {
            int nb = 0;
            if (move_idx + tid < N) nb += s_seq[move_idx + tid];
            if (move_idx - tid >= 0) nb += s_seq[move_idx - tid];
            s_corr[tid] += -2 * old_sign * nb;
        }
        __syncthreads();
    }

    // Write back
    if (tid < N) d_pop[bid * MAX_N + tid] = s_best_seq[tid];
    if (tid == 0) {
        d_energies[bid] = s_best_e;
        states[bid] = rng;
    }
}

// Elite Selection on GPU (Simple Odd-Even Sort for Elite subset)
// For POP_SIZE=16k, we only need the top 256. 
__global__ void select_elites_kernel(long long* d_energies, int* d_elite_indices) {
    int tid = threadIdx.x;
    __shared__ int indices[ELITE_COUNT * 2];
    __shared__ long long vals[ELITE_COUNT * 2];

    // Each block samples a portion of the population to find local candidates
    // To keep it simple and fast, we use a single-block approach for the global top
    if (tid < ELITE_COUNT * 2) {
        indices[tid] = tid * (POP_SIZE / (ELITE_COUNT * 2));
        vals[tid] = d_energies[indices[tid]];
    }
    __syncthreads();

    // Simple Bitonic-style sort for the small candidate pool
    for (int size = 2; size <= ELITE_COUNT * 2; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            int pos = tid;
            if (pos < ELITE_COUNT) {
                int i = (pos / stride) * (stride * 2) + (pos % stride);
                int j = i + stride;
                if (vals[i] > vals[j]) {
                    long long tv = vals[i]; vals[i] = vals[j]; vals[j] = tv;
                    int ti = indices[i]; indices[i] = indices[j]; indices[j] = ti;
                }
            }
            __syncthreads();
        }
    }

    if (tid < ELITE_COUNT) d_elite_indices[tid] = indices[tid];
}

__global__ void crossover_kernel(int N, int8_t* d_pop, int* d_elite_indices, curandState* states) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= POP_SIZE) return;

    // Don't mutate the absolute best (index 0)
    if (id == d_elite_indices[0]) return;

    curandState local_rng = states[id];
    int p1 = d_elite_indices[curand(&local_rng) % ELITE_COUNT];
    int p2 = d_elite_indices[curand(&local_rng) % ELITE_COUNT];

    for (int i = 0; i < N; i++) {
        int8_t gene = (curand_uniform(&local_rng) > 0.5f) ? d_pop[p1 * MAX_N + i] : d_pop[p2 * MAX_N + i];
        // Mutation
        if (curand_uniform(&local_rng) < 0.02f) gene = -gene;
        d_pop[id * MAX_N + i] = gene;
    }
    states[id] = local_rng;
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------

int main(int argc, char** argv) {
    long long MAX_ITERS_PER_N = (argc > 1) ? atoll(argv[1]) : 500000;
    
    int8_t *d_pop; 
    long long *d_energies; 
    curandState *d_states; 
    int *d_elites;
    
    CUDA_CHECK(cudaMalloc(&d_pop, POP_SIZE * MAX_N));
    CUDA_CHECK(cudaMalloc(&d_energies, POP_SIZE * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_states, POP_SIZE * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_elites, ELITE_COUNT * sizeof(int)));

    size_t smem_size = (MAX_N * 2) + (MAX_N * 4 * 2) + 128;

    std::cout << std::left << std::setw(5) << "N" << std::setw(10) << "Target" 
              << std::setw(10) << "Found" << std::setw(12) << "Time(s)" << "MTPS (Moves/sec)" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    for (int N = 3; N <= 66; N++) {
        auto start_n = std::chrono::high_resolution_clock::now();
        
        init_kernel<<<POP_SIZE / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(N, d_pop, d_states, (unsigned long)time(NULL));
        
        long long iter_count = 0;
        long long h_best_e = 2e18;
        
        while (iter_count < MAX_ITERS_PER_N) {
            tabu_kernel<<<POP_SIZE, THREADS_PER_BLOCK, smem_size>>>(N, MEMETIC_FREQ, d_pop, d_energies, d_states);
            
            // Check global best energy (Only one value copied to host for exit condition)
            select_elites_kernel<<<1, ELITE_COUNT>>>(d_energies, d_elites);
            int best_idx;
            cudaMemcpy(&best_idx, d_elites, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_best_e, &d_energies[best_idx], sizeof(long long), cudaMemcpyDeviceToHost);

            iter_count += MEMETIC_FREQ;
            if (h_best_e <= TARGETS[N]) break;

            crossover_kernel<<<POP_SIZE / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(N, d_pop, d_elites, d_states);
        }

        auto end_n = std::chrono::high_resolution_clock::now();
        double dur = std::chrono::duration<double>(end_n - start_n).count();
        double mtps = (double)POP_SIZE * iter_count / (dur * 1e6);

        std::cout << std::left << std::setw(5) << N << std::setw(10) << TARGETS[N] 
                  << std::setw(10) << h_best_e << std::fixed << std::setprecision(3) 
                  << std::setw(12) << dur << mtps << std::endl;
    }

    cudaFree(d_pop); cudaFree(d_energies); cudaFree(d_states); cudaFree(d_elites);
    return 0;
}
