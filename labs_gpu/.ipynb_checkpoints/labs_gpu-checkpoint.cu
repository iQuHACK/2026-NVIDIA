#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <iomanip>

// Hardware Optimizations (A100/A6000 Compatible)
#define MAX_N 512
#define THREADS 128
#define POP_SIZE 8192       
#define MEMETIC_FREQ 1000   
#define ELITE_COUNT 128     
#define TABU_MIN 10
#define TABU_VAR 20

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s in %s at line %d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

// Target Energy Table from provided image (N=0 to 66)
const int TARGETS[] = {
    0, 0, 0, 1, 2, 2, 7, 3, 8, 12, 13, 5, 10, 6, 19, 15, 24, 32, 25, 29, 26, 
    26, 39, 47, 36, 36, 45, 37, 50, 62, 59, 67, 64, 64, 65, 73, 82, 86, 87, 99, 
    108, 180, 101, 109, 122, 118, 131, 135, 140, 136, 153, 153, 166, 170, 175, 
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

__global__ void verify_kernel(int N, int8_t* d_pop, long long* d_energies) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ int8_t s_seq[MAX_N];
    __shared__ unsigned long long s_energy;
    if (tid < N) s_seq[tid] = (d_pop[bid * MAX_N + tid] > 0) ? 1 : -1;
    if (tid == 0) s_energy = 0;
    __syncthreads();
    unsigned long long local_sum = 0;
    for (int k = tid + 1; k < N; k += blockDim.x) {
        int ck = 0;
        for (int i = 0; i < N - k; i++) ck += (int)s_seq[i] * (int)s_seq[i + k];
        local_sum += (unsigned long long)(ck * ck);
    }
    atomicAdd(&s_energy, local_sum);
    __syncthreads();
    if (tid == 0) d_energies[bid] = (long long)s_energy;
}

__device__ __forceinline__ void warpReduceMin(long long &val, int &idx) {
    for (int offset = 16; offset > 0; offset /= 2) {
        long long o_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
        int o_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
        if (o_val < val) { val = o_val; idx = o_idx; }
    }
}

__global__ void tabu_kernel(int N, int iterations, int8_t* d_pop, curandState* states) {
    extern __shared__ char smem[];
    int8_t* s_seq = (int8_t*)smem; 
    int8_t* s_best_seq = (int8_t*)&s_seq[MAX_N]; 
    int* s_corr = (int*)((((size_t)&s_best_seq[MAX_N]) + 7) & ~7); 
    int* s_tabu = (int*)&s_corr[MAX_N];

    __shared__ long long s_cur_e; __shared__ long long s_best_e;
    __shared__ long long s_warp_d[8]; 
    __shared__ int s_warp_i[8];    __shared__ bool s_upd;

    int bid = blockIdx.x; int tid = threadIdx.x;
    curandState rng = states[bid];

    if (tid < N) {
        s_seq[tid] = (d_pop[bid * MAX_N + tid] > 0) ? 1 : -1;
        s_best_seq[tid] = s_seq[tid];
        s_tabu[tid] = 0;
    }
    __syncthreads();

    if (tid > 0 && tid < N) {
        int ck = 0;
        for (int i = 0; i < N - tid; i++) ck += s_seq[i] * s_seq[i + tid];
        s_corr[tid] = ck;
    }
    __syncthreads();

    if (tid == 0) {
        long long e = 0;
        for (int k = 1; k < N; k++) e += (long long)s_corr[k] * s_corr[k];
        s_cur_e = e; s_best_e = e;
    }
    __syncthreads();

    for (int iter = 0; iter < iterations; iter++) {
        long long my_d = LLONG_MAX; int my_i = -1;
        if (tid < N) {
            long long dE = 0; int s_p = s_seq[tid];
            for (int k = 1; k < N; k++) {
                int nb = 0;
                if (tid + k < N) nb += s_seq[tid + k];
                if (tid - k >= 0) nb += s_seq[tid - k];
                int dCk = -2 * s_p * nb;
                dE += (long long)dCk * (2 * s_corr[k] + dCk);
            }
            if (iter >= s_tabu[tid] || (s_cur_e + dE < s_best_e)) { my_d = dE; my_i = tid; }
        }
        warpReduceMin(my_d, my_i);
        if ((tid & 31) == 0) { s_warp_d[tid >> 5] = my_d; s_warp_i[tid >> 5] = my_i; }
        __syncthreads();
        if (tid == 0) {
            long long bd = LLONG_MAX; int bp = -1;
            // INCREASE LOOP LIMIT FROM 4 TO 8 (or blockDim.x / 32)
            for (int w = 0; w < (blockDim.x >> 5); w++) { 
                 if (s_warp_d[w] < bd) { bd = s_warp_d[w]; bp = s_warp_i[w]; }
            }
            s_warp_i[0] = bp; 
            if (bp != -1) {
                s_warp_i[1] = s_seq[bp]; s_cur_e += bd;
                s_seq[bp] = -s_seq[bp]; 
                s_tabu[bp] = iter + TABU_MIN + (curand(&rng) % TABU_VAR);
                if (s_cur_e < s_best_e) { s_best_e = s_cur_e; s_upd = true; } else s_upd = false;
            }
        }
        __syncthreads();
        int move_p = s_warp_i[0];
        if (move_p == -1) break;
        if (s_upd && tid < N) s_best_seq[tid] = s_seq[tid];
        if (tid > 0 && tid < N) {
            int nb = 0;
            if (move_p + tid < N) nb += s_seq[move_p + tid];
            if (move_p - tid >= 0) nb += s_seq[move_p - tid];
            s_corr[tid] += -2 * s_warp_i[1] * nb;
        }
        __syncthreads();
    }
    if (tid < N) d_pop[bid * MAX_N + tid] = s_best_seq[tid];
    if (tid == 0) states[bid] = rng;
}

__global__ void crossover_kernel(int N, int8_t* d_pop, int* d_elite_indices, curandState* states) {
    int bid = blockIdx.x; int tid = threadIdx.x;
    __shared__ bool elite;
    if(tid==0) {
        elite = false;
        for(int i = 0; i < ELITE_COUNT; i++) if(bid == d_elite_indices[i]) elite = true;
    }
    __syncthreads();
    if (!elite && tid < N) {
        curandState local_rng = states[bid];
        int p1 = d_elite_indices[curand(&local_rng) % ELITE_COUNT];
        int p2 = d_elite_indices[curand(&local_rng) % ELITE_COUNT];
        int8_t g = (curand_uniform(&local_rng) > 0.5f) ? d_pop[p1*MAX_N+tid] : d_pop[p2*MAX_N+tid];
        if (curand_uniform(&local_rng) < 0.01f) g = -g;
        d_pop[bid*MAX_N+tid] = (g > 0) ? 1 : -1;
        if (tid == 0) states[bid] = local_rng;
    }
}

// -------------------------------------------------------------------------
// Main Benchmark Logic
// -------------------------------------------------------------------------
int main(int argc, char** argv) {
    long long MAX_ITERS_PER_N = (argc > 1) ? atoll(argv[1]) : 1000000;
    
    std::ofstream csv("labs_results.csv");
    csv << "N,Target_E,Found_E,Status,Time_sec,Moves_Million" << std::endl;

    std::cout << std::left << std::setw(5) << "N" << std::setw(10) << "Target" 
              << std::setw(10) << "Found" << std::setw(10) << "Status" 
              << std::setw(12) << "Time(s)" << "Moves(M)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // Allocate GPU buffers once
    int8_t *d_pop; long long *d_energies; curandState *d_states; int *d_elites;
    CUDA_CHECK(cudaMalloc(&d_pop, POP_SIZE * MAX_N));
    CUDA_CHECK(cudaMalloc(&d_energies, POP_SIZE * 8));
    CUDA_CHECK(cudaMalloc(&d_states, POP_SIZE * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_elites, ELITE_COUNT * 4));
    size_t smem = (MAX_N * 2) + (MAX_N * 8) + 64;

    double total_wall_time = 0;
    unsigned long long total_moves = 0;

    for (int N = 3; N <= 64; N++) {
        int target_e = TARGETS[N];
        long long current_best = LLONG_MAX;
        
        auto start_n = std::chrono::high_resolution_clock::now();
        
        // Init N-specific population
        init_kernel<<<POP_SIZE/THREADS, THREADS>>>(N, d_pop, d_states, (unsigned long)time(NULL));
        CUDA_CHECK(cudaDeviceSynchronize());

        long long iter_count = 0;
        bool success = false;

        while (iter_count < MAX_ITERS_PER_N) {
            tabu_kernel<<<POP_SIZE, THREADS, smem>>>(N, MEMETIC_FREQ, d_pop, d_states);
            verify_kernel<<<POP_SIZE, THREADS>>>(N, d_pop, d_energies);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<long long> h_e(POP_SIZE);
            cudaMemcpy(h_e.data(), d_energies, POP_SIZE * 8, cudaMemcpyDeviceToHost);
            std::vector<int> idx(POP_SIZE); std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int a, int b) { return h_e[a] < h_e[b]; });

            current_best = h_e[idx[0]];
            iter_count += MEMETIC_FREQ;

            if (current_best <= target_e) { success = true; break; }

            cudaMemcpy(d_elites, idx.data(), ELITE_COUNT * 4, cudaMemcpyHostToDevice);
            crossover_kernel<<<POP_SIZE/THREADS, THREADS>>>(N, d_pop, d_elites, d_states);
        }

        auto end_n = std::chrono::high_resolution_clock::now();
        double dur_n = std::chrono::duration<double>(end_n - start_n).count();
        double moves_m = (double)POP_SIZE * iter_count / 1e6;

        total_wall_time += dur_n;
        total_moves += (unsigned long long)(POP_SIZE * iter_count);

        // Print to Terminal
        std::cout << std::left << std::setw(5) << N << std::setw(10) << target_e 
                  << std::setw(10) << current_best << std::setw(10) << (success ? "PASS" : "FAIL") 
                  << std::fixed << std::setprecision(3) << std::setw(12) << dur_n << moves_m << std::endl;

        // Save to CSV
        csv << N << "," << target_e << "," << current_best << "," << (success ? "Success" : "Timeout") 
            << "," << dur_n << "," << moves_m << std::endl;
    }

    std::cout << std::string(60, '-') << std::endl;
    std::cout << "TOTAL WALL TIME: " << total_wall_time << " seconds" << std::endl;
    std::cout << "TOTAL OPERATIONS: " << total_moves / 1e6 << " Million moves" << std::endl;
    std::cout << "CSV saved to labs_results.csv" << std::endl;

    cudaFree(d_pop); cudaFree(d_energies); cudaFree(d_states); cudaFree(d_elites);
    return 0;
}