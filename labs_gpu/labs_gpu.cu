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
// POP_SIZE is now a runtime variable
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

// Target Energy Table (N=0 to 66)
const int TARGETS[] = {
    0, 0, 0, 1, 2, 2, 7, 3, 8, 12, 13, 5, 10, 6, 19, 15, 24, 32, 25, 29, 26, 
    26, 39, 47, 36, 36, 45, 37, 50, 62, 59, 67, 64, 64, 65, 73, 82, 86, 87, 99, 
    108, 180, 101, 109, 122, 118, 131, 135, 140, 136, 153, 153, 166, 170, 175, 
    171, 192, 188, 197, 205, 218, 226, 235, 207, 208, 240, 257
};

// -------------------------------------------------------------------------
// GPU Kernels
// -------------------------------------------------------------------------

// Added pop_size argument
__global__ void init_kernel(int N, int pop_size, int8_t* d_pop, curandState* states, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < pop_size) {
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
    
    // Stride loop to support N > 128
    for (int i = tid; i < N; i += blockDim.x) {
        s_seq[i] = (d_pop[bid * MAX_N + i] > 0) ? 1 : -1;
    }
    
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

__launch_bounds__(THREADS, 16)
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

    // Load Sequence (Strided for N > 128)
    for (int i = tid; i < N; i += blockDim.x) {
        s_seq[i] = (d_pop[bid * MAX_N + i] > 0) ? 1 : -1;
        s_best_seq[i] = s_seq[i];
        s_tabu[i] = 0;
    }
    __syncthreads();

    // Initial Correlations (Strided)
    for (int i = tid; i < N; i += blockDim.x) {
        if (i > 0) {
            int ck = 0;
            for (int j = 0; j < N - i; j++) ck += s_seq[j] * s_seq[j + i];
            s_corr[i] = ck;
        }
    }
    __syncthreads();

    // Initial Energy
    if (tid == 0) {
        long long e = 0;
        for (int k = 1; k < N; k++) e += (long long)s_corr[k] * s_corr[k];
        s_cur_e = e; s_best_e = e;
    }
    __syncthreads();

    for (int iter = 0; iter < iterations; iter++) {
        long long my_d = LLONG_MAX; int my_i = -1;
        
        // Evaluate Moves (Strided)
        for (int i = tid; i < N; i += blockDim.x) {
            long long dE = 0; int s_p = s_seq[i];
            for (int k = 1; k < N; k++) {
                int nb = 0;
                if (i + k < N) nb += s_seq[i + k];
                if (i - k >= 0) nb += s_seq[i - k];
                int dCk = -2 * s_p * nb;
                dE += (long long)dCk * (2 * s_corr[k] + dCk);
            }
            if (iter >= s_tabu[i] || (s_cur_e + dE < s_best_e)) { 
                if (dE < my_d) { my_d = dE; my_i = i; }
            }
        }
        
        warpReduceMin(my_d, my_i);
        if ((tid & 31) == 0) { s_warp_d[tid >> 5] = my_d; s_warp_i[tid >> 5] = my_i; }
        __syncthreads();
        
        if (tid == 0) {
            long long bd = LLONG_MAX; int bp = -1;
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
        
        // Update Best (Strided)
        if (s_upd) {
            for (int i = tid; i < N; i += blockDim.x) s_best_seq[i] = s_seq[i];
        }
        
        // Update Correlations (Strided)
        for (int i = tid; i < N; i += blockDim.x) {
            if (i > 0) {
                int nb = 0;
                if (move_p + i < N) nb += s_seq[move_p + i];
                if (move_p - i >= 0) nb += s_seq[move_p - i];
                s_corr[i] += -2 * s_warp_i[1] * nb;
            }
        }
        __syncthreads();
    }
    
    // Store Result (Strided)
    for (int i = tid; i < N; i += blockDim.x) {
        d_pop[bid * MAX_N + i] = s_best_seq[i];
    }
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
    
    if (!elite) {
        curandState local_rng = states[bid];
        // Stride loop to support N > 128
        for (int i = tid; i < N; i += blockDim.x) {
            int p1 = d_elite_indices[curand(&local_rng) % ELITE_COUNT];
            int p2 = d_elite_indices[curand(&local_rng) % ELITE_COUNT];
            int8_t g = (curand_uniform(&local_rng) > 0.5f) ? d_pop[p1*MAX_N+i] : d_pop[p2*MAX_N+i];
            if (curand_uniform(&local_rng) < 0.01f) g = -g;
            d_pop[bid*MAX_N+i] = (g > 0) ? 1 : -1;
        }
        if (tid == 0) states[bid] = local_rng;
    }
}

// -------------------------------------------------------------------------
// Helper: Binary Loader
// -------------------------------------------------------------------------
void load_warm_start(const char* filename, int8_t* d_pop, size_t expected_bytes) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open warm start file: " << filename << std::endl;
        exit(1);
    }
    
    std::vector<int8_t> h_buffer(expected_bytes);
    file.read((char*)h_buffer.data(), expected_bytes);
    
    if (file.gcount() != expected_bytes) {
        std::cerr << "Error: File size mismatch! Expected " << expected_bytes 
                  << " bytes, but read " << file.gcount() << " bytes. Check POP_SIZE." << std::endl;
        exit(1);
    }

    CUDA_CHECK(cudaMemcpy(d_pop, h_buffer.data(), expected_bytes, cudaMemcpyHostToDevice));
    std::cout << ">>> Loaded Warm Start Population from " << filename << std::endl;
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main(int argc, char** argv) {
    // New Usage: ./labs_solver [N] [POP_SIZE] [FILE] [ITERS (optional)]
    if (argc < 4) {
        std::cout << "Usage: ./labs_solver [N] [POP_SIZE] [WARM_START_FILE] [ITERS=1000000]" << std::endl;
        return 0;
    }

    int target_N = atoi(argv[1]);
    int pop_size = atoi(argv[2]);
    std::string warm_start_file = argv[3];
    long long MAX_ITERS_PER_N = (argc > 4) ? atoll(argv[4]) : 1000000;

    std::cout << "Running LABS Solver: N=" << target_N 
              << ", PopSize=" << pop_size 
              << ", Iters=" << MAX_ITERS_PER_N 
              << ", File=" << warm_start_file << std::endl;

    // 1. Memory Allocation
    int8_t *d_pop; long long *d_energies; curandState *d_states; int *d_elites;
    
    CUDA_CHECK(cudaMalloc(&d_pop, pop_size * MAX_N));
    CUDA_CHECK(cudaMalloc(&d_energies, pop_size * 8));
    CUDA_CHECK(cudaMalloc(&d_states, pop_size * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_elites, ELITE_COUNT * 4));
    
    size_t smem = (MAX_N * 2) + (MAX_N * 8) + 64;
    size_t total_pop_bytes = pop_size * MAX_N;

    int target_e = TARGETS[target_N];
    long long current_best = LLONG_MAX;
    
    // 2. Initialize RNG
    // Grid size = ceil(pop_size / threads)
    int init_grid = (pop_size + THREADS - 1) / THREADS;
    init_kernel<<<init_grid, THREADS>>>(target_N, pop_size, d_pop, d_states, (unsigned long)time(NULL));
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. Load Warm Start
    load_warm_start(warm_start_file.c_str(), d_pop, total_pop_bytes);

    // 4. Optimization Loop
    auto start_timer = std::chrono::high_resolution_clock::now();
    long long iter_count = 0;
    bool success = false;

    while (iter_count < MAX_ITERS_PER_N) {
        // Kernels launched with Grid=POP_SIZE (1 block per individual)
        tabu_kernel<<<pop_size, THREADS, smem>>>(target_N, MEMETIC_FREQ, d_pop, d_states);
        
        verify_kernel<<<pop_size, THREADS>>>(target_N, d_pop, d_energies);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Sort on Host
        std::vector<long long> h_e(pop_size);
        cudaMemcpy(h_e.data(), d_energies, pop_size * 8, cudaMemcpyDeviceToHost);
        
        std::vector<int> idx(pop_size); 
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return h_e[a] < h_e[b]; });

        current_best = h_e[idx[0]];
        iter_count += MEMETIC_FREQ;

        if (current_best <= target_e) { success = true; break; }

        cudaMemcpy(d_elites, idx.data(), ELITE_COUNT * 4, cudaMemcpyHostToDevice);
        
        // Crossover launches with Grid=POP_SIZE (1 block per individual)
        // Note: Previously this might have been undersized. Now it matches population.
        crossover_kernel<<<pop_size, THREADS>>>(target_N, d_pop, d_elites, d_states);
        
        if (iter_count % (MEMETIC_FREQ * 50) == 0) {
                std::cout << "Iter=" << iter_count/1e6 << "M Best=" << current_best << "\r" << std::flush;
        }
    }

    auto end_timer = std::chrono::high_resolution_clock::now();
    double dur = std::chrono::duration<double>(end_timer - start_timer).count();
    double moves_m = (double)pop_size * iter_count / 1e6;

    std::cout << std::left << std::setw(5) << target_N << std::setw(10) << target_e 
              << std::setw(10) << current_best << std::setw(10) << (success ? "PASS" : "FAIL") 
              << std::fixed << std::setprecision(3) << std::setw(12) << dur << moves_m << std::endl;

    cudaFree(d_pop); cudaFree(d_energies); cudaFree(d_states); cudaFree(d_elites);
    return 0;
}