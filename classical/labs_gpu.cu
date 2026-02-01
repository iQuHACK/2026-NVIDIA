#include "kernels.cuh"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  }

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <N> [target_energy]" << std::endl;
    return 1;
  }

  int N = std::atoi(argv[1]);
  int target_energy = (argc > 2) ? std::atoi(argv[2]) : -1000000;

  std::cout << "Solving LABS for N=" << N
            << " with target energy=" << target_energy << std::endl;

  // GPU properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Using GPU: " << prop.name << std::endl;

  // Define Grid and Block dimensions
  // The paper suggests using many blocks (replicas).
  // Threads per block should be enough to cover the neighborhood check or just
  // many parallel checks. Ideally min(N, 1024) or similar. For now let's say
  // 128 threads per block.
  int threads_per_block = 128;
  int num_blocks = prop.multiProcessorCount * 32; // Occupancy factor

  std::cout << "Launching " << num_blocks << " blocks with "
            << threads_per_block << " threads each." << std::endl;

  // Global termination flag
  int *d_stop_flag;
  int h_stop_flag = 0;
  CHECK_CUDA(cudaMalloc(&d_stop_flag, sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_stop_flag, &h_stop_flag, sizeof(int),
                        cudaMemcpyHostToDevice));

  // Result storage (best found)
  // Sequence
  int ints_per_seq = (N + 31) / 32;
  uint32_t *d_best_seq;
  CHECK_CUDA(cudaMalloc(&d_best_seq, ints_per_seq * sizeof(uint32_t)));
  CHECK_CUDA(cudaMemset(d_best_seq, 0, ints_per_seq * sizeof(uint32_t)));

  // Logging
  long long *d_log_time;
  int *d_log_energy;
  int *d_log_count;
  int *d_lock;
  long long *d_start_clk;

  CHECK_CUDA(cudaMalloc(&d_log_time, 100000 * sizeof(long long)));
  CHECK_CUDA(cudaMalloc(&d_log_energy, 100000 * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_log_count, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_lock, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_start_clk, sizeof(long long)));

  CHECK_CUDA(cudaMemset(d_log_count, 0, sizeof(int)));
  CHECK_CUDA(cudaMemset(d_lock, 0, sizeof(int)));

  // Result storage (best found)
  int *d_global_best_energy;
  int h_global_best_energy = 9999999;
  CHECK_CUDA(cudaMalloc(&d_global_best_energy, sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_global_best_energy, &h_global_best_energy,
                        sizeof(int), cudaMemcpyHostToDevice));

  // Seeds
  uint64_t *d_seeds;
  CHECK_CUDA(cudaMalloc(&d_seeds, num_blocks * sizeof(uint64_t)));

  // Initialize seeds on host
  std::vector<uint64_t> h_seeds(num_blocks);
  std::random_device rd;
  std::mt19937_64 gen(rd());
  for (int i = 0; i < num_blocks; ++i)
    h_seeds[i] = gen();
  CHECK_CUDA(cudaMemcpy(d_seeds, h_seeds.data(), num_blocks * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));

  // Launch Kernel
  auto start_time = std::chrono::high_resolution_clock::now();

  // Calculate Shared Memory Size
  int pop_size = 100;
  // Layout matched with kernels.cu:
  // pop_seqs: pop_size * ints_per_seq
  // pop_energies: pop_size
  // child_seq: ints_per_seq
  // best_tabu_seq: ints_per_seq
  // vectorC: N
  // tabu_list: N
  // shared_reduction: 64 (safe margin)

  size_t shared_mem_ints = (pop_size * ints_per_seq) + pop_size + ints_per_seq +
                           ints_per_seq + N + N + 64;
  size_t shared_mem_bytes = shared_mem_ints * sizeof(int);

  std::cout << "Shared Memory per Block: " << shared_mem_bytes / 1024.0 << " KB"
            << std::endl;

  if (shared_mem_bytes >
      48 * 1024) { // Standard 48KB limit, though A100 has more
    std::cout << "WARNING: Shared memory request might exceed standard 48KB "
                 "limit depending on Config."
              << std::endl;
    // Adjust logic if needed or require larger shared mem config on host
    cudaFuncSetAttribute(memetic_search_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shared_mem_bytes);
  }

  // Launch Kernel
  memetic_search_kernel<<<num_blocks, threads_per_block, shared_mem_bytes>>>(
      N, target_energy, d_stop_flag, d_global_best_energy, d_seeds, d_best_seq,
      d_log_time, d_log_energy, d_log_count, d_lock, d_start_clk);

  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaDeviceSynchronize());
  auto end_time = std::chrono::high_resolution_clock::now();

  // Retrieve Start Cycle
  long long h_start_clk = 0;
  CHECK_CUDA(cudaMemcpy(&h_start_clk, d_start_clk, sizeof(long long),
                        cudaMemcpyDeviceToHost));
  std::cout << "Start Cycle: " << h_start_clk << std::endl;

  CHECK_CUDA(cudaMemcpy(&h_global_best_energy, d_global_best_energy,
                        sizeof(int), cudaMemcpyDeviceToHost));

  std::chrono::duration<double> elapsed = end_time - start_time;
  std::cout << "Finished in " << elapsed.count() << "s" << std::endl;
  std::cout << "Best Energy Found: " << h_global_best_energy << std::endl;
  // Calculate Merit Factor F = N^2 / (2 * E)
  if (h_global_best_energy > 0)
    std::cout << "Merit Factor: "
              << (double)(N * N) / (2.0 * h_global_best_energy) << std::endl;

  // Retrieve and print Sequence
  std::vector<uint32_t> h_best_seq(ints_per_seq);
  CHECK_CUDA(cudaMemcpy(h_best_seq.data(), d_best_seq,
                        ints_per_seq * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));

  std::cout << "Best Sequence: ";
  for (int i = 0; i < N; ++i) {
    int word = i / 32;
    int bit = i % 32;
    int val = (h_best_seq[word] >> bit) & 1;
    std::cout << (val ? "+" : "-");
  }
  std::cout << std::endl;

  // Retrieve Logs
  int h_log_count = 0;
  CHECK_CUDA(cudaMemcpy(&h_log_count, d_log_count, sizeof(int),
                        cudaMemcpyDeviceToHost));

  if (h_log_count > 100000)
    h_log_count = 100000;

  std::vector<long long> h_log_time(h_log_count);
  std::vector<int> h_log_energy(h_log_count);

  CHECK_CUDA(cudaMemcpy(h_log_time.data(), d_log_time,
                        h_log_count * sizeof(long long),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_log_energy.data(), d_log_energy,
                        h_log_count * sizeof(int), cudaMemcpyDeviceToHost));

  std::cout << "ClockRate: " << (long long)prop.clockRate * 1000 << " Hz"
            << std::endl;
  std::cout << "Convergence History (Cycle, Energy):" << std::endl;

  // Sort logs by time (since parallelism might reorder slightly)
  // Simple bubble sort or pair sort
  std::vector<std::pair<long long, int>> logs(h_log_count);
  for (int i = 0; i < h_log_count; ++i)
    logs[i] = {h_log_time[i], h_log_energy[i]};
  std::sort(logs.begin(), logs.end());

  for (const auto &p : logs) {
    std::cout << p.first << ", " << p.second << std::endl;
  }

  cudaFree(d_stop_flag);
  cudaFree(d_global_best_energy);
  cudaFree(d_seeds);
  cudaFree(d_best_seq);
  cudaFree(d_log_time);
  cudaFree(d_log_energy);
  cudaFree(d_log_count);
  cudaFree(d_lock);
  cudaFree(d_start_clk);

  return 0;
}
