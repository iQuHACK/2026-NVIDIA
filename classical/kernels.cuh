#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cstdint>
#include <cuda_runtime.h>

// Forward declarations
__global__ void memetic_search_kernel(int N, int target_energy, int *stop_flag,
                                      int *global_best_energy, uint64_t *seeds,
                                      uint32_t *global_best_seq,
                                      long long *log_time, int *log_energy,
                                      int *log_count, int *d_lock,
                                      long long *start_clk);

#endif // KERNELS_CUH
