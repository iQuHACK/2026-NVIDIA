#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cstdint>
#include <cuda_runtime.h>

// Forward declarations
__global__ void memetic_search_kernel(int N, int target_energy, int *stop_flag,
                                      int *global_best_energy, uint64_t *seeds);

#endif // KERNELS_CUH
