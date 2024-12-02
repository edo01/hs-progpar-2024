#include "easypap.h"

#include <omp.h>

/* Task 2.4 Local Memory – transpose Kernel
 *
 * Task #1 CPU vs GPU (naive versions) 
 * 
 * 
 * Task #2
 * 
*/

__global__ void transpose_cuda_tiled(unsigned *in, unsigned *out, unsigned dim)
{
     // Shared memory tile
    __shared__ unsigned tile[TILE_H][TILE_W]; 
    // global thread 
    unsigned xg = blockIdx.x * TILE_SIZE + threadIdx.x;
    unsigned yg = blockIdx.y * TILE_SIZE + threadIdx.y;
    // Local thread 
    unsigned xl = threadIdx.x;  
    unsigned yl = threadIdx.y;
    // Step 1: Load data from global memory to shared memory
    if (xg < dim && yg < dim) {
        tile[yl][xl] = in[yg * dim + xg];
    }
    __syncthreads(); // Synchronize threads to ensure all data is loaded

    // Step 2: Perform transposition within shared memory
    xg = blockIdx.y * TILE_H + threadIdx.x;
    yg = blockIdx.x * TILE_W + threadIdx.y;
    // Step 3: Write transposed data from shared memory back to global memory
    if (xg < dim && yg < dim) {
        out[yg * dim + xg] = tile[xl][yl];
    }
}

    
EXTERN unsigned transpose_compute_cuda_tiled  (unsigned *in, unsigned *out, unsigned dim)
{
  cudaError_t ret;
  dim3 grid  = {GPU_SIZE_X / TILE_W, GPU_SIZE_Y / TILE_H, 1};
  dim3 block = {TILE_W, TILE_H, 1};

  ret = cudaSetDevice (cuda_device (0));
  check (ret, "cudaSetDevice");

  uint64_t clock = monitoring_start_tile (easypap_gpu_lane (0));

  for (int i = 0; i < nb_iter; i++)
    transpose_cuda_tiled<<<grid, block, 0,
                    cuda_stream (0)>>> (in, out, dim);

  ret = cudaStreamSynchronize (cuda_stream (0));
  check (ret, "cudaStreamSynchronize");

  monitoring_end_tile (clock, 0, 0, DIM, DIM, easypap_gpu_lane (0));

  return 0;
}