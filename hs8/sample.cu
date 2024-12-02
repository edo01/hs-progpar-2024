#include "cppdefs.h"
#include "cuda_kernels.h"
EXTERN
{
#include "easypap.h"
}

// Suggested cmdline:
//   ./run -k sample -g -i 1
static __global__ void sample_cuda (unsigned *img, unsigned DIM)
{
  unsigned x = gpu_get_col ();
  unsigned y = gpu_get_row ();

  //img[y * DIM + x] = rgb (255, 255, 0);
  img[y * DIM + x] = rgb (255, 0, 0);
}

// We redefine the kernel launcher function because
// the kernel onlu uses a single image
EXTERN unsigned sample_compute_cuda (unsigned nb_iter)
{
  cudaError_t ret;
  dim3 grid  = {GPU_SIZE_X / TILE_W, GPU_SIZE_Y / TILE_H, 1};
  dim3 block = {TILE_W, TILE_H, 1};

  cudaSetDevice (cuda_device (0));

  uint64_t clock = monitoring_start_tile (easypap_gpu_lane (0));

  for (int i = 0; i < nb_iter; i++)
    sample_cuda<<<grid, block, 0, cuda_stream (0)>>> (cuda_cur_buffer (0), DIM);

  // FIXME: should only be performed when monitoring/tracing is activated
  ret = cudaStreamSynchronize (cuda_stream (0));
  check (ret, "cudaStreamSynchronize");

  monitoring_end_tile (clock, 0, 0, DIM, DIM, easypap_gpu_lane (0));

  return 0;
}


__global__ void sample_cuda_grad(unsigned *img, unsigned DIM) {
    unsigned x = gpu_get_col();
    unsigned y = gpu_get_row();

    uint8_t red = x & 255; //  the greater x, the more red we use
    uint8_t blue = y & 255; // the greater y, the more blue we use
    img[y * DIM + x] = rgb(red, 0, blue);
}



__global__ void sample_cuda_grad_ydiv2(unsigned *img, unsigned DIM) {
    unsigned x = gpu_get_col();
    unsigned y = gpu_get_row();// each thread is responsible for two pixels

    uint8_t red1 = x & 255;
    uint8_t blue1 = y & 255;
    img[y * DIM + x] = rgb(red1, 0, blue1);
    y = y + DIM/2
    uint8_t red2 = x & 255;
    uint8_t blue2 = (y) & 255;
    img[(y) * DIM + x] = rgb(red2, 0, blue2);
}



EXTERN unsigned sample_compute_cuda_grad_ydiv2(unsigned nb_iter) {
    cudaError_t ret;

    // Halve the grid size in the y direction
    dim3 grid = {GPU_SIZE_X / TILE_W, GPU_SIZE_Y / (TILE_H * 2), 1};
    dim3 block = {TILE_W, TILE_H, 1};

    cudaSetDevice(cuda_device(0));

    uint64_t clock = monitoring_start_tile(easypap_gpu_lane(0));

    for (int i = 0; i < nb_iter; i++) {
        sample_cuda_grad_ydiv2<<<grid, block, 0, cuda_stream(0)>>>(cuda_cur_buffer(0), DIM);
    }

    ret = cudaStreamSynchronize(cuda_stream(0));
    check(ret, "cudaStreamSynchronize");

    monitoring_end_tile(clock, 0, 0, DIM, DIM, easypap_gpu_lane(0));

    return 0;
}
