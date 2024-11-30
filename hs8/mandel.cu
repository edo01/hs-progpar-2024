#include "cppdefs.h"
#include "cuda_kernels.h"
EXTERN
{
#include "easypap.h"
}

__global__ void mandel_cuda(unsigned *img, float leftX, float rightX, float topY, float bottomY, int DIM, int MAX_ITERATIONS) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 

    if (x < DIM && y < DIM) {
        // Calculate pixel coordinates in the complex plane
        float xstep = (rightX - leftX) / DIM;
        float ystep = (topY - bottomY) / DIM;

        float cr = leftX + xstep * x;
        float ci = topY - ystep * y;
        float zr = 0.0f, zi = 0.0f;

        int iter;
        for (iter = 0; iter < MAX_ITERATIONS; iter++) {
            float x2 = zr * zr;
            float y2 = zi * zi;

            if (x2 + y2 > 4.0f) break;

            float twoxy = 2.0f * zr * zi;
            zr = x2 - y2 + cr;
            zi = twoxy + ci;
        }

        img[y * DIM + x] = ezv_rgb(iter % 256, (iter * 2) % 256, (iter * 4) % 256);
    }
}


EXTERN unsigned mandel_compute_cuda(unsigned nb_iter) {
    dim3 grid  = {GPU_SIZE_X / TILE_W, GPU_SIZE_Y / TILE_H, 1};
    dim3 block = {TILE_W, TILE_H, 1};

    for (unsigned it = 0; it < nb_iter; it++) {
        mandel_cuda<<<grid, block>>>(cuda_cur_buffer(0), leftX, rightX, topY, bottomY, DIM, MAX_ITERATIONS);
        cudaDeviceSynchronize();
        zoom();
    }

    return 0;
}
