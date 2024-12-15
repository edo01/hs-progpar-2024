#include "cppdefs.h"
#include "cuda_kernels.h"
EXTERN
{
#include "easypap.h"
}
/*
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

*/
#include "cppdefs.h"
#include "cuda_kernels.h"
EXTERN
{
#include "easypap.h"
}
/*
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
*/

__global__ void mandelbrot_cuda(float* img, int width, int height, int max_iter, float leftX, float rightX, float topY, float bottomY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float x_step = (rightX - leftX) / width;
    float y_step = (topY - bottomY) / height;

    if (x < width && y < height / 2) {
        // Upper part of the image
        float cr = leftX + x * x_step;
        float ci = topY - y * y_step;
        float zr = 0.0, zi = 0.0;
        int iter = 0;
        while (iter < max_iter) {
            float zr2 = zr * zr, zi2 = zi * zi;
            if (zr2 + zi2 > 4.0f) break;
            float temp = zr2 - zi2 + cr;
            zi = 2.0f * zr * zi + ci;
            zr = temp;
            iter++;
        }
        img[y * width + x] = (float)iter / max_iter;

        // Lower half of the image
        int y_bottom = y + height / 2;
        ci = topY - y_bottom * y_step;
        zr = zi = 0.0f;
        iter = 0;
        while (iter < max_iter) {
            float zr2 = zr * zr, zi2 = zi * zi;
            if (zr2 + zi2 > 4.0f) break;
            float temp = zr2 - zi2 + cr;
            zi = 2.0f * zr * zi + ci;
            zr = temp;
            iter++;
        }
        img[y_bottom * width + x] = (float)iter / max_iter;
    }
}


EXTERN unsigned mandel_compute_cuda(unsigned nb_iter) {
    cudaError_t ret;
    dim3 grid  = {GPU_SIZE_X / TILE_W, GPU_SIZE_Y / (TILE_H * 2), 1}; // 高度分为两部分
    dim3 block = {TILE_W, TILE_H, 1};

    cudaSetDevice(cuda_device(0));

    uint64_t clock = monitoring_start_tile(easypap_gpu_lane(0));

    for (int i = 0; i < nb_iter; i++) {
        mandelbrot_cuda<<<grid, block, 0, cuda_stream(0)>>>(cuda_cur_buffer(0),DIM,DIM, MAX_ITERATIONS,leftX,rightX,topY, bottomY);
        ret = cudaStreamSynchronize(cuda_stream(0));
        check(ret, "cudaStreamSynchronize");

        zoom();
    }

    monitoring_end_tile(clock, 0, 0, DIM, DIM, easypap_gpu_lane(0));

    return 0;
}


