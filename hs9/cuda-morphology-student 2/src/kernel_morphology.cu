#include "kernel_morphology.h"

/*
//Computes the minimum value in the neighborhood.
void kernel_erosion(const unsigned char* Frame_in, unsigned char* Frame_out, int height, int width, int* Mask, int mask_radius)
{
    int x = 0;
    int y = 0;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {

            // Dummy copy code
            Frame_out[y * width + x] = Frame_in[y * width + x];

        }
    }
}
//Computes the maximum value in the neighborhood.
void kernel_dilation(const unsigned char* Frame_in, unsigned char* Frame_out, int height, int width, int* Mask, int mask_radius)
{
    int x = 0;
    int y = 0;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {

            // Dummy copy code
            Frame_out[y * width + x] = Frame_in[y * width + x];

        }
    }
}
*/

__global__ 
void kernel_erosion(const unsigned char* Frame_in, unsigned char* Frame_out, int height, int width, int* Mask, int mask_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int min_val = 255;
        for (int dy = -mask_radius; dy <= mask_radius; dy++) {
            for (int dx = -mask_radius; dx <= mask_radius; dx++) {
                if (Mask[(dy + mask_radius) * (2 * mask_radius + 1) + (dx + mask_radius)]) {
                    int nx = min(max(x + dx, 0), width - 1);
                    int ny = min(max(y + dy, 0), height - 1);
                    min_val = min(min_val, (int)Frame_in[ny * width + nx]);
                }
            }
        }
        Frame_out[y * width + x] = (unsigned char)min_val;
    }
}

__global__ 
void kernel_dilation(const unsigned char* Frame_in, unsigned char* Frame_out, int height, int width, int* Mask, int mask_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int max_val = 0;
        for (int dy = -mask_radius; dy <= mask_radius; dy++) {
            for (int dx = -mask_radius; dx <= mask_radius; dx++) {
                if (Mask[(dy + mask_radius) * (2 * mask_radius + 1) + (dx + mask_radius)]) {
                    int nx = min(max(x + dx, 0), width - 1);
                    int ny = min(max(y + dy, 0), height - 1);
                    max_val = max(max_val, (int)Frame_in[ny * width + nx]);
                }
            }
        }
        Frame_out[y * width + x] = (unsigned char)max_val;
    }
}
