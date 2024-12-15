#pragma once

#define KERNEL_THD_FLOPS 10//number of operations per threads
#define KERNEL_THD_BYTES 10//number of bytes per threads

__global__ void gpu_kernel_erosion(const unsigned char* Frame_in, unsigned char* Frame_out, int height, int width, int* Mask, int mask_radius);

__global__ void gpu_kernel_erosion_no_boundary(const unsigned char* Frame_in, unsigned char* Frame_out, int height, int width, int* Mask, int mask_radius);

__global__ void gpu_kernel_dilation(const unsigned char* Frame_in, unsigned char* Frame_out, int height, int width, int* Mask, int mask_radius);

__global__ void gpu_kernel_dilation_no_boundary(const unsigned char* Frame_in, unsigned char* Frame_out, int height, int width, int* Mask, int mask_radius);

void kernel_erosion(const unsigned char* Frame_in, unsigned char* Frame_out, int height, int width, int* Mask, int mask_radius);

void kernel_dilation(const unsigned char* Frame_in, unsigned char* Frame_out, int height, int width, int* Mask, int mask_radius);

