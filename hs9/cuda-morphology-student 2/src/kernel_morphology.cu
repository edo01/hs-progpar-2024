#include "kernel_morphology.h"

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