#include <stdio.h>
#include <time.h>

#include "kernel_morphology.h"
#include "pgm.h"

int tile_width = 32;
int tile_height = 4;

int width_Frame = 1024;
int height_Frame = 436;
const char input_folder[]  = "temple_3/";
const char output_folder[] = "output_images/";
/*
int test_morphology(int argc, char** argv)
{
  // Timing variables
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float t = 0.f;
  clock_t fps_start, fps_end;
  double fps;

  // Various variables declarations and allocations
  char input_filename[128];
  char output_filename[128];

  unsigned char *h_Frame_in, *h_Frame_out, *h_Frame_tmp;

  for (int frame_num = 1; frame_num <= 50; ++frame_num) {
    // Start clock for full fps measure
    fps_start = clock();

    // Build erosion/dilation structuring element mask
    int h_Mask[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    int mask_radius = 1;

    // Build new in/out frames filename
    sprintf(input_filename , "%sframe_%04d.pgm", input_folder, frame_num);
    sprintf(output_filename, "%sframe_%04d.pgm", output_folder, frame_num);

    // Load the input image using LoadPGM_ui8matrix
    h_Frame_in  = LoadPGM_ui8matrix(input_filename, &height_Frame, &width_Frame);
    h_Frame_out = (unsigned char *)malloc(height_Frame * width_Frame * sizeof(unsigned char));
    h_Frame_tmp = (unsigned char *)malloc(height_Frame * width_Frame * sizeof(unsigned char));

    // Start CUDA event for kernel execution timing
    cudaEventRecord(start);

    // DUMMY Kernel launch configuration and execution
    // Erosion
    kernel_erosion(h_Frame_in, h_Frame_tmp, height_Frame, width_Frame, h_Mask, mask_radius);

    // Dilation
    kernel_dilation(h_Frame_tmp, h_Frame_out, height_Frame, width_Frame, h_Mask, mask_radius);

    // Stop CUDA event to measure kernel execution time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t, start, stop);

    // Save processed image
    SavePGM_ui8matrix(h_Frame_out, height_Frame, width_Frame, output_filename);


    // Free allocated memory for this frame
    free(h_Frame_in);
    free(h_Frame_out);
    free(h_Frame_tmp);

    // Stop clock for fps estimation
    fps_end = clock();
    fps = (double)CLOCKS_PER_SEC/(fps_end-fps_start);

    printf("Frame     : %04d\n", frame_num);
    printf("Time      : %f ms\n", t);
    printf("Pixel perf: %5.3f ns/pix\n", (t * 10e6) / (width_Frame * height_Frame));
    printf("FLOPs     : %5.3f GFlops\n", (KERNEL_THD_FLOPS * width_Frame * height_Frame * 10e3) / (t * 10e9) );
    printf("Bandwidth : %5.3f Go/s\n", (KERNEL_THD_BYTES * width_Frame * height_Frame * 10e3) / (t * 10e9));
    printf("FPS       : %5.2f\n\n", fps);
  }

  return 0;
}
*/

int test_morphology(int argc, char** argv) {
    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float t = 0.f;
    clock_t fps_start, fps_end;
    double fps;

    // Various variables declarations and allocations
    char input_filename[128];
    char output_filename[128];

    unsigned char *h_Frame_in, *h_Frame_out, *h_Frame_tmp;
    unsigned char *d_Frame_in, *d_Frame_out, *d_Frame_tmp;
    int *d_Mask;

    // Build erosion/dilation structuring element mask
    int h_Mask[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    int mask_radius = 1;

    // Allocate GPU memory for the mask
    cudaMalloc((void**)&d_Mask, sizeof(h_Mask));
    cudaMemcpy(d_Mask, h_Mask, sizeof(h_Mask), cudaMemcpyHostToDevice);

    for (int frame_num = 1; frame_num <= 50; ++frame_num) {
        // Start clock for full fps measure
        fps_start = clock();

        // Build new in/out frames filename
        sprintf(input_filename , "%sframe_%04d.pgm", input_folder, frame_num);
        sprintf(output_filename, "%sframe_%04d.pgm", output_folder, frame_num);

        // Load the input image using LoadPGM_ui8matrix
        h_Frame_in  = LoadPGM_ui8matrix(input_filename, &height_Frame, &width_Frame);
        h_Frame_out = (unsigned char *)malloc(height_Frame * width_Frame * sizeof(unsigned char));

        // Allocate GPU memory
        cudaMalloc((void**)&d_Frame_in, height_Frame * width_Frame * sizeof(unsigned char));
        cudaMalloc((void**)&d_Frame_tmp, height_Frame * width_Frame * sizeof(unsigned char));
        cudaMalloc((void**)&d_Frame_out, height_Frame * width_Frame * sizeof(unsigned char));

        // Copy input data to GPU
        cudaMemcpy(d_Frame_in, h_Frame_in, height_Frame * width_Frame * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Start CUDA event for kernel execution timing
        cudaEventRecord(start);

        // Configure grid and block dimensions
        dim3 block(tile_width, tile_height);
        dim3 grid((width_Frame + tile_width - 1) / tile_width, 
                  (height_Frame + tile_height - 1) / tile_height);

        // Erosion
        kernel_erosion<<<grid, block>>>(d_Frame_in, d_Frame_tmp, height_Frame, width_Frame, d_Mask, mask_radius);

        // Dilation
        kernel_dilation<<<grid, block>>>(d_Frame_tmp, d_Frame_out, height_Frame, width_Frame, d_Mask, mask_radius);

        // Stop CUDA event to measure kernel execution time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t, start, stop);

        // Copy the processed data back to host
        cudaMemcpy(h_Frame_out, d_Frame_out, height_Frame * width_Frame * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        // Save processed image
        SavePGM_ui8matrix(h_Frame_out, height_Frame, width_Frame, output_filename);

        // Free allocated memory for this frame
        free(h_Frame_in);
        free(h_Frame_out);

        cudaFree(d_Frame_in);
        cudaFree(d_Frame_tmp);
        cudaFree(d_Frame_out);

        // Stop clock for fps estimation
        fps_end = clock();
        fps = (double)CLOCKS_PER_SEC / (fps_end - fps_start);

        printf("Frame     : %04d\n", frame_num);
        printf("Time      : %f ms\n", t);
        printf("Pixel perf: %5.3f ns/pix\n", (t * 10e6) / (width_Frame * height_Frame));
        printf("FLOPs     : %5.3f GFlops\n", (KERNEL_THD_FLOPS * width_Frame * height_Frame * 10e3) / (t * 10e9));
        printf("Bandwidth : %5.3f Go/s\n", (KERNEL_THD_BYTES * width_Frame * height_Frame * 10e3) / (t * 10e9));
        printf("FPS       : %5.2f\n\n", fps);
    }

    // Free mask memory
    cudaFree(d_Mask);
    return 0;
}


int main(int argc, char** argv)
{
  test_morphology(argc, argv);
  return 0;
}



