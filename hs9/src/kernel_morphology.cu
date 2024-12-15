/**
##############################################################
##############################################################
##############################################################

AUTHORS: MENGQIAN XU (21306077), EDOARDO CARRA' (21400562)
BOARD ID: Q

##############################################################
##############################################################
##############################################################
*/

#include "kernel_morphology.h"
#include <stdio.h>


/**
 * ##############################################
 * #                                            #
 * #                CPU VERSION                 #
 * #                                            #
 * ##############################################
 */

//Computes the minimum value in the neighborhood.
void kernel_erosion(const unsigned char* Frame_in, unsigned char* Frame_out, int height, int width, int* Mask, int mask_radius)
{
    int x = 0;
    int y = 0;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            unsigned char min_val = 255;
            for (int dy = -mask_radius; dy <= mask_radius; dy++) {
                for (int dx = -mask_radius; dx <= mask_radius; dx++) {
                    int ny = y + dy;
                    int nx = x + dx;
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width && Mask[(dy + mask_radius) * (2 * mask_radius + 1) + (dx + mask_radius)]) {
                        min_val = min(min_val, Frame_in[ny * width + nx]);
                    }
                }
            }
            Frame_out[y * width + x] = min_val;

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
            unsigned char max_val = 0;
            for (int dy = -mask_radius; dy <= mask_radius; dy++) {
                for (int dx = -mask_radius; dx <= mask_radius; dx++) {
                    int ny = y + dy;
                    int nx = x + dx;
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width && Mask[(dy + mask_radius) * (2 * mask_radius + 1) + (dx + mask_radius)]) {
                        max_val = max(max_val, Frame_in[ny * width + nx]);
                    }
                }
            }
            Frame_out[y * width + x] = max_val;

        }
    }
}


/**
 * ##############################################
 * #                                            #
 * #                GPU VERSION                 #
 * #                                            #
 * ##############################################
 */

// handle the boundary and use shared memory

 /**
  * The tile size is increased by 2 * mask_radius in both dimensions in order to load the necessary pixels from the global memory. 
  */
__global__
void gpu_kernel_erosion(const unsigned char* Frame_in, unsigned char* Frame_out, int height, int width, int* Mask, int mask_radius)
{
    // load from shared memory (mask_radius x 2 + blockDim.y) rows, each of them with  blockDim.x + 2 * mask_radius elements
    // borders will be handled in a second step
    extern __shared__ unsigned char smem[];
    
    // x and y coordinates of the block in the image
    int x_start_block = blockIdx.x * blockDim.x;
    int y_start_block = blockIdx.y * blockDim.y;

    // x and y coordinates of the pixel in the image
    int x = x_start_block + threadIdx.x; 
    int y = y_start_block + threadIdx.y; 

    // shared memory tile size
    int tile_width = blockDim.x + 2 * mask_radius; // it also contains the mask_radius pixels
    int tile_height = blockDim.y + 2 * mask_radius;

    /**
     * ##############################
     * #       LOAD INTO SMEM       #
     * ##############################
     */

    // Skip if out of bounds. 
    /** When the block is at the border of the image, we pad the shared memory with 255.
     *  In this way we avoid the need to check for out of bounds in the computation phase,
     *  reducing the divergence of the threads.
     */
    for(int y_smem = threadIdx.y; y_smem < tile_height ; y_smem+=blockDim.y)
    {
        int y_load = y_start_block + y_smem - mask_radius; // global row index (we start from the y_start_block - mask_radius)
        int x_load = x_start_block + threadIdx.x - mask_radius; // global col index (we start from the x_start_block - mask_radius)
        int x_smem = threadIdx.x; // shared memory col index

        // first we load the first block size pixels into the shared memory
        // check that the pixel is within the image bounds otherwise we pad with 255 into the shared  
        smem[y_smem*tile_width + x_smem] = (x_load >= 0 && y_load >= 0 && x_load < width && y_load < height)? Frame_in[y_load*width + x_load] : 255; // coalesced access
        
        x_load += blockDim.x; 
        x_smem += blockDim.x;

        // load the remaining 2 * mask_radius pixels
        if( x_smem < (blockDim.x + 2 * mask_radius))
        {
            smem[y_smem*tile_width + x_smem] = (x_load >= 0 && y_load >= 0 && x_load < width && y_load < height)? Frame_in[y_load*width + x_load] : 255;
        }
    }
    __syncthreads(); 

    /**
     * ##############################
     * #       EROSION COMPUTE      #
     * ##############################
     */

    unsigned char min_val = 255;

# pragma unroll
    for(int dy = -mask_radius; dy <= mask_radius; dy++)
    {
        for(int dx = -mask_radius; dx <= mask_radius; dx++)
        {
            // compute the shared memory coordinates
            int x_smes = threadIdx.x + mask_radius + dx;
            int y_smem = threadIdx.y + mask_radius + dy;
            
            if(Mask[(dy + mask_radius) * (2 * mask_radius + 1) + (dx + mask_radius)])
            {
                min_val = min(min_val, smem[y_smem * tile_width + x_smes]); // no bank conflict
            }
        }
    }
    /**
     * ##############################
     * #       WRITE TO GLOBAL      #
     * ##############################
     */

    //__syncthreads(); not needed since each threads does not depend on the others

    // write the result back to the global memory
    Frame_out[y*width + x] = min_val;
}

__global__
void gpu_kernel_dilation(const unsigned char* Frame_in, unsigned char* Frame_out, int height, int width, int* Mask, int mask_radius) {
        // load from shared memory (mask_radius x 2 + blockDim.y) rows, each of them with  blockDim.x + 2 * mask_radius elements
    // borders will be handled in a second step
    extern __shared__ unsigned char smem[];
    
    // x and y coordinates of the block in the image
    int x_start_block = blockIdx.x * blockDim.x;
    int y_start_block = blockIdx.y * blockDim.y;

    // x and y coordinates of the pixel in the image
    int x = x_start_block + threadIdx.x; 
    int y = y_start_block + threadIdx.y; 

    // shared memory tile size
    int tile_width = blockDim.x + 2 * mask_radius; // it also contains the mask_radius pixels
    int tile_height = blockDim.y + 2 * mask_radius;

    /**
     * ##############################
     * #       LOAD INTO SMEM       #
     * ##############################
     */

    // Skip if out of bounds. 
    /** When the block is at the border of the image, we pad the shared memory with 255.
     *  In this way we avoid the need to check for out of bounds in the computation phase,
     *  reducing the divergence of the threads.
     */
    for(int y_smem = threadIdx.y; y_smem < tile_height ; y_smem+=blockDim.y)
    {
        int y_load = y_start_block + y_smem - mask_radius; // global row index (we start from the y_start_block - mask_radius)
        int x_load = x_start_block + threadIdx.x - mask_radius; // global col index (we start from the x_start_block - mask_radius)
        int x_smem = threadIdx.x; // shared memory col index

        // first we load the first block size pixels into the shared memory
        // check that the pixel is within the image bounds otherwise we pad with 255 into the shared  
        smem[y_smem*tile_width + x_smem] =  (x_load >= 0 && y_load >= 0 && x_load < width && y_load < height)? Frame_in[y_load*width + x_load] : 0; // coalesced access
        
        x_load += blockDim.x; 
        x_smem += blockDim.x;

        // load the remaining 2 * mask_radius pixels
        if( x_smem < (blockDim.x + 2 * mask_radius))
        {
            smem[y_smem*tile_width + x_smem] = (x_load >= 0 && y_load >= 0 && x_load < width && y_load < height)? Frame_in[y_load*width + x_load] : 0;
        }
    }
    __syncthreads(); 

    /**
     * ##############################
     * #       EROSION COMPUTE      #
     * ##############################
     */

    unsigned char max_val = 0;

    #pragma unroll
    for(int dy = -mask_radius; dy <= mask_radius; dy++)
    {
        for(int dx = -mask_radius; dx <= mask_radius; dx++)
        {
            // compute the shared memory coordinates
            int x_smes = threadIdx.x + mask_radius + dx;
            int y_smem = threadIdx.y + mask_radius + dy;
            
            if(Mask[(dy + mask_radius) * (2 * mask_radius + 1) + (dx + mask_radius)])
            {
                max_val = max(max_val, smem[y_smem * tile_width + x_smes]); // no bank conflict
            }
        }
    }
    /**
     * ##############################
     * #       WRITE TO GLOBAL      #
     * ##############################
     */

    //__syncthreads(); not needed since each threads does not depend on the others

    // write the result back to the global memory
    Frame_out[y*width + x] = max_val;
};


// no handling of the boundary and no shared memory
__global__
void gpu_kernel_erosion_no_boundary(const unsigned char* Frame_in, unsigned char* Frame_out, int height, int width, int* Mask, int mask_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // check if the thread is out of bounds 
    if (x >= width || y >= height) return;

    // Directly copy the input pixel value to the output
    if (x < mask_radius || x >= width - mask_radius || y < mask_radius || y >= height - mask_radius) {
        Frame_out[y * width + x] = Frame_in[y * width + x];
        return;
    }

    int min_val = 255;
    for (int dy = -mask_radius; dy <= mask_radius; dy++) {
        //If the value of Mask is 1, the corresponding neighboring pixels participate in the calculation
        for (int dx = -mask_radius; dx <= mask_radius; dx++) {
            if (Mask[(dy + mask_radius) * (2 * mask_radius + 1) + (dx + mask_radius)]) {
                int neighbor_val = Frame_in[(y + dy) * width + (x + dx)];
                min_val = min(min_val, neighbor_val);
            }
        }
    }

    // copy the value to the output
    Frame_out[y * width + x] = (unsigned char)min_val;
}

__global__
void gpu_kernel_dilation_no_boundary(const unsigned char* Frame_in, unsigned char* Frame_out, int height, int width, int* Mask, int mask_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 

    // check if the thread is out of bounds 
    if (x >= width || y >= height) return;

    // Directly copy the input pixel value to the output
    if (x < mask_radius || x >= width - mask_radius || y < mask_radius || y >= height - mask_radius) {
        Frame_out[y * width + x] = Frame_in[y * width + x];
        return;
    }

    int max_val = 0;
    for (int dy = -mask_radius; dy <= mask_radius; dy++) {
        for (int dx = -mask_radius; dx <= mask_radius; dx++) {
            if (Mask[(dy + mask_radius) * (2 * mask_radius + 1) + (dx + mask_radius)]) {
                int neighbor_val = Frame_in[(y + dy) * width + (x + dx)];
                max_val = max(max_val, neighbor_val);
            }
        }
    }

    // copy the value to the output
    Frame_out[y * width + x] = (unsigned char)max_val;
}
