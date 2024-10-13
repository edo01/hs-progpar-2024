//###########################################################################
// ######################## 2.1 Work to do #4 ###############################
//###########################################################################
/** 
 * We run the kernel for 10 iteration, using different configurations 
 * for the tiling,the scheduling and the number of threads. We used a 
 * script to run the program with all the possible combinations of the
 * parameters, skipping the invalid ones(number of tiles less than the 
 * number of threads etc...). We then collected the results and we 
 * got the best configuration under 3000ms.
 * 
 * cpu,th,tw,threads,schedule,schedule_n,time
 * full_system,64,64,6,static,1,2840.531
 * full_system,64,128,6,static,1,2996.948
 * full_system,64,256,6,static,1,2566.211 # BEST CONFIGURATION
 * full_system,128,64,6,static,1,2957.591
 * full_system,256,64,6,static,1,2657.949
 * full_system,256,256,6,static,1,2956.679
 * full_system,1024,64,6,static,1,2912.659
 * full_system,128,128,6,static,3,2965.859
 * full_system,64,64,6,dynamic,1,2820.350
 * full_system,512,64,6,dynamic,1,2882.839
 * full_system,1024,64,6,dynamic,1,2760.596
 * full_system,128,256,6,dynamic,3,2980.233
 * 
 * The best configuration is the one with:
 * cpu: full_system
 * th: 64
 * tw: 256
 * threads: 6
 * schedule: static,1
 * time: 2566.211 ms
 * 
 * 
 * We can observe that we can find both dynamic and static 
 * scheduling under 3000ms. Differently from the blur2 kernel,
 * Mandel kernel may be unbalanced due to some pixels that
 * require more iterations to converge. 
 * 
 * 
 */

//###########################################################################
// ######################## 2.1 Work to do #6 ###############################
//###########################################################################
// Memory Analysis for 1024.png
/* 
 * We are dealing with a 1024x1024 image, which means that the image has a size of 4MB.
 * From our previous analysis, we know that when dealing with a workload of that size,
 * we are out of the L1 cache and L2 cache, and we are using the RAM. Moreover, the 
 * only access to memory is when we store the result of the computation. This means
 * that we can directly look at the benchmark results of the write bandwidth of the
 * memory for 4MB of data.
 */


// Operational Intensity (OI) Analysis
/**
 * Each pixel requires at least 4 + 8 FLOPs  and then, depending on the number of iterations,
 * it performs 8 FLOPS per iteration. In the worst case, each pixel will perform 4 + 8*MAX_ITERATIONS
 * FLOPs.
 * At the end of the computation, we store the result of the computation, which requires only 1 store.
 *  
 */

// The ARITHMETIC INTENSITY is ops/mem_acc = (4 + 8*AVG_ITERATIONS_PER_PIXEL)*1024*1024 / 1024*1024 = 4 + 8*AVG_ITERATIONS_PER_PIXEL

// The OPERATIONAL INTENSITY is AI/size_of_data = (4 + 8*AVG_ITERATIONS_PER_PIXEL)*1024*1024 / 4 = 2 + 4*AVG_ITERATIONS_PER_PIXEL
//(We are using floats, so the size of the data is 4 bytes.)

// Roofline Model Analysis
/**
 * In order to define the Roofline Model, we need to know the peak performance of the CPU.
 * For the Cortex-A57 we have: 
 *  - freq: 2.04 GHz
 *  - nops: 3 instructions per cycle x 4 simd width when using integers = 12 instructions per cycle
 *  - number of cores: 4
 * The peak performance is 2.04 GHz x 12 instructions per cycle x 4 cores = 97.92 Gops/s
 * 
 * The memory bandwidth, for the write operation when accessing 4MB, according to the previous analysis,
 * is 39.2 GB/s.
 * 
 * From the Roofline model, we can compute the minimum between the peak performance and the memory 
 * bandwidth*Operational Intensity, which is equal to 39.2 GB/s x (2 + 4*AVG_ITERATIONS_PER_PIXEL) = 78.4 Gops/s + 156.8*AVG_ITERATIONS_PER_PIXEL Gops/s.
 * 
 * So, since product bandwidth*Operational Intensity is always greater than the peak performance, for any value of 
 * AVG_ITERATIONS_PER_PIXEL (at least 1), the kernel is compute bounds when using all the Cortex's cores. The maximum performance
 * achievable will be of 97.92 Gops/s.
 * 
 * 
 * For the Denver2, we have:
 *  - freq: 2.04 GHz
 *  - nops: 2 instructions per cycle x 4 simd width when using integers = 8 instructions per cycle
 *  - number of cores: 2
 * The peak performance is 2.04 GHz x 8 instructions per cycle x 2 cores = 32.64 Gops/s
 * 
 * The memory bandwidth for the Denver, according to the previous analysis, is 21.7 GB/s
 * when using all the cores and accessing the RAM. 
 * From the Roofline model, we can compute the minimum between the peak performance and the memory 
 * bandwidth*Operational Intensity, which is equal to 21.7 GB/s x (2 + 4*AVG_ITERATIONS_PER_PIXEL) = 43.4 Gops/s + 86.8*AVG_ITERATIONS_PER_PIXEL Gops/s.
 * 
 * So, since product bandwidth*Operational Intensity is always greater than the peak performance, for any value of
 * AVG_ITERATIONS_PER_PIXEL (at least 1), the kernel is compute bounds when using all the Denver's cores. The maximum
 * performance achievable will be of 32.64 Gops/s.
 */

#include "easypap.h"

#include <omp.h>
#include <mipp.h>
#include <arm_neon.h>

#define MAX_ITERATIONS 4096
#define ZOOM_SPEED -0.01

static float leftX   = -0.2395;
static float rightX  = -0.2275;
static float topY    = .660;
static float bottomY = .648;

static float xstep;
static float ystep;

static unsigned compute_one_pixel (int i, int j);
static void zoom (void);

void mandel_init ()
{
  xstep = (rightX - leftX) / DIM;
  ystep = (topY - bottomY) / DIM;
}

int mandel_do_tile_default (int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      cur_img (i, j) = compute_one_pixel (i, j);

  return 0;
}

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline:
// ./run --kernel mandel
//
unsigned mandel_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    do_tile (0, 0, DIM, DIM);

    zoom ();
  }

  return 0;
}


///////////////////////////// Tiled sequential version (tiled)
// Suggested cmdline:
// ./run -k mandel -v tiled -ts 64
//
unsigned mandel_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H);

    zoom ();
  }

  return 0;
}

unsigned mandel_compute_omp_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    #pragma omp parallel for collapse(2) schedule(runtime)
    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H);

    zoom ();
  }

  return 0;
}

// Memory Analysis for 1024x1024 Mandelbrot
// Most loads and stores occur when accessing real and imaginary values during the iterative Mandelbrot computation.
// Estimated memory accesses per pixel: 11 (10 loads + 1 store)
// Estimated FLOPs per pixel: 1000 FLOPs (assuming 100 iterations with 10 FLOPs/iteration)
// Operational Intensity (OI): 22.73 FLOPs/byte
// According to the Roofline model, the Mandelbrot computation is compute-bound as it has a high OI and performs a significant number of FLOPs per memory access.

/*
 * Estimate Memory Accesses per Pixel: 
 * Loads: For each pixel, we need to: 
 * Load the initial complex number (2 floating-point values: real and imaginary parts).
 * Load temporary values during the iterative computation (e.g., z_r, z_i)
 * Stores: After calculating the iteration count for a pixel, we store the result (1 store per pixel).
 *
 * Calculate Operational Intensity (OI):
 * FLOPs: The number of floating-point operations per pixel.In the Mandelbrot algorithm, each pixel involves complex number arithmetic
 * Each iteration involves:
 * 1. Two multiplications z_r^2, z_i^2.
 * 2. Two additions z_r^2 - z_i^2 + c_r , and 2z_rz_i + c_i.
 * we assume 100 iterations with 10 FLOPs per iteration, so the result; OI = 1000 FLOPs / 11 * 4 bytes = 22.73 FLOS/byte.
 *
 * Memory-bound: If the OI is low and the system is limited by memory bandwidth, the code is memory-bound.
 * Compute-bound: If the OI is high and the code is limited by CPU performance (FLOPs), the code is compute-bound.
 * Here, we have an OI of 22.73 FLOPs/byte, the Mandelbrot computation is likely compute-bound.
 */


/*
 * MIPP version 
 * Write the comput_one_pixel function into the mandel_compute_simd_tiled
 * Early Exit:
 * Use mipp::testz(mask) to reduce the mask register to determine whether all pixels satisfy the condition |Z| > 2. 
 * If so, exit the loop early.
*/
void mandel_compute_simd_tiled(unsigned nb_iter)
{
  const int vector_size = mipp::N<float>();  // Get the size of the SIMD vector

  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int y = 0; y < DIM; y += TILE_H) {
      for (int x = 0; x < DIM; x += TILE_W) {
        
        // Traverse the pixels in the tile and use SIMD to process multiple pixels at the same time
        for (int i = y; i < y + TILE_H; i++) {
          for (int j = x; j < x + TILE_W; j += vector_size) {

            // init C（cr, ci）
            mipp::Reg<float> cr = mipp::Reg<float>(leftX + xstep * j) + mipp::Reg<float>::iota(0, xstep);
            mipp::Reg<float> ci = mipp::Reg<float>(topY - ystep * i);

            // init Z = 0（zr, zi）
            mipp::Reg<float> zr(0.0f), zi(0.0f);
            mipp::Reg<int> iter_count(0);

            mipp::Reg<float> zr2 = zr * zr;
            mipp::Reg<float> zi2 = zi * zi;

            for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
              // Determine whether to continue iterating
              auto mask = (zr2 + zi2) < 4.0f;

              // If all elements no longer need to be iterated, exit the loop early
              if (mipp::testz(mask)) {
                break;
              }

              // Increase the iteration count of pixels that meet the criteria
              iter_count = mipp::blend(iter_count + 1, iter_count, mask);

             // Update the value of Z
              mipp::Reg<float> tmp_zi = 2.0f * zr * zi + ci;
              zr = zr2 - zi2 + cr;
              zi = tmp_zi;

              zr2 = zr * zr;
              zi2 = zi * zi;
            }

            // store
            for (int v = 0; v < vector_size; v++) {
              cur_img(i, j + v) = iteration_to_color(iter_count[v]);
            }
          }
        }
      }
    }

    zoom();  
  }
}

/////////////// Mandelbrot basic computation

static unsigned iteration_to_color (unsigned iter)
{
  uint8_t r = 0, g = 0, b = 0;

  if (iter < MAX_ITERATIONS) {
    if (iter < 64) {
      r = iter * 2; /* 0x0000 to 0x007E */
    } else if (iter < 128) {
      r = (((iter - 64) * 128) / 126) + 128; /* 0x0080 to 0x00C0 */
    } else if (iter < 256) {
      r = (((iter - 128) * 62) / 127) + 193; /* 0x00C1 to 0x00FF */
    } else if (iter < 512) {
      r = 255;
      g = (((iter - 256) * 62) / 255) + 1; /* 0x01FF to 0x3FFF */
    } else if (iter < 1024) {
      r = 255;
      g = (((iter - 512) * 63) / 511) + 64; /* 0x40FF to 0x7FFF */
    } else if (iter < 2048) {
      r = 255;
      g = (((iter - 1024) * 63) / 1023) + 128; /* 0x80FF to 0xBFFF */
    } else {
      r = 255;
      g = (((iter - 2048) * 63) / 2047) + 192; /* 0xC0FF to 0xFFFF */
    }
  }
  return ezv_rgb (r, g, b);
}

static void zoom (void)
{
  float xrange = (rightX - leftX);
  float yrange = (topY - bottomY);

  leftX += ZOOM_SPEED * xrange;
  rightX -= ZOOM_SPEED * xrange;
  topY -= ZOOM_SPEED * yrange;
  bottomY += ZOOM_SPEED * yrange;

  xstep = (rightX - leftX) / DIM;
  ystep = (topY - bottomY) / DIM;
}

static unsigned compute_one_pixel (int i, int j)
{
  float cr = leftX + xstep * j; // 2 floating-point operations
  float ci = topY - ystep * i; // 2 floating-point operations
  float zr = 0.0, zi = 0.0;

  int iter;

  // Pour chaque pixel, on calcule les termes d'une suite, et on
  // s'arrête lorsque |Z| > 2 ou lorsqu'on atteint MAX_ITERATIONS
  
  // 8 FLOPS per iteration
  for (iter = 0; iter < MAX_ITERATIONS; iter++) {
    float x2 = zr * zr; 
    float y2 = zi * zi; 

    /* Stop iterations when |Z| > 2 */
    if (x2 + y2 > 4.0)
      break;

    float twoxy = (float)2.0 * zr * zi;
    /* Z = Z^2 + C */
    zr = x2 - y2 + cr; 
    zi = twoxy + ci; 
  }

  return iteration_to_color (iter); //
}

