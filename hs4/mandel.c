
#include "easypap.h"

#include <omp.h>


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
  float cr = leftX + xstep * j;
  float ci = topY - ystep * i;
  float zr = 0.0, zi = 0.0;

  int iter;

  // Pour chaque pixel, on calcule les termes d'une suite, et on
  // s'arrête lorsque |Z| > 2 ou lorsqu'on atteint MAX_ITERATIONS
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

  return iteration_to_color (iter);
}

