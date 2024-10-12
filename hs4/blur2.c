/**
##############################################################
##############################################################
##############################################################

AUTHORS: MENGQIAN XU (21306077), EDOARDO CARRA' (21400562)
BOARD ID: Q

Note: For each optimization of the kernel, we run the program 
multiple times to get the best (lowest) execution time. In this
way, we can obtain a more accurate result, since the execution
time can vary depending on the system load. This process was
carried out for both Denver2 and Cortex-A57.

##############################################################
##############################################################
##############################################################
*/

#include "easypap.h"

#include <omp.h>

///////////////////////////// Sequential version (tiled)
// Suggested cmdline(s):
// ./run -l images/1024.png -k blur2 -v seq -si
//
int blur2_do_tile_default (int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++) {
      unsigned r = 0, g = 0, b = 0, a = 0, n = 0;

      int i_d = (i > 0) ? i - 1 : i;
      int i_f = (i < DIM - 1) ? i + 1 : i;
      int j_d = (j > 0) ? j - 1 : j;
      int j_f = (j < DIM - 1) ? j + 1 : j;

      for (int yloc = i_d; yloc <= i_f; yloc++)
        for (int xloc = j_d; xloc <= j_f; xloc++) {
            unsigned c = cur_img (yloc, xloc);
            r += ezv_c2r (c);
            g += ezv_c2g (c);
            b += ezv_c2b (c);
            a += ezv_c2a (c);
            n += 1;
        }

      r /= n;
      g /= n;
      b /= n;
      a /= n;

      next_img (i, j) = ezv_rgba (r, g, b, a);
    }
    
  return 0;
}

///////////////////////////// Sequential version (tiled)
// Suggested cmdline(s):
// ./run -l images/1024.png -k blur2 -v seq
//
unsigned blur2_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    do_tile (0, 0, DIM, DIM, 0);

    swap_images ();
  }

  return 0;
}

///////////////////////////// Tiled sequential version (tiled)
// Suggested cmdline(s):
// ./run -l images/1024.png -k blur2 -v tiled -ts 32 -m si
//
unsigned blur2_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H, 0);

    swap_images ();
  }

  return 0;
}

unsigned blur2_compute_omp_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    /* 
     * adding collapse allows to split the image both in 
     * vertical tiles and horizontal tiles. 
    */
    #pragma omp parallel for collapse(2) schedule(runtime)
    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile(x, y, TILE_W, TILE_H, omp_get_thread_num());

    swap_images ();
  }

  return 0;
}

/**
 * #########################################################################
 * #########################################################################
 */

void compute_borders(int x, int y, int width, int height, int bsize) 
{
  assert(bsize > 0);
  // left -------------------------------------------------------------------------------------------------------------
  if (x == 0) {
    for (int i = y + 1; i < y + height - 1; i++) {
      uint16_t r = 0, g = 0, b = 0, a = 0;
      unsigned c_0_1 = cur_img (i - 1, x + 0), c_1_1 = cur_img (i + 0, x + 0), c_2_1 = cur_img (i + 1, x + 0);
      unsigned c_0_2 = cur_img (i - 1, x + 1), c_1_2 = cur_img (i + 0, x + 1), c_2_2 = cur_img (i + 1, x + 1);
      r += ezv_c2r (c_0_1); g += ezv_c2g (c_0_1); b += ezv_c2b (c_0_1); a += ezv_c2a (c_0_1);
      r += ezv_c2r (c_1_1); g += ezv_c2g (c_1_1); b += ezv_c2b (c_1_1); a += ezv_c2a (c_1_1);
      r += ezv_c2r (c_2_1); g += ezv_c2g (c_2_1); b += ezv_c2b (c_2_1); a += ezv_c2a (c_2_1);
      r += ezv_c2r (c_0_2); g += ezv_c2g (c_0_2); b += ezv_c2b (c_0_2); a += ezv_c2a (c_0_2);
      r += ezv_c2r (c_1_2); g += ezv_c2g (c_1_2); b += ezv_c2b (c_1_2); a += ezv_c2a (c_1_2);
      r += ezv_c2r (c_2_2); g += ezv_c2g (c_2_2); b += ezv_c2b (c_2_2); a += ezv_c2a (c_2_2);
      r /= 6; g /= 6; b /= 6; a /= 6;
      next_img (i, 0) = ezv_rgba (r, g, b, a);
    }
  }

  for (int i = y + 1; i < y + height - 1; i++) {
    for (int j = x + (x == 0) ? 1 : 0; j < x + bsize; j++) {
      uint16_t r = 0, g = 0, b = 0, a = 0;
      unsigned c_0_0 = cur_img (i - 1, j - 1), c_1_0 = cur_img (i + 0, j - 1), c_2_0 = cur_img (i + 1, j - 1);
      unsigned c_0_1 = cur_img (i - 1, j + 0), c_1_1 = cur_img (i + 0, j + 0), c_2_1 = cur_img (i + 1, j + 0);
      unsigned c_0_2 = cur_img (i - 1, j + 1), c_1_2 = cur_img (i + 0, j + 1), c_2_2 = cur_img (i + 1, j + 1);
      r += ezv_c2r (c_0_0); g += ezv_c2g (c_0_0); b += ezv_c2b (c_0_0); a += ezv_c2a (c_0_0);
      r += ezv_c2r (c_1_0); g += ezv_c2g (c_1_0); b += ezv_c2b (c_1_0); a += ezv_c2a (c_1_0);
      r += ezv_c2r (c_2_0); g += ezv_c2g (c_2_0); b += ezv_c2b (c_2_0); a += ezv_c2a (c_2_0);
      r += ezv_c2r (c_0_1); g += ezv_c2g (c_0_1); b += ezv_c2b (c_0_1); a += ezv_c2a (c_0_1);
      r += ezv_c2r (c_1_1); g += ezv_c2g (c_1_1); b += ezv_c2b (c_1_1); a += ezv_c2a (c_1_1);
      r += ezv_c2r (c_2_1); g += ezv_c2g (c_2_1); b += ezv_c2b (c_2_1); a += ezv_c2a (c_2_1);
      r += ezv_c2r (c_0_2); g += ezv_c2g (c_0_2); b += ezv_c2b (c_0_2); a += ezv_c2a (c_0_2);
      r += ezv_c2r (c_1_2); g += ezv_c2g (c_1_2); b += ezv_c2b (c_1_2); a += ezv_c2a (c_1_2);
      r += ezv_c2r (c_2_2); g += ezv_c2g (c_2_2); b += ezv_c2b (c_2_2); a += ezv_c2a (c_2_2);
      r /= 9; g /= 9; b /= 9; a /= 9;
      next_img (i, j) = ezv_rgba (r, g, b, a);
    }
  }

  // right ------------------------------------------------------------------------------------------------------------
  if (x + width == DIM) {
    for (int i = y + 1; i < y + height - 1; i++) {
      uint16_t r = 0, g = 0, b = 0, a = 0;
      unsigned c_0_0 = cur_img (i - 1, x + width -2), c_1_0 = cur_img (i + 0, x + width -2), c_2_0 = cur_img (i + 1, x + width -2);
      unsigned c_0_1 = cur_img (i - 1, x + width -1), c_1_1 = cur_img (i + 0, x + width -1), c_2_1 = cur_img (i + 1, x + width -1);
      r += ezv_c2r (c_0_1); g += ezv_c2g (c_0_1); b += ezv_c2b (c_0_1); a += ezv_c2a (c_0_1);
      r += ezv_c2r (c_1_1); g += ezv_c2g (c_1_1); b += ezv_c2b (c_1_1); a += ezv_c2a (c_1_1);
      r += ezv_c2r (c_2_1); g += ezv_c2g (c_2_1); b += ezv_c2b (c_2_1); a += ezv_c2a (c_2_1);
      r += ezv_c2r (c_0_0); g += ezv_c2g (c_0_0); b += ezv_c2b (c_0_0); a += ezv_c2a (c_0_0);
      r += ezv_c2r (c_1_0); g += ezv_c2g (c_1_0); b += ezv_c2b (c_1_0); a += ezv_c2a (c_1_0);
      r += ezv_c2r (c_2_0); g += ezv_c2g (c_2_0); b += ezv_c2b (c_2_0); a += ezv_c2a (c_2_0);
      r /= 6; g /= 6; b /= 6; a /= 6;
      next_img (i, x + width - 1) = ezv_rgba (r, g, b, a);
    }
  }

  for (int i = y + 1; i < y + height - 1; i++) {
    for (int j = x + width - bsize; j < x + width - (x + width == DIM) ? 1 : 0; j++) {
      uint16_t r = 0, g = 0, b = 0, a = 0;
      unsigned c_0_0 = cur_img (i - 1, j - 1), c_1_0 = cur_img (i + 0, j - 1), c_2_0 = cur_img (i + 1, j - 1);
      unsigned c_0_1 = cur_img (i - 1, j + 0), c_1_1 = cur_img (i + 0, j + 0), c_2_1 = cur_img (i + 1, j + 0);
      unsigned c_0_2 = cur_img (i - 1, j + 1), c_1_2 = cur_img (i + 0, j + 1), c_2_2 = cur_img (i + 1, j + 1);
      r += ezv_c2r (c_0_0); g += ezv_c2g (c_0_0); b += ezv_c2b (c_0_0); a += ezv_c2a (c_0_0);
      r += ezv_c2r (c_1_0); g += ezv_c2g (c_1_0); b += ezv_c2b (c_1_0); a += ezv_c2a (c_1_0);
      r += ezv_c2r (c_2_0); g += ezv_c2g (c_2_0); b += ezv_c2b (c_2_0); a += ezv_c2a (c_2_0);
      r += ezv_c2r (c_0_1); g += ezv_c2g (c_0_1); b += ezv_c2b (c_0_1); a += ezv_c2a (c_0_1);
      r += ezv_c2r (c_1_1); g += ezv_c2g (c_1_1); b += ezv_c2b (c_1_1); a += ezv_c2a (c_1_1);
      r += ezv_c2r (c_2_1); g += ezv_c2g (c_2_1); b += ezv_c2b (c_2_1); a += ezv_c2a (c_2_1);
      r += ezv_c2r (c_0_2); g += ezv_c2g (c_0_2); b += ezv_c2b (c_0_2); a += ezv_c2a (c_0_2);
      r += ezv_c2r (c_1_2); g += ezv_c2g (c_1_2); b += ezv_c2b (c_1_2); a += ezv_c2a (c_1_2);
      r += ezv_c2r (c_2_2); g += ezv_c2g (c_2_2); b += ezv_c2b (c_2_2); a += ezv_c2a (c_2_2);
      r /= 9; g /= 9; b /= 9; a /= 9;
      next_img (i, j) = ezv_rgba (r, g, b, a);
    }
  }

  // top & bottom -----------------------------------------------------------------------------------------------------
  for (int i = y; i < y + height; i += height - 1) {
    for (int j = x; j < x + width; j++) {
      uint16_t r = 0, g = 0, b = 0, a = 0, n = 0;
      int i_d = (i > 0) ? i - 1 : i;
      int i_f = (i < DIM - 1) ? i + 1 : i;
      int j_d = (j > 0) ? j - 1 : j;
      int j_f = (j < DIM - 1) ? j + 1 : j;
      for (int yloc = i_d; yloc <= i_f; yloc++)
        for (int xloc = j_d; xloc <= j_f; xloc++) {
          unsigned c = cur_img (yloc, xloc);
          r += ezv_c2r (c); g += ezv_c2g (c); b += ezv_c2b (c); a += ezv_c2a (c);
          n += 1;
        }
      r /= n; g /= n; b /= n; a /= n;
      next_img (i, j) = ezv_rgba (r, g, b, a);
    }
  }
}

int blur2_do_tile_urrot1 (int x, int y, int width, int height)
{
  // loop over y (start from +1, end at -1 => no border)
  for (int i = y + 1; i < y + height - 1; i++) {
    uint16_t c_0_r = 0, c_0_g = 0, c_0_b = 0, c_0_a = 0; // col 0 -> 4 color components {r,g,b,a}
    uint16_t c_1_r = 0, c_1_g = 0, c_1_b = 0, c_1_a = 0; // col 1 -> 4 color components {r,g,b,a}

    // read 3 pixels of column 0
    unsigned c_0_0 = cur_img(i - 1, x + 0), c_1_0 = cur_img(i + 0, x + 0), c_2_0 = cur_img(i + 1, x + 0);
    // read 3 pixels of column 1
    unsigned c_0_1 = cur_img(i - 1, x + 1), c_1_1 = cur_img(i + 0, x + 1), c_2_1 = cur_img(i + 1, x + 1);

    // reduction of the pixels of column 0 (per components {r,g,b,a})
    c_0_r += ezv_c2r(c_0_0); c_0_g += ezv_c2g(c_0_0); c_0_b += ezv_c2b(c_0_0); c_0_a += ezv_c2a(c_0_0);
    c_0_r += ezv_c2r(c_1_0); c_0_g += ezv_c2g(c_1_0); c_0_b += ezv_c2b(c_1_0); c_0_a += ezv_c2a(c_1_0);
    c_0_r += ezv_c2r(c_2_0); c_0_g += ezv_c2g(c_2_0); c_0_b += ezv_c2b(c_2_0); c_0_a += ezv_c2a(c_2_0);

    // reduction of the pixels of column 1 (per components {r,g,b,a})
    c_1_r += ezv_c2r(c_0_1); c_1_g += ezv_c2g(c_0_1); c_1_b += ezv_c2b(c_0_1); c_1_a += ezv_c2a(c_0_1);
    c_1_r += ezv_c2r(c_1_1); c_1_g += ezv_c2g(c_1_1); c_1_b += ezv_c2b(c_1_1); c_1_a += ezv_c2a(c_1_1);
    c_1_r += ezv_c2r(c_2_1); c_1_g += ezv_c2g(c_2_1); c_1_b += ezv_c2b(c_2_1); c_1_a += ezv_c2a(c_2_1);

    // loop over x (start from +1, end at -1 => no border)
    for (int j = x + 1; j < x + width - 1; j++) {
      uint16_t c_2_r = 0, c_2_g = 0, c_2_b = 0, c_2_a = 0; // col 2 -> 4 color components {r,g,b,a}

      // read 3 pixels of column 2
      unsigned c_0_2 = cur_img(i - 1, j + 1);
      unsigned c_1_2 = cur_img(i + 0, j + 1);
      unsigned c_2_2 = cur_img(i + 1, j + 1);

      // reduction of the pixels of column 2 (per components {r,g,b,a})
      c_2_r += ezv_c2r(c_0_2); c_2_g += ezv_c2g(c_0_2); c_2_b += ezv_c2b(c_0_2); c_2_a += ezv_c2a(c_0_2);
      c_2_r += ezv_c2r(c_1_2); c_2_g += ezv_c2g(c_1_2); c_2_b += ezv_c2b(c_1_2); c_2_a += ezv_c2a(c_1_2);
      c_2_r += ezv_c2r(c_2_2); c_2_g += ezv_c2g(c_2_2); c_2_b += ezv_c2b(c_2_2); c_2_a += ezv_c2a(c_2_2);

      // compute the sum of all the columns 0,1,2 per components {r,g,b,a}
      uint16_t r = 0, g = 0, b = 0, a = 0;
      r = c_0_r+c_1_r+c_2_r; g = c_0_g+c_1_g+c_2_g; b = c_0_b+c_1_b+c_2_b; a = c_0_a+c_1_a+c_2_a;
      // compute the average (sum = sum / 9)
      r /= 9; g /= 9; b /= 9; a /= 9;

      // variables rotations (col0 <- col1 and col1 <- col2)
      c_0_r = c_1_r; c_0_g = c_1_g; c_0_b = c_1_b; c_0_a = c_1_a;
      c_1_r = c_2_r; c_1_g = c_2_g; c_1_b = c_2_b; c_1_a = c_2_a;

      // write the current pixel
      next_img(i, j) = ezv_rgba (r, g, b, a);
    }
  }

  // left-right borders size
  uint32_t bsize = 1;
  // compute the borders
  compute_borders(x, y, width, height, bsize);

  return 0;
}

///////////////////////////// Sequential version (tiled)
// Suggested cmdline(s):
// ./run -l images/1024.png -k blur2 -v seq -si --wt urrot2
//
int blur2_do_tile_urrot2 (int x, int y, int width, int height)
{
  // small arrays (4 elements each) to store all the components of 1 pixel
  uint8_t c_0_0[4], c_1_0[4], c_2_0[4];
  uint8_t c_0_1[4], c_1_1[4], c_2_1[4];
  uint8_t c_0_2[4], c_1_2[4], c_2_2[4];

  // loop over y (start from +1, end at -1 => no border)
  for (int i = y + 1; i < y + height - 1; i++) {
    uint8_t* cur_img_ptr;
    // load the 3 pixels of the column 0 (no extraction of the components)
    cur_img_ptr = (uint8_t*)&cur_img(i - 1, x + 0); for(int c=0;c<4;c++) c_0_0[c] = cur_img_ptr[c];
    cur_img_ptr = (uint8_t*)&cur_img(i + 0, x + 0); for(int c=0;c<4;c++) c_1_0[c] = cur_img_ptr[c];
    cur_img_ptr = (uint8_t*)&cur_img(i + 1, x + 0); for(int c=0;c<4;c++) c_2_0[c] = cur_img_ptr[c];

    // load the 3 pixels of the column 1 (no extraction of the components)
    cur_img_ptr = (uint8_t*)&cur_img(i - 1, x + 1); for(int c=0;c<4;c++) c_0_1[c] = cur_img_ptr[c];
    cur_img_ptr = (uint8_t*)&cur_img(i + 0, x + 1); for(int c=0;c<4;c++) c_1_1[c] = cur_img_ptr[c];
    cur_img_ptr = (uint8_t*)&cur_img(i + 1, x + 1); for(int c=0;c<4;c++) c_2_1[c] = cur_img_ptr[c];

    // column 0 and column 1 reduction
    uint16_t c_0[4], c_1[4];
    for(int c=0;c<4;c++) c_0[c] = c_0_0[c] + c_1_0[c] + c_2_0[c];
    for(int c=0;c<4;c++) c_1[c] = c_0_1[c] + c_1_1[c] + c_2_1[c];

    // loop over x (start from +1, end at -1 => no border)
    for (int j = x + 1; j < x + width - 1; j++) {
      // load the 3 pixels of the column 2 (no extraction of the components)
      cur_img_ptr = (uint8_t*)&cur_img(i - 1, j + 1); for(int c=0;c<4;c++) c_0_2[c] = cur_img_ptr[c];
      cur_img_ptr = (uint8_t*)&cur_img(i + 0, j + 1); for(int c=0;c<4;c++) c_1_2[c] = cur_img_ptr[c];
      cur_img_ptr = (uint8_t*)&cur_img(i + 1, j + 1); for(int c=0;c<4;c++) c_2_2[c] = cur_img_ptr[c];

      // column 2 reduction
      uint16_t c_2[4] = {0, 0, 0, 0};
      for(int c=0;c<4;c++) c_2[c] += c_0_2[c] + c_1_2[c] + c_2_2[c];

      // add column 0, 1 and 2 and compute the avg (div9)
      uint16_t avg[4] = {0, 0, 0, 0};
      for(int c=0;c<4;c++) avg[c] = (c_0[c] + c_1[c] + c_2[c]) / 9;

      // variables rotations
      for(int c=0;c<4;c++) c_0[c] = c_1[c];
      for(int c=0;c<4;c++) c_1[c] = c_2[c];

      // store the resulting pixel (no need for the 'rgba' function)
      uint8_t* next_img_ptr = (uint8_t*)&next_img (i, j);
      for(int c=0;c<4;c++) next_img_ptr[c] = avg[c];
    }
  }

  // left-right borders size
  uint32_t bsize = 1;
  // compute the borders
  compute_borders(x, y, width, height, bsize);

  return 0;
}

#if defined(ENABLE_VECTO) && (defined(__ARM_NEON__) || defined(__ARM_NEON))
#include <arm_neon.h>

void print_reg_u8(const uint8x16_t r, const char* name) {
  printf("%s = [", name);
  printf("%u, ", vgetq_lane_u8(r, 0));
  printf("%u, ", vgetq_lane_u8(r, 1));
  printf("%u, ", vgetq_lane_u8(r, 2));
  printf("%u, ", vgetq_lane_u8(r, 3));
  printf("%u, ", vgetq_lane_u8(r, 4));
  printf("%u, ", vgetq_lane_u8(r, 5));
  printf("%u, ", vgetq_lane_u8(r, 6));
  printf("%u, ", vgetq_lane_u8(r, 7));
  printf("%u, ", vgetq_lane_u8(r, 8));
  printf("%u, ", vgetq_lane_u8(r, 9));
  printf("%u, ", vgetq_lane_u8(r,10));
  printf("%u, ", vgetq_lane_u8(r,11));
  printf("%u, ", vgetq_lane_u8(r,12));
  printf("%u, ", vgetq_lane_u8(r,13));
  printf("%u, ", vgetq_lane_u8(r,14));
  printf("%u", vgetq_lane_u8(r,15));
  printf("]\n");
}

void print_reg_u16(const uint16x8_t r, const char* name) {
  printf("%s = [", name);
  printf("%u, ", vgetq_lane_u16(r, 0));
  printf("%u, ", vgetq_lane_u16(r, 1));
  printf("%u, ", vgetq_lane_u16(r, 2));
  printf("%u, ", vgetq_lane_u16(r, 3));
  printf("%u, ", vgetq_lane_u16(r, 4));
  printf("%u, ", vgetq_lane_u16(r, 5));
  printf("%u, ", vgetq_lane_u16(r, 6));
  printf("%u", vgetq_lane_u16(r, 7));
  printf("]\n");
}

void print_reg_u32(const uint32x4_t r, const char* name) {
  printf("%s = [", name);
  printf("%u, ", vgetq_lane_u32(r, 0));
  printf("%u, ", vgetq_lane_u32(r, 1));
  printf("%u, ", vgetq_lane_u32(r, 2));
  printf("%u", vgetq_lane_u32(r, 3));
  printf("]\n");
}

void print_reg_f32(const float32x4_t r, const char* name) {
  printf("%s = [", name);
  printf("%f, ", vgetq_lane_f32(r, 0));
  printf("%f, ", vgetq_lane_f32(r, 1));
  printf("%f, ", vgetq_lane_f32(r, 2));
  printf("%f", vgetq_lane_f32(r, 3));
  printf("]\n");
}

/**
 * Divide by 9 using a series of shifts and additions. In this 
 * way we can avoid the division operation and the corresponding
 * cost of conversion to float and back to integer.
 */
inline uint16x8_t neon_vdiv9_u16(uint16x8_t n) {
    // q1 = n - (n >> 3)
    uint16x8_t q1 = vsubq_u16(n, vshrq_n_u16(n, 3));
    // q1 += (q1 >> 6)
    q1 = vaddq_u16(q1, vshrq_n_u16(q1, 6));
    // q2 = q1 >> 3
    uint16x8_t q2 = vshrq_n_u16(q1, 3);
    // r = n - (q1 + q2)
    uint16x8_t r = vsubq_u16(n, vaddq_u16(q1, q2));
    // r = q2 + ((r + 7) >> 4)
    r = vaddq_u16(q2, vshrq_n_u16(vaddq_u16(r, vdupq_n_u16(7)), 4));
    return r;
}

/**
 * SCALAR VERSION with boarder management and variable rotation
 * 
 * D2: 755.031  ms
 * CA52: 1146.858  ms
 */

// urrot1_neon_div9_f32
/**
 * In this first version of the vectorized blur2, we will use the
 * NEON intrinsics to perform the computation of the blur. Every iteration
 * of the loop will compute the blur of 16 pixels, using the left-right
 * pattern to compute the 2D convolution. We perform the variable rotation,
 * and also we manage the borders of the image using vectorized operations.
 * Only the first and last row and column are computed using scalar operations.
 * 
 * 1. Prolugue: Load the first two columns of the image and compute the reduction.
 *    The first column is composed of only three pixels, so we need to handle the
 *    higher part of the registers. 
 * 2. In the main loop, we load the third column of the image and compute the reduction.
 *    We load and compute the pixels deinterlived, so we can process 16 pixels at each
 *    iteration.
 * 3. We perform the left-right pattern to compute the 2D convolution.
 * 4. We perform first the promotion to 32 bits, then the conversion to float and 
 *    finally the division by 9.
 * 5. Convert the result back to 16 bits and store it in the output image.
 * 6. Perform the variable rotation.
 * 6. Handle the borders of the image.
 * 
 * D2: 1065.894  ms
 * CA57: 844.604  ms
 * 
 * We observe a performance speedup of 1.36x on the Cortex-A57 and a slowdown
 * of 0.71x on the Denver 2, compared to the scalar version of the first hands-on.
 * The factors that may be influencing the performance are the overhead of the
 * promotion to float and the division by 9.
 * 
 */
int blur2_do_tile_urrot1_neon_div9_f32 (int x, int y, int width, int height) {
  /* #########################################################################
   * #########################################################################
   * Variables notation: ^r(a[0-4])?_c_[0-3]_l_[0-2]_u[8,16,32,64]
   * r -> register
   * a[0-4] -> array of registers and its dimension
   * c_[0-2] -> column number
   * l_[0-2] -> line number
   * u[8,16,32,64] -> type of the registers
   * 
   * Example ra4_c_2_l_0_u8 -> array of four registers u_0, right column, first line 
   * #########################################################################
   * #########################################################################
   */

  /*
   * Contain the deinterlived pixel colors of the right group column. One array
   * for each line.
   */
  uint8x16x4_t ra4_c_0_l_0_u8, ra4_c_0_l_1_u8, ra4_c_0_l_2_u8;
  uint8x16x4_t ra4_c_1_l_0_u8, ra4_c_1_l_1_u8, ra4_c_1_l_2_u8;
  uint8x16x4_t ra4_c_2_l_0_u8, ra4_c_2_l_1_u8, ra4_c_2_l_2_u8;
  uint8x16x4_t ra4_sum_u8;

  /**
   * Two array, one for the lower and one for higher part, for each
   * line and for each column.
   */  
  uint16x8x4_t ra4_c_0_l_0_u16_l; // left column, first line, no need for higher part
  uint16x8x4_t ra4_c_0_l_1_u16_h, ra4_c_0_l_1_u16_l; // left column, second line
  uint16x8x4_t ra4_c_0_l_2_u16_l; // left column, third line, no need for higher part
  
  uint16x8x4_t ra4_c_1_l_0_u16_h, ra4_c_1_l_0_u16_l; // central column, first line
  uint16x8x4_t ra4_c_1_l_1_u16_h, ra4_c_1_l_1_u16_l; // central column, second line
  uint16x8x4_t ra4_c_1_l_2_u16_h, ra4_c_1_l_2_u16_l; // central column, third line
  
  uint16x8x4_t ra4_c_2_l_0_u16_h, ra4_c_2_l_0_u16_l; // right column, first line
  uint16x8x4_t ra4_c_2_l_1_u16_h, ra4_c_2_l_1_u16_l; // right column, second line
  uint16x8x4_t ra4_c_2_l_2_u16_h, ra4_c_2_l_2_u16_l; // right column, third line

  // for storing the sum of the pixels on the higher part of the central column
  // used in variable reduction
  uint16x8x4_t ra4_c_1_l_1_u16_h_temp;

  // To store shiffted values for the reduction
  uint16x8x4_t ra4_left_l, ra4_left_h;
  uint16x8x4_t ra4_right_l, ra4_right_h; 

  // for the promotion to 32 bits
  uint32x4x4_t ra4_sum_l_l, ra4_sum_l_h, ra4_sum_h_l, ra4_sum_h_h;

  // for the division by 9
  float32x4x4_t ra4_sumf_l_l, ra4_sumf_l_h, ra4_sumf_h_l, ra4_sumf_h_h;
  float32_t p = 9;
  float32x4_t R_NINE = vld1q_dup_f32(&p); // broadcast the 

  // loop over y (start from +1, end at -1 => no border)
  for (int i = y + 1; i < y + height - 1; i++) {
    // #############################################################
    //                      PROLOGUE
    // #############################################################
    /*
     * In order to start the variable rotation, we need to precompute
     * the left and central columns. The computation of the central column
     * is done in the same way as the right column in the main loop. 
     * Since the x-loop starts from the second pixel, the left column 
     * is composed by only the first pixel of each line. So, one 
     * possible strategy is to load to perform the exact same computation
     * as the central column, but starting from the first pixel. After that,
     * we can use vextq_u16 to shift the first pixel to the last position.
     * 
     * In this way, we can use simd instructions to perform the computation
     * also for the first 16 pixels of each line instead of using scalar
     * instructions.
     */ 
    ra4_c_0_l_0_u8 = vld4q_u8((uint8_t*)&cur_img(i - 1, x));
    ra4_c_0_l_1_u8 = vld4q_u8((uint8_t*)&cur_img(i + 0, x));
    ra4_c_0_l_2_u8 = vld4q_u8((uint8_t*)&cur_img(i + 1, x));
    
    ra4_c_1_l_0_u8 = vld4q_u8((uint8_t*)&cur_img(i - 1, x + 1));
    ra4_c_1_l_1_u8 = vld4q_u8((uint8_t*)&cur_img(i + 0, x + 1));
    ra4_c_1_l_2_u8 = vld4q_u8((uint8_t*)&cur_img(i + 1, x + 1));

    for(int index=0; index<4; index++){
      /*
       * please note that we need only the lower part of the 16 pixels for the 
       * left column. The higher part is not used in the computation but only
       * to store the first pixel for the computation in the loop.
       */
      // first line
      ra4_c_0_l_0_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_0_l_0_u8.val[index]));
      
      ra4_c_1_l_0_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_1_l_0_u8.val[index]));
      ra4_c_1_l_0_u16_h.val[index] = vmovl_u8(vget_high_u8(ra4_c_1_l_0_u8.val[index]));

      // second line
      ra4_c_0_l_1_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_0_l_1_u8.val[index]));
      
      ra4_c_1_l_1_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_1_l_1_u8.val[index]));
      ra4_c_1_l_1_u16_h.val[index] = vmovl_u8(vget_high_u8(ra4_c_1_l_1_u8.val[index]));

      // third line
      ra4_c_0_l_2_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_0_l_2_u8.val[index]));
      
      ra4_c_1_l_2_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_1_l_2_u8.val[index]));
      ra4_c_1_l_2_u16_h.val[index] = vmovl_u8(vget_high_u8(ra4_c_1_l_2_u8.val[index]));

      // reduction
      ra4_c_0_l_1_u16_l.val[index] = vaddq_u16(ra4_c_0_l_2_u16_l.val[index],
                                              vaddq_u16(ra4_c_0_l_1_u16_l.val[index], ra4_c_0_l_0_u16_l.val[index])); // lower part
      
      ra4_c_1_l_1_u16_l.val[index] = vaddq_u16(ra4_c_1_l_2_u16_l.val[index],
                                              vaddq_u16(ra4_c_1_l_1_u16_l.val[index], ra4_c_1_l_0_u16_l.val[index])); // lower part
      ra4_c_1_l_1_u16_h.val[index] = vaddq_u16(ra4_c_1_l_2_u16_h.val[index],
                                              vaddq_u16(ra4_c_1_l_1_u16_h.val[index], ra4_c_1_l_0_u16_h.val[index])); // higher part

      // move the first bit to the last position using vextq_u16
      ra4_c_0_l_1_u16_h.val[index] = vextq_u16(ra4_c_0_l_1_u16_h.val[index], ra4_c_0_l_1_u16_l.val[index], 1);    
    }
    
    // #############################################################
    //                      END PROLOGUE
    // #############################################################
    
    // loop over x (start from +1, end at -1 => no border)
    for (int j = x + 1; j < x + width; j+=16) {

      // 2.
      /*uint8x16_t r_c_2_l_0 = vld1q_u8((uint8_t*)&cur_img(i - 1, j + 4));
      uint8x16_t r_c_2_l_1 = vld1q_u8((uint8_t*)&cur_img(i + 0, j + 4));
      uint8x16_t r_c_2_l_2 = (uint8x16_t)vld1q_u8((uint8_t*)&cur_img(i + 1, j + 4));*/

      // 3. Memory deinterliving
      /* 
       * Use vld4q_u8 instructions to perform a deinterleving of the four pixel components.
       * For all the three lines of the right column-group, we load from memory 16 pixels 
       * deinterliving the colors. Now each group of pixel is composed by an array of 4 registers 
       * 8b x 16 elements.
       */
      ra4_c_2_l_0_u8 = vld4q_u8((uint8_t*)&cur_img(i - 1, j + 16)); // [[ r x 16 ] [ g x 16 ] [ b x 16 ] [ a x 16 ]]
      ra4_c_2_l_1_u8 = vld4q_u8((uint8_t*)&cur_img(i + 0, j + 16));
      ra4_c_2_l_2_u8 = vld4q_u8((uint8_t*)&cur_img(i + 1, j + 16));
      

      for(int index=0; index<4; index++){
        // 4. 
        /*
         * Promote the 8-bit components into 16-bit components to perform the accumulation
         * First we extract the lower and higher part of the 8-bit components using the 
         * vget_low_u8 and vget_high_u8.
         * Then we promote them to 16-bit components using the vmovl_u8 instruction.
         *  
         */
        // first line
        ra4_c_2_l_0_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_2_l_0_u8.val[index]));
        ra4_c_2_l_0_u16_h.val[index] = vmovl_u8(vget_high_u8(ra4_c_2_l_0_u8.val[index]));

        // second line
        ra4_c_2_l_1_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_2_l_1_u8.val[index]));
        ra4_c_2_l_1_u16_h.val[index] = vmovl_u8(vget_high_u8(ra4_c_2_l_1_u8.val[index]));

        // third line
        ra4_c_2_l_2_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_2_l_2_u8.val[index]));
        ra4_c_2_l_2_u16_h.val[index] = vmovl_u8(vget_high_u8(ra4_c_2_l_2_u8.val[index]));

        // 5. Compute the reduction of the second column
        /*
         * Accumulate the color component of the right column-group. The accumulation is performed
         * using 16 bit registers.
         */
        ra4_c_2_l_1_u16_l.val[index] = vaddq_u16(ra4_c_2_l_2_u16_l.val[index],
                                                vaddq_u16(ra4_c_2_l_1_u16_l.val[index], ra4_c_2_l_0_u16_l.val[index])); // lower part
        ra4_c_2_l_1_u16_h.val[index] = vaddq_u16(ra4_c_2_l_2_u16_h.val[index],
                                                vaddq_u16(ra4_c_2_l_1_u16_h.val[index], ra4_c_2_l_0_u16_h.val[index])); // higher part

        // 6. left-right pattern
        /*
         * Perform the left-right pattern to compute the sum of the pixel components. 
         *
         * [lh8, lh9, lh10, lh11, lh12, lh13, lh14, lh15] -> ra4_c_0_l_1_u16_h
         * [cl0, cl1, cl2, cl3, cl4, cl5, cl6, cl7] -> ra4_c_1_l_1_u16_l
         * [ch8, ch9, ch10, ch11, ch12, ch13, ch14, ch15] -> ra4_c_1_l_1_u16_h
         * [rl0, rl1, rl2, rl3, rl4, rl5, rl6, rl7] -> ra4_c_2_l_1_u16_l
         * 
         * From horizontal to vertical computation:
         * - lower part:
         * [cl1, cl2, cl3, cl4, cl5, cl6, cl7, ch8] -> right lower
         * [cl0, cl1, cl2, cl3, cl4, cl5, cl6, cl7] -> central lower
         * [lh15, cl0, cl1, cl2, cl3, cl4, cl5, cl6] -> left lower
         * 
         * - higher part:
         * [ch9, ch10, ch11, ch12, ch13, ch14, ch15, rl0] -> right higher
         * [ch8, ch9, ch10, ch11, ch12, ch13, ch14, rl0] -> central higher
         * [cl7, ch8, ch9, ch10, ch11, ch12, ch13, ch14] -> left higher
         * 
         * 
         */
        ra4_left_l.val[index]  = vextq_u16(ra4_c_0_l_1_u16_h.val[index], ra4_c_1_l_1_u16_l.val[index], 7);  // [lh15, cl0, cl1, cl2, cl3, cl4, cl5, cl6]
        ra4_right_l.val[index] = vextq_u16(ra4_c_1_l_1_u16_l.val[index], ra4_c_1_l_1_u16_h.val[index], 1); // [cl1, cl2, cl3, cl4, cl5, cl6, cl7, ch8]

        ra4_left_h.val[index]  = vextq_u16(ra4_c_1_l_1_u16_l.val[index], ra4_c_1_l_1_u16_h.val[index], 7); //[cl7, ch8, ch9, ch10, ch11, ch12, ch13, ch14]
        ra4_right_h.val[index] = vextq_u16(ra4_c_1_l_1_u16_h.val[index], ra4_c_2_l_1_u16_l.val[index], 1); // [ch9, ch10, ch11, ch12, ch13, ch14, ch15, rl0]

        // store the previous value before overwrite 
        ra4_c_1_l_1_u16_h_temp.val[index] = ra4_c_1_l_1_u16_h.val[index];

        // vertical sum
        ra4_c_1_l_1_u16_l.val[index] = vaddq_u16(vaddq_u16(ra4_left_l.val[index], ra4_c_1_l_1_u16_l.val[index]), ra4_right_l.val[index]); // sum of the lower part
        ra4_c_1_l_1_u16_h.val[index] = vaddq_u16(vaddq_u16(ra4_left_h.val[index], ra4_c_1_l_1_u16_h.val[index]), ra4_right_h.val[index]); // sum of the higher part

        // 7. promotion to uint32x4_t
        ra4_sum_l_l.val[index] = vmovl_u16(vget_low_u16(ra4_c_1_l_1_u16_l.val[index]));
        ra4_sum_l_h.val[index] = vmovl_u16(vget_high_u16(ra4_c_1_l_1_u16_l.val[index]));
        ra4_sum_h_l.val[index] = vmovl_u16(vget_low_u16(ra4_c_1_l_1_u16_h.val[index]));
        ra4_sum_h_h.val[index] = vmovl_u16(vget_high_u16(ra4_c_1_l_1_u16_h.val[index]));
      
        // 7. convert to float32x4_t
        ra4_sumf_l_l.val[index] = vcvtq_n_f32_u32(ra4_sum_l_l.val[index], 10);
        ra4_sumf_l_h.val[index] = vcvtq_n_f32_u32(ra4_sum_l_h.val[index], 10);
        ra4_sumf_h_l.val[index] = vcvtq_n_f32_u32(ra4_sum_h_l.val[index], 10);
        ra4_sumf_h_h.val[index] = vcvtq_n_f32_u32(ra4_sum_h_h.val[index], 10);
        
        // 8. divison by 9
        ra4_sumf_l_l.val[index] = vdivq_f32(ra4_sumf_l_l.val[index], R_NINE);
        ra4_sumf_l_h.val[index] = vdivq_f32(ra4_sumf_l_h.val[index], R_NINE);
        ra4_sumf_h_l.val[index] = vdivq_f32(ra4_sumf_h_l.val[index], R_NINE);
        ra4_sumf_h_h.val[index] = vdivq_f32(ra4_sumf_h_h.val[index], R_NINE);

        // 9. convert back to uint32x4_t
        ra4_sum_l_l.val[index] = vcvtq_n_u32_f32(ra4_sumf_l_l.val[index], 10);
        ra4_sum_l_h.val[index] = vcvtq_n_u32_f32(ra4_sumf_l_h.val[index], 10);
        ra4_sum_h_l.val[index] = vcvtq_n_u32_f32(ra4_sumf_h_l.val[index], 10);
        ra4_sum_h_h.val[index] = vcvtq_n_u32_f32(ra4_sumf_h_h.val[index], 10);

        // 10. convert back to uint16x8_t 
        ra4_c_1_l_1_u16_l.val[index] = vcombine_u16(vqmovn_u32(ra4_sum_l_l.val[index]), vqmovn_u32(ra4_sum_l_h.val[index]));
        ra4_c_1_l_1_u16_h.val[index] = vcombine_u16(vqmovn_u32(ra4_sum_h_l.val[index]), vqmovn_u32(ra4_sum_h_h.val[index]));

        // 11. convert back to uint8x16_t
        ra4_sum_u8.val[index] = vcombine_u8(vqmovn_u16(ra4_c_1_l_1_u16_l.val[index]), vqmovn_u16(ra4_c_1_l_1_u16_h.val[index]));

      }
      // 12. store 
      // use vst4 to store back and interleave the data
      vst4q_u8((uint8_t*)&next_img(i, j), ra4_sum_u8);

      // 13. variable rotation
      // copy the lowest part of the middle column into the highest part of the left column ( we should pass the sum of the pixels)
      // col 0 <- col 1
      ra4_c_0_l_1_u16_h = ra4_c_1_l_1_u16_h_temp; 
      // col 1 <- col 2
      ra4_c_1_l_1_u16_l = ra4_c_2_l_1_u16_l;
      ra4_c_1_l_1_u16_h = ra4_c_2_l_1_u16_h;

    }
  }

  // left-right borders size
  uint32_t bsize = 1;
  // compute the borders
  compute_borders(x, y, width, height, bsize);

  return 0;
}


// urrot1_neon_div9_u16
/*
 * In this version, we will use neon_vdiv9_u16 to perform the division by 9
 * using a series of shifts and additions. In this way we can avoid the division
 * operation and the corresponding cost of conversion to float and back to integer.
 * 
 * D2: 453.231  ms
 * CA57: 775.876  ms
 * 
 * In this version, we observe a dramatic performance speedup of 2.35 on the Denver 2
 * and a slight performance speedup of 1.09 on the Cortex-A57 compared to the 
 * urrot1_neon_div9_f32. This suggests that the overhead of the conversion to float
 * and the division by 9 may be the real bottleneck in D2. Also, if we look 
 * at the Cortex A57 documentation, we can see that the cvt instruction has a latency
 * of 8 cycles, which is quite high. This may be the reason for the speedup on the
 * Cortex A57. 
 * 
 */
int blur2_do_tile_urrot1_neon_div9_u16 (int x, int y, int width, int height) {
  /* #########################################################################
   * #########################################################################
   * Variables notation: ^r(a[0-4])?_c_[0-3]_l_[0-2]_u[8,16,32,64]
   * r -> register
   * a[0-4] -> array of registers and its dimension
   * c_[0-2] -> column number
   * l_[0-2] -> line number
   * u[8,16,32,64] -> type of the registers
   * 
   * Example ra4_c_2_l_0_u8 -> array of four registers u_0, right column, first line 
   * #########################################################################
   * #########################################################################
   */

  /*
   * Contain the deinterlived pixel colors of the right group column. One array
   * for each line.
   */
  uint8x16x4_t ra4_c_0_l_0_u8, ra4_c_0_l_1_u8, ra4_c_0_l_2_u8;
  uint8x16x4_t ra4_c_1_l_0_u8, ra4_c_1_l_1_u8, ra4_c_1_l_2_u8;
  uint8x16x4_t ra4_c_2_l_0_u8, ra4_c_2_l_1_u8, ra4_c_2_l_2_u8;
  uint8x16x4_t ra4_sum_u8;


  /**
   * Two array, one for the lower and one for higher part, for each
   * line and for each column.
   */  
  uint16x8x4_t ra4_c_0_l_0_u16_l; // left column, first line, no need for higher part
  uint16x8x4_t ra4_c_0_l_1_u16_h, ra4_c_0_l_1_u16_l; // left column, second line
  uint16x8x4_t ra4_c_0_l_2_u16_l; // left column, third line, no need for higher part
  
  
  uint16x8x4_t ra4_c_1_l_0_u16_h, ra4_c_1_l_0_u16_l; // central column, first line
  uint16x8x4_t ra4_c_1_l_1_u16_h, ra4_c_1_l_1_u16_l; // central column, second line
  uint16x8x4_t ra4_c_1_l_2_u16_h, ra4_c_1_l_2_u16_l; // central column, third line
  
  uint16x8x4_t ra4_c_2_l_0_u16_h, ra4_c_2_l_0_u16_l; // right column, first line
  uint16x8x4_t ra4_c_2_l_1_u16_h, ra4_c_2_l_1_u16_l; // right column, second line
  uint16x8x4_t ra4_c_2_l_2_u16_h, ra4_c_2_l_2_u16_l; // right column, third line

  // for storing the sum of the pixels on the higher part of the central column
  // used in variable reduction
  uint16x8x4_t ra4_c_1_l_1_u16_h_temp;

  // To store shiffted values for the reduction
  uint16x8x4_t ra4_left_l, ra4_left_h;
  uint16x8x4_t ra4_right_l, ra4_right_h; 

  // loop over y (start from +1, end at -1 => no border)
  for (int i = y + 1; i < y + height - 1; i++) {
    // #############################################################
    //                      PROLOGUE
    // #############################################################
    /*
     * In order to start the variable rotation, we need to precompute
     * the left and central columns. The computation of the central column
     * is done in the same way as the right column in the main loop. 
     * Since the x-loop starts from the second pixel, the left column 
     * is composed by only the first pixel of each line. So, one 
     * possible strategy is to load to perform the exact same computation
     * as the central column, but starting from the first pixel. After that,
     * we can use vextq_u16 to shift the first pixel to the last position.
     * 
     * In this way, we can use simd instructions to perform the computation
     * also for the first 16 pixels of each line instead of using scalar
     * instructions.
     */ 
    ra4_c_0_l_0_u8 = vld4q_u8((uint8_t*)&cur_img(i - 1, x));
    ra4_c_0_l_1_u8 = vld4q_u8((uint8_t*)&cur_img(i + 0, x));
    ra4_c_0_l_2_u8 = vld4q_u8((uint8_t*)&cur_img(i + 1, x));
    
    ra4_c_1_l_0_u8 = vld4q_u8((uint8_t*)&cur_img(i - 1, x + 1));
    ra4_c_1_l_1_u8 = vld4q_u8((uint8_t*)&cur_img(i + 0, x + 1));
    ra4_c_1_l_2_u8 = vld4q_u8((uint8_t*)&cur_img(i + 1, x + 1));


    for(int index=0; index<4; index++){
      /*
       * please note that we need only the lower part of the 16 pixeò for the 
       * left column. The higher part is not used in the computation but only
       * to store the first pixel for the computation in the loop.
       */
      // first line
      ra4_c_0_l_0_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_0_l_0_u8.val[index]));
      
      ra4_c_1_l_0_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_1_l_0_u8.val[index]));
      ra4_c_1_l_0_u16_h.val[index] = vmovl_u8(vget_high_u8(ra4_c_1_l_0_u8.val[index]));

      // second line
      ra4_c_0_l_1_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_0_l_1_u8.val[index]));
      
      ra4_c_1_l_1_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_1_l_1_u8.val[index]));
      ra4_c_1_l_1_u16_h.val[index] = vmovl_u8(vget_high_u8(ra4_c_1_l_1_u8.val[index]));

      // third line
      ra4_c_0_l_2_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_0_l_2_u8.val[index]));
      
      ra4_c_1_l_2_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_1_l_2_u8.val[index]));
      ra4_c_1_l_2_u16_h.val[index] = vmovl_u8(vget_high_u8(ra4_c_1_l_2_u8.val[index]));

      // reduction
      ra4_c_0_l_1_u16_l.val[index] = vaddq_u16(ra4_c_0_l_2_u16_l.val[index],
                                              vaddq_u16(ra4_c_0_l_1_u16_l.val[index], ra4_c_0_l_0_u16_l.val[index])); // lower part
      
      ra4_c_1_l_1_u16_l.val[index] = vaddq_u16(ra4_c_1_l_2_u16_l.val[index],
                                              vaddq_u16(ra4_c_1_l_1_u16_l.val[index], ra4_c_1_l_0_u16_l.val[index])); // lower part
      ra4_c_1_l_1_u16_h.val[index] = vaddq_u16(ra4_c_1_l_2_u16_h.val[index],
                                              vaddq_u16(ra4_c_1_l_1_u16_h.val[index], ra4_c_1_l_0_u16_h.val[index])); // higher part

      // move the first bit to the last position using vextq_u16
      ra4_c_0_l_1_u16_h.val[index] = vextq_u16(ra4_c_0_l_1_u16_h.val[index], ra4_c_0_l_1_u16_l.val[index], 1);    
    }
    

    // #############################################################
    //                      END PROLOGUE
    // #############################################################
    
    // loop over x (start from +1, end at -1 => no border)
    for (int j = x + 1; j < x + width; j+=16) {

      // 3. Memory deinterliving
      /* 
       * Use vld4q_u8 instructions to perform a deinterleving of the four pixel components.
       * For all the three lines of the right column-group, we load from memory 16 pixels 
       * deinterliving the colors. Now each group of pixel is composed by an array of 4 registers 
       * 8b x 16 elements.
       */
      ra4_c_2_l_0_u8 = vld4q_u8((uint8_t*)&cur_img(i - 1, j + 16)); // [[ r x 16 ] [ g x 16 ] [ b x 16 ] [ a x 16 ]]
      ra4_c_2_l_1_u8 = vld4q_u8((uint8_t*)&cur_img(i + 0, j + 16));
      ra4_c_2_l_2_u8 = vld4q_u8((uint8_t*)&cur_img(i + 1, j + 16));
      

      for(int index=0; index<4; index++){
        // 4. 
        /*
         * Promote the 8-bit components into 16-bit components to perform the accumulation
         * First we extract the lower and higher part of the 8-bit components using the 
         * vget_low_u8 and vget_high_u8.
         * Then we promote them to 16-bit components using the vmovl_u8 instruction.
         *  
         */
        // first line
        ra4_c_2_l_0_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_2_l_0_u8.val[index]));
        ra4_c_2_l_0_u16_h.val[index] = vmovl_u8(vget_high_u8(ra4_c_2_l_0_u8.val[index]));

        // second line
        ra4_c_2_l_1_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_2_l_1_u8.val[index]));
        ra4_c_2_l_1_u16_h.val[index] = vmovl_u8(vget_high_u8(ra4_c_2_l_1_u8.val[index]));

        // third line
        ra4_c_2_l_2_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_2_l_2_u8.val[index]));
        ra4_c_2_l_2_u16_h.val[index] = vmovl_u8(vget_high_u8(ra4_c_2_l_2_u8.val[index]));

        // 5. Compute the reduction of the second column
        /*
         * Accumulate the color component of the right column-group. The accumulation is performed
         * using 16 bit registers.
         */
        ra4_c_2_l_1_u16_l.val[index] = vaddq_u16(ra4_c_2_l_2_u16_l.val[index],
                                                vaddq_u16(ra4_c_2_l_1_u16_l.val[index], ra4_c_2_l_0_u16_l.val[index])); // lower part
        ra4_c_2_l_1_u16_h.val[index] = vaddq_u16(ra4_c_2_l_2_u16_h.val[index],
                                                vaddq_u16(ra4_c_2_l_1_u16_h.val[index], ra4_c_2_l_0_u16_h.val[index])); // higher part

        // 6. left-right pattern
        /*
         * Perform the left-right pattern to compute the sum of the pixel components. 
         *
         * [lh8, lh9, lh10, lh11, lh12, lh13, lh14, lh15] -> ra4_c_0_l_1_u16_h
         * [cl0, cl1, cl2, cl3, cl4, cl5, cl6, cl7] -> ra4_c_1_l_1_u16_l
         * [ch8, ch9, ch10, ch11, ch12, ch13, ch14, ch15] -> ra4_c_1_l_1_u16_h
         * [rl0, rl1, rl2, rl3, rl4, rl5, rl6, rl7] -> ra4_c_2_l_1_u16_l
         * 
         * From horizontal to vertical computation:
         * - lower part:
         * [cl1, cl2, cl3, cl4, cl5, cl6, cl7, ch8] -> right lower
         * [cl0, cl1, cl2, cl3, cl4, cl5, cl6, cl7] -> central lower
         * [lh15, cl0, cl1, cl2, cl3, cl4, cl5, cl6] -> left lower
         * 
         * - higher part:
         * [ch9, ch10, ch11, ch12, ch13, ch14, ch15, rl0] -> right higher
         * [ch8, ch9, ch10, ch11, ch12, ch13, ch14, rl0] -> central higher
         * [cl7, ch8, ch9, ch10, ch11, ch12, ch13, ch14] -> left higher
         * 
         * 
         */
        ra4_left_l.val[index]  = vextq_u16(ra4_c_0_l_1_u16_h.val[index], ra4_c_1_l_1_u16_l.val[index], 7);  // [lh15, cl0, cl1, cl2, cl3, cl4, cl5, cl6]
        ra4_right_l.val[index] = vextq_u16(ra4_c_1_l_1_u16_l.val[index], ra4_c_1_l_1_u16_h.val[index], 1); // [cl1, cl2, cl3, cl4, cl5, cl6, cl7, ch8]

        ra4_left_h.val[index]  = vextq_u16(ra4_c_1_l_1_u16_l.val[index], ra4_c_1_l_1_u16_h.val[index], 7); //[cl7, ch8, ch9, ch10, ch11, ch12, ch13, ch14]
        ra4_right_h.val[index] = vextq_u16(ra4_c_1_l_1_u16_h.val[index], ra4_c_2_l_1_u16_l.val[index], 1); // [ch9, ch10, ch11, ch12, ch13, ch14, ch15, rl0]

        // store the previous value before overwrite 
        ra4_c_1_l_1_u16_h_temp.val[index] = ra4_c_1_l_1_u16_h.val[index];

        // vertical sum
        ra4_c_1_l_1_u16_l.val[index] = vaddq_u16(vaddq_u16(ra4_left_l.val[index], ra4_c_1_l_1_u16_l.val[index]), ra4_right_l.val[index]); // sum of the lower part
        ra4_c_1_l_1_u16_h.val[index] = vaddq_u16(vaddq_u16(ra4_left_h.val[index], ra4_c_1_l_1_u16_h.val[index]), ra4_right_h.val[index]); // sum of the higher part

        // 8. divison by 9 using the neon_vdiv9_u16 function
        ra4_c_1_l_1_u16_l.val[index] = neon_vdiv9_u16(ra4_c_1_l_1_u16_l.val[index]);
        ra4_c_1_l_1_u16_h.val[index] = neon_vdiv9_u16(ra4_c_1_l_1_u16_h.val[index]);

        // 11. convert back to uint8x16_t
        ra4_sum_u8.val[index] = vcombine_u8(vqmovn_u16(ra4_c_1_l_1_u16_l.val[index]), vqmovn_u16(ra4_c_1_l_1_u16_h.val[index]));
      }
      
      // 12. store 
      // use vst4 to store back and interleave the data
      vst4q_u8((uint8_t*)&next_img(i, j), ra4_sum_u8);

      // 13. variable rotation
      // copy the lowest part of the middle column into the highest part of the left column ( we should pass the sum of the pixels)
      // col 0 <- col 1
      ra4_c_0_l_1_u16_h = ra4_c_1_l_1_u16_h_temp; 
      // col 1 <- col 2
      ra4_c_1_l_1_u16_l = ra4_c_2_l_1_u16_l;
      ra4_c_1_l_1_u16_h = ra4_c_2_l_1_u16_h;

    }
  }

  // left-right borders size
  uint32_t bsize = 1;
  // compute the borders
  compute_borders(x, y, width, height, bsize);

  return 0;
}

// urrot2_neon_div9_f32
/**
 * In this version, the pixel colors will be left interleaved in the registers
 * and the computation will be performed on 4 pixels at a time. In this way, we
 * can avoid inserting another nested loop to handle each color component separately, 
 * at the cost of increasing the number of loop iterations on the x-axis.
 * 
 * D2: 1041.107  ms
 * CA57: 938.001  ms 
 * 
 * In both cases, we do not observe a significant performance change compared 
 * to the urrot1_neon_div9_f32. This is in contrast to the performance speedup
 * observed from urrot1 to urrot2 in the scalar version.    
 */
int blur2_do_tile_urrot2_neon_div9_f32 (int x, int y, int width, int height) {
  /* #########################################################################
   * #########################################################################
   * Variables notation: ^r(a[0-4])?_c_[0-3]_l_[0-2]_u[8,16,32,64]
   * r -> register
   * a[0-4] -> array of registers and its dimension
   * c_[0-2] -> column number
   * l_[0-2] -> line number
   * u[8,16,32,64] -> type of the registers
   * 
   * Example ra4_c_2_l_0_u8 -> array of four registers u_0, right column, first line 
   * #########################################################################
   * #########################################################################
   */

  /*
   * Contain the interleaved pixel colors of the right group column. 
   * Four pixels are interleaved in a single register.
   */
  uint8x16_t r_c_0_l_0_u8, r_c_0_l_1_u8, r_c_0_l_2_u8;
  uint8x16_t r_c_1_l_0_u8, r_c_1_l_1_u8, r_c_1_l_2_u8;
  uint8x16_t r_c_2_l_0_u8, r_c_2_l_1_u8, r_c_2_l_2_u8;
  uint8x16_t r_sum_u8;

  /**
   * Two array, one for the lower and one for higher part, for each
   * line and for each column.
   */  
  uint16x8_t r_c_0_l_0_u16_l; // left column, first line, no need for higher part
  uint16x8_t r_c_0_l_1_u16_h, r_c_0_l_1_u16_l; // left column, second line
  uint16x8_t r_c_0_l_2_u16_l; // left column, third line, no need for higher part
  
  uint16x8_t r_c_1_l_0_u16_h, r_c_1_l_0_u16_l; // central column, first line
  uint16x8_t r_c_1_l_1_u16_h, r_c_1_l_1_u16_l; // central column, second line
  uint16x8_t r_c_1_l_2_u16_h, r_c_1_l_2_u16_l; // central column, third line
  
  uint16x8_t r_c_2_l_0_u16_h, r_c_2_l_0_u16_l; // right column, first line
  uint16x8_t r_c_2_l_1_u16_h, r_c_2_l_1_u16_l; // right column, second line
  uint16x8_t r_c_2_l_2_u16_h, r_c_2_l_2_u16_l; // right column, third line

  // for storing the sum of the pixels on the higher part of the central column
  // used in variable reduction
  uint16x8_t r_c_1_l_1_u16_h_temp;

  // To store shiffted values for the reduction
  uint16x8_t r_left_l, r_left_h;
  uint16x8_t r_right_l, r_right_h; 

  // for the promotion to 32 bits
  uint32x4_t r_sum_l_l, r_sum_l_h, r_sum_h_l, r_sum_h_h;

  // for the division by 9
  float32x4_t r_sumf_l_l, r_sumf_l_h, r_sumf_h_l, r_sumf_h_h;
  float32_t p = 9;
  float32x4_t R_NINE = vld1q_dup_f32(&p); // broadcast the 

  // loop over y (start from +1, end at -1 => no border)
  for (int i = y + 1; i < y + height - 1; i++) {
   // #############################################################
    //                      PROLOGUE
    // #############################################################
    /*
     * In order to start the variable rotation, we need to precompute
     * the left and central columns. The computation of the central column
     * is done in the same way as the right column in the main loop. 
     * Since the x-loop starts from the second pixel, the left column 
     * is composed by only the first pixel of each line. So, one 
     * possible strategy is to load to perform the exact same computation
     * as the central column, but starting from the first pixel. After that,
     * we can use vextq_u16 to shift the first pixel to the last position.
     * 
     * In this way, we can use simd instructions to perform the computation
     * also for the first 4 pixels of each line instead of using scalar
     * instructions.
     */ 
    {
      r_c_0_l_0_u8 = vld1q_u8((uint8_t*)&cur_img(i - 1, x));
      r_c_0_l_1_u8 = vld1q_u8((uint8_t*)&cur_img(i + 0, x));
      r_c_0_l_2_u8 = vld1q_u8((uint8_t*)&cur_img(i + 1, x));
      
      r_c_1_l_0_u8 = vld1q_u8((uint8_t*)&cur_img(i - 1, x + 1));
      r_c_1_l_1_u8 = vld1q_u8((uint8_t*)&cur_img(i + 0, x + 1));
      r_c_1_l_2_u8 = vld1q_u8((uint8_t*)&cur_img(i + 1, x + 1));

      /*
      * please note that we need only the lower part of the 16 pixel for the 
      * left column. The higher part is not used in the computation but only
      * to store the first pixel for the computation in the loop.
      */
      // first line
      r_c_0_l_0_u16_l = vmovl_u8(vget_low_u8(r_c_0_l_0_u8));
      
      r_c_1_l_0_u16_l = vmovl_u8(vget_low_u8(r_c_1_l_0_u8));
      r_c_1_l_0_u16_h = vmovl_u8(vget_high_u8(r_c_1_l_0_u8));

      // second line
      r_c_0_l_1_u16_l = vmovl_u8(vget_low_u8(r_c_0_l_1_u8));
      
      r_c_1_l_1_u16_l = vmovl_u8(vget_low_u8(r_c_1_l_1_u8));
      r_c_1_l_1_u16_h = vmovl_u8(vget_high_u8(r_c_1_l_1_u8));

      // third line
      r_c_0_l_2_u16_l = vmovl_u8(vget_low_u8(r_c_0_l_2_u8));
      
      r_c_1_l_2_u16_l = vmovl_u8(vget_low_u8(r_c_1_l_2_u8));
      r_c_1_l_2_u16_h = vmovl_u8(vget_high_u8(r_c_1_l_2_u8));

      // reduction
      r_c_0_l_1_u16_l = vaddq_u16(r_c_0_l_2_u16_l,
                                    vaddq_u16(r_c_0_l_1_u16_l, r_c_0_l_0_u16_l)); // lower part
      
      r_c_1_l_1_u16_l = vaddq_u16(r_c_1_l_2_u16_l,
                                    vaddq_u16(r_c_1_l_1_u16_l, r_c_1_l_0_u16_l)); // lower part
      r_c_1_l_1_u16_h = vaddq_u16(r_c_1_l_2_u16_h,
                                    vaddq_u16(r_c_1_l_1_u16_h, r_c_1_l_0_u16_h)); // higher part

      // move the first bit to the last position using vextq_u16
      r_c_0_l_1_u16_h = vextq_u16(r_c_0_l_1_u16_h, r_c_0_l_1_u16_l, 4);    
    }
    // #############################################################
    //                      END PROLOGUE
    // #############################################################

    // loop over x (start from +1, end at -1 => no border)
    for (int j = x + 1; j < x + width; j+=4) {

      // 3. Memory load
      /* 
       * Use vld1q_u8 instructions to perform an interleved load of the four pixel components.
       * For all the three lines of the right column-group, we load from memory 4 pixels,
       * leaving interleaved the colors. Now each register is composed by 4 pixels.
       */
      r_c_2_l_0_u8 = vld1q_u8((uint8_t*)&cur_img(i - 1, j + 4)); // [[ r1 g1 b1 a1 ] [r2 g2 b2 a2] [r3 g3 b3 a3] [ r4 g4 b4 a4 ]]
      r_c_2_l_1_u8 = vld1q_u8((uint8_t*)&cur_img(i + 0, j + 4));
      r_c_2_l_2_u8 = vld1q_u8((uint8_t*)&cur_img(i + 1, j + 4));
      
      // 4. 
      /*
       * Promote the 8-bit components into 16-bit components to perform the accumulation
       * First we extract the lower and higher part of the 8-bit components using the 
       * vget_low_u8 and vget_high_u8.
       * Then we promote them to 16-bit components using the vmovl_u8 instruction.
       *  
       */
      // first line
      r_c_2_l_0_u16_l = vmovl_u8(vget_low_u8(r_c_2_l_0_u8));
      r_c_2_l_0_u16_h = vmovl_u8(vget_high_u8(r_c_2_l_0_u8));

      // second line
      r_c_2_l_1_u16_l = vmovl_u8(vget_low_u8(r_c_2_l_1_u8));
      r_c_2_l_1_u16_h = vmovl_u8(vget_high_u8(r_c_2_l_1_u8));

      // third line
      r_c_2_l_2_u16_l = vmovl_u8(vget_low_u8(r_c_2_l_2_u8));
      r_c_2_l_2_u16_h = vmovl_u8(vget_high_u8(r_c_2_l_2_u8));

      // 5. Compute the reduction of the second column
      /*
       * Accumulate the color component of the right column-group.
       */
      r_c_2_l_1_u16_l = vaddq_u16(r_c_2_l_2_u16_l,
                                              vaddq_u16(r_c_2_l_1_u16_l, r_c_2_l_0_u16_l)); // lower part
      r_c_2_l_1_u16_h = vaddq_u16(r_c_2_l_2_u16_h,
                                              vaddq_u16(r_c_2_l_1_u16_h, r_c_2_l_0_u16_h)); // higher part

      // 6. left-right pattern
      /*
        * Perform the left-right pattern to compute the sum of the pixel components. 
        * Now, since colors are interleaved, we need to shift 4 colors when preparing the
        * left and right part of the sum. 
        */
      r_left_l  = vextq_u16(r_c_0_l_1_u16_h, r_c_1_l_1_u16_l, 4);  // [lh15, cl0, cl1, cl2, cl3, cl4, cl5, cl6]
      r_right_l = vextq_u16(r_c_1_l_1_u16_l, r_c_1_l_1_u16_h, 4); // [cl1, cl2, cl3, cl4, cl5, cl6, cl7, ch8]

      r_left_h  = vextq_u16(r_c_1_l_1_u16_l, r_c_1_l_1_u16_h, 4); //[cl7, ch8, ch9, ch10, ch11, ch12, ch13, ch14]
      r_right_h = vextq_u16(r_c_1_l_1_u16_h, r_c_2_l_1_u16_l, 4); // [ch9, ch10, ch11, ch12, ch13, ch14, ch15, rl0]

      // store the previous value before overwrite 
      r_c_1_l_1_u16_h_temp = r_c_1_l_1_u16_h;

      // vertical sum
      r_c_1_l_1_u16_l = vaddq_u16(vaddq_u16(r_left_l, r_c_1_l_1_u16_l), r_right_l); // sum of the lower part
      r_c_1_l_1_u16_h = vaddq_u16(vaddq_u16(r_left_h, r_c_1_l_1_u16_h), r_right_h); // sum of the higher part

      // 7. promotion to uint32x4_t
      r_sum_l_l = vmovl_u16(vget_low_u16(r_c_1_l_1_u16_l));
      r_sum_l_h = vmovl_u16(vget_high_u16(r_c_1_l_1_u16_l));
      r_sum_h_l = vmovl_u16(vget_low_u16(r_c_1_l_1_u16_h));
      r_sum_h_h = vmovl_u16(vget_high_u16(r_c_1_l_1_u16_h));
    
      // 7. convert to float32x4_t
      r_sumf_l_l = vcvtq_n_f32_u32(r_sum_l_l, 10);
      r_sumf_l_h = vcvtq_n_f32_u32(r_sum_l_h, 10);
      r_sumf_h_l = vcvtq_n_f32_u32(r_sum_h_l, 10);
      r_sumf_h_h = vcvtq_n_f32_u32(r_sum_h_h, 10);
      
      // 8. divison by 9
      r_sumf_l_l = vdivq_f32(r_sumf_l_l, R_NINE);
      r_sumf_l_h = vdivq_f32(r_sumf_l_h, R_NINE);
      r_sumf_h_l = vdivq_f32(r_sumf_h_l, R_NINE);
      r_sumf_h_h = vdivq_f32(r_sumf_h_h, R_NINE);

      // 9. convert back to uint32x4_t
      r_sum_l_l = vcvtq_n_u32_f32(r_sumf_l_l, 10);
      r_sum_l_h = vcvtq_n_u32_f32(r_sumf_l_h, 10);
      r_sum_h_l = vcvtq_n_u32_f32(r_sumf_h_l, 10);
      r_sum_h_h = vcvtq_n_u32_f32(r_sumf_h_h, 10);

      // 10. convert back to uint16x8_t 
      r_c_1_l_1_u16_l = vcombine_u16(vqmovn_u32(r_sum_l_l), vqmovn_u32(r_sum_l_h));
      r_c_1_l_1_u16_h = vcombine_u16(vqmovn_u32(r_sum_h_l), vqmovn_u32(r_sum_h_h));

      // 11. convert back to uint8x16_t
      r_sum_u8 = vcombine_u8(vqmovn_u16(r_c_1_l_1_u16_l), vqmovn_u16(r_c_1_l_1_u16_h));

      // 12. store 
      // use vst1 to store back the data
      vst1q_u8((uint8_t*)&next_img(i, j), r_sum_u8);

      // 13. variable rotation
      // copy the lowest part of the middle column into the highest part of the left column ( we should pass the sum of the pixels)
      // col 0 <- col 1
      r_c_0_l_1_u16_h = r_c_1_l_1_u16_h_temp; 
      // col 1 <- col 2
      r_c_1_l_1_u16_l = r_c_2_l_1_u16_l;
      r_c_1_l_1_u16_h = r_c_2_l_1_u16_h;

    }
  }

  // left-right borders size
  uint32_t bsize = 1;
  // compute the borders
  compute_borders(x, y, width, height, bsize);

  return 0;
}

// urrot2_neon_div9_u16
/**
 * As in urrot1_neon_div9_u16, we can use the same strategy to perform the division
 * by 9 using the vdiv9_u16 function. 
 * 
 * D2: 363.881  ms
 * CA57: 708.589  ms
 * 
 * We observe a similar performance speedup as in the urrot1_neon_div9_u16 version.
 */
int blur2_do_tile_urrot2_neon_div9_u16 (int x, int y, int width, int height) {
  /* #########################################################################
   * #########################################################################
   * Variables notation: ^r(a[0-4])?_c_[0-3]_l_[0-2]_u[8,16,32,64]
   * r -> register
   * a[0-4] -> array of registers and its dimension
   * c_[0-2] -> column number
   * l_[0-2] -> line number
   * u[8,16,32,64] -> type of the registers
   * 
   * Example ra4_c_2_l_0_u8 -> array of four registers u_0, right column, first line 
   * #########################################################################
   * #########################################################################
   */

  /*
   * Contain the interleaved pixel colors of the right group column. 
   * Four pixels are interleaved in a single register.
   */
  uint8x16_t r_c_0_l_0_u8, r_c_0_l_1_u8, r_c_0_l_2_u8;
  uint8x16_t r_c_1_l_0_u8, r_c_1_l_1_u8, r_c_1_l_2_u8;
  uint8x16_t r_c_2_l_0_u8, r_c_2_l_1_u8, r_c_2_l_2_u8;
  uint8x16_t r_sum_u8;

  /**
   * Two array, one for the lower and one for higher part, for each
   * line and for each column.
   */  
  uint16x8_t r_c_0_l_0_u16_l; // left column, first line, no need for higher part
  uint16x8_t r_c_0_l_1_u16_h, r_c_0_l_1_u16_l; // left column, second line
  uint16x8_t r_c_0_l_2_u16_l; // left column, third line, no need for higher part
  
  uint16x8_t r_c_1_l_0_u16_h, r_c_1_l_0_u16_l; // central column, first line
  uint16x8_t r_c_1_l_1_u16_h, r_c_1_l_1_u16_l; // central column, second line
  uint16x8_t r_c_1_l_2_u16_h, r_c_1_l_2_u16_l; // central column, third line
  
  uint16x8_t r_c_2_l_0_u16_h, r_c_2_l_0_u16_l; // right column, first line
  uint16x8_t r_c_2_l_1_u16_h, r_c_2_l_1_u16_l; // right column, second line
  uint16x8_t r_c_2_l_2_u16_h, r_c_2_l_2_u16_l; // right column, third line

  // for storing the sum of the pixels on the higher part of the central column
  // used in variable reduction
  uint16x8_t r_c_1_l_1_u16_h_temp;

  // To store shiffted values for the reduction
  uint16x8_t r_left_l, r_left_h;
  uint16x8_t r_right_l, r_right_h; 

  // loop over y (start from +1, end at -1 => no border)
  for (int i = y + 1; i < y + height - 1; i++) {
   // #############################################################
    //                      PROLOGUE
    // #############################################################
    /*
     * In order to start the variable rotation, we need to precompute
     * the left and central columns. The computation of the central column
     * is done in the same way as the right column in the main loop. 
     * Since the x-loop starts from the second pixel, the left column 
     * is composed by only the first pixel of each line. So, one 
     * possible strategy is to load to perform the exact same computation
     * as the central column, but starting from the first pixel. After that,
     * we can use vextq_u16 to shift the first pixel to the last position.
     * 
     * In this way, we can use simd instructions to perform the computation
     * also for the first 4 pixels of each line instead of using scalar
     * instructions.
     */ 
    {
      r_c_0_l_0_u8 = vld1q_u8((uint8_t*)&cur_img(i - 1, x));
      r_c_0_l_1_u8 = vld1q_u8((uint8_t*)&cur_img(i + 0, x));
      r_c_0_l_2_u8 = vld1q_u8((uint8_t*)&cur_img(i + 1, x));
      
      r_c_1_l_0_u8 = vld1q_u8((uint8_t*)&cur_img(i - 1, x + 1));
      r_c_1_l_1_u8 = vld1q_u8((uint8_t*)&cur_img(i + 0, x + 1));
      r_c_1_l_2_u8 = vld1q_u8((uint8_t*)&cur_img(i + 1, x + 1));

      /*
      * please note that we need only the lower part of the 16 pixel for the 
      * left column. The higher part is not used in the computation but only
      * to store the first pixel for the computation in the loop.
      */
      // first line
      r_c_0_l_0_u16_l = vmovl_u8(vget_low_u8(r_c_0_l_0_u8));
      
      r_c_1_l_0_u16_l = vmovl_u8(vget_low_u8(r_c_1_l_0_u8));
      r_c_1_l_0_u16_h = vmovl_u8(vget_high_u8(r_c_1_l_0_u8));

      // second line
      r_c_0_l_1_u16_l = vmovl_u8(vget_low_u8(r_c_0_l_1_u8));
      
      r_c_1_l_1_u16_l = vmovl_u8(vget_low_u8(r_c_1_l_1_u8));
      r_c_1_l_1_u16_h = vmovl_u8(vget_high_u8(r_c_1_l_1_u8));

      // third line
      r_c_0_l_2_u16_l = vmovl_u8(vget_low_u8(r_c_0_l_2_u8));
      
      r_c_1_l_2_u16_l = vmovl_u8(vget_low_u8(r_c_1_l_2_u8));
      r_c_1_l_2_u16_h = vmovl_u8(vget_high_u8(r_c_1_l_2_u8));

      // reduction
      r_c_0_l_1_u16_l = vaddq_u16(r_c_0_l_2_u16_l,
                                    vaddq_u16(r_c_0_l_1_u16_l, r_c_0_l_0_u16_l)); // lower part
      
      r_c_1_l_1_u16_l = vaddq_u16(r_c_1_l_2_u16_l,
                                    vaddq_u16(r_c_1_l_1_u16_l, r_c_1_l_0_u16_l)); // lower part
      r_c_1_l_1_u16_h = vaddq_u16(r_c_1_l_2_u16_h,
                                    vaddq_u16(r_c_1_l_1_u16_h, r_c_1_l_0_u16_h)); // higher part

      // move the first bit to the last position using vextq_u16
      r_c_0_l_1_u16_h = vextq_u16(r_c_0_l_1_u16_h, r_c_0_l_1_u16_l, 4);    
    }
    // #############################################################
    //                      END PROLOGUE
    // #############################################################

    // loop over x (start from +1, end at -1 => no border)
    for (int j = x + 1; j < x + width; j+=4) {

      // 3. Memory interliving
      /* 
       * Use vld1q_u8 instructions to perform an interleved load of the four pixel components.
       * For all the three lines of the right column-group, we load from memory 4 pixels,
       * leaving interleaved the colors. Now each register is composed by 4 pixels.
       */
      r_c_2_l_0_u8 = vld1q_u8((uint8_t*)&cur_img(i - 1, j + 4)); // [[ r1 g1 b1 a1 ] [r2 g2 b2 a2] [r3 g3 b3 a3] [ r4 g4 b4 a4 ]]
      r_c_2_l_1_u8 = vld1q_u8((uint8_t*)&cur_img(i + 0, j + 4));
      r_c_2_l_2_u8 = vld1q_u8((uint8_t*)&cur_img(i + 1, j + 4));
      
      // 4. 
      /*
       * Promote the 8-bit components into 16-bit components to perform the accumulation
       * First we extract the lower and higher part of the 8-bit components using the 
       * vget_low_u8 and vget_high_u8.
       * Then we promote them to 16-bit components using the vmovl_u8 instruction.
       *  
       */
      // first line
      r_c_2_l_0_u16_l = vmovl_u8(vget_low_u8(r_c_2_l_0_u8));
      r_c_2_l_0_u16_h = vmovl_u8(vget_high_u8(r_c_2_l_0_u8));

      // second line
      r_c_2_l_1_u16_l = vmovl_u8(vget_low_u8(r_c_2_l_1_u8));
      r_c_2_l_1_u16_h = vmovl_u8(vget_high_u8(r_c_2_l_1_u8));

      // third line
      r_c_2_l_2_u16_l = vmovl_u8(vget_low_u8(r_c_2_l_2_u8));
      r_c_2_l_2_u16_h = vmovl_u8(vget_high_u8(r_c_2_l_2_u8));

      // 5. Compute the reduction of the second column
      /*
       * Accumulate the color component of the right column-group.
       */
      r_c_2_l_1_u16_l = vaddq_u16(r_c_2_l_2_u16_l,
                                              vaddq_u16(r_c_2_l_1_u16_l, r_c_2_l_0_u16_l)); // lower part
      r_c_2_l_1_u16_h = vaddq_u16(r_c_2_l_2_u16_h,
                                              vaddq_u16(r_c_2_l_1_u16_h, r_c_2_l_0_u16_h)); // higher part

      // 6. left-right pattern
      /*
        * Perform the left-right pattern to compute the sum of the pixel components. 
        * Now, since colors are interleaved, we need to shift 4 colors when preparing the
        * left and right part of the sum. 
        */
      r_left_l  = vextq_u16(r_c_0_l_1_u16_h, r_c_1_l_1_u16_l, 4);  // [lh15, cl0, cl1, cl2, cl3, cl4, cl5, cl6]
      r_right_l = vextq_u16(r_c_1_l_1_u16_l, r_c_1_l_1_u16_h, 4); // [cl1, cl2, cl3, cl4, cl5, cl6, cl7, ch8]

      r_left_h  = vextq_u16(r_c_1_l_1_u16_l, r_c_1_l_1_u16_h, 4); //[cl7, ch8, ch9, ch10, ch11, ch12, ch13, ch14]
      r_right_h = vextq_u16(r_c_1_l_1_u16_h, r_c_2_l_1_u16_l, 4); // [ch9, ch10, ch11, ch12, ch13, ch14, ch15, rl0]

      // store the previous value before overwrite 
      r_c_1_l_1_u16_h_temp = r_c_1_l_1_u16_h;

      // vertical sum
      r_c_1_l_1_u16_l = vaddq_u16(vaddq_u16(r_left_l, r_c_1_l_1_u16_l), r_right_l); // sum of the lower part
      r_c_1_l_1_u16_h = vaddq_u16(vaddq_u16(r_left_h, r_c_1_l_1_u16_h), r_right_h); // sum of the higher part

      // division by 9 using the neon_vdiv9_u16 function
      r_c_1_l_1_u16_l = neon_vdiv9_u16(r_c_1_l_1_u16_l);
      r_c_1_l_1_u16_h = neon_vdiv9_u16(r_c_1_l_1_u16_h);

      // 11. convert back to uint8x16_t
      r_sum_u8 = vcombine_u8(vqmovn_u16(r_c_1_l_1_u16_l), vqmovn_u16(r_c_1_l_1_u16_h));

      // 12. store 
      // use vst1 to store back the data
      vst1q_u8((uint8_t*)&next_img(i, j), r_sum_u8);

      // 13. variable rotation
      // copy the lowest part of the middle column into the highest part of the left column ( we should pass the sum of the pixels)
      // col 0 <- col 1
      r_c_0_l_1_u16_h = r_c_1_l_1_u16_h_temp; 
      // col 1 <- col 2
      r_c_1_l_1_u16_l = r_c_2_l_1_u16_l;
      r_c_1_l_1_u16_h = r_c_2_l_1_u16_h;

    }
  }

  // left-right borders size
  uint32_t bsize = 1;
  // compute the borders
  compute_borders(x, y, width, height, bsize);

  return 0;
}

// urrot2_neon_div8_u16
/**
 * In this version, we do a step further and we try to avoid completely the division
 * by 9. We can remove the central pixel from the sum of the pixels and then divide
 * by 8, which is equivalent to a shift of 3 bits to the right. This requires to
 * store the central column and remove it from the sum of the pixels. Obviously,
 * this will end up in a different result, but still a good approximation of the
 * blur effect. 
 * 
 * D2: 230.567  ms
 * CA57: 556.205  ms
 * 
 * We observe a speedup of 1.5x in the Denver 2 and 1.27x in the Cortex A57. This 
 * confirms that avoiding the division by 9 is a good strategy to improve the performance
 * of the blur effect.
 */
/**
 * 1024X1024*4 B = 4MB -> we are out of the L1 cache and L2 cache -> we are using the RAM
 */
int blur2_do_tile_urrot2_neon_div8_u16 (int x, int y, int width, int height) {
  /* #########################################################################
   * #########################################################################
   * Variables notation: ^r(a[0-4])?_c_[0-3]_l_[0-2]_u[8,16,32,64]
   * r -> register
   * a[0-4] -> array of registers and its dimension
   * c_[0-2] -> column number
   * l_[0-2] -> line number
   * u[8,16,32,64] -> type of the registers
   * 
   * Example ra4_c_2_l_0_u8 -> array of four registers u_0, right column, first line 
   * #########################################################################
   * #########################################################################
   */

  /*
   * Contain the interleaved pixel colors of the right group column. 
   * Four pixels are interleaved in a single register.
   */
  uint8x16_t r_c_0_l_0_u8, r_c_0_l_1_u8, r_c_0_l_2_u8;
  uint8x16_t r_c_1_l_0_u8, r_c_1_l_1_u8, r_c_1_l_2_u8;
  uint8x16_t r_c_2_l_0_u8, r_c_2_l_1_u8, r_c_2_l_2_u8;
  uint8x16_t r_sum_u8;

  /**
   * Two array, one for the lower and one for higher part, for each
   * line and for each column.
   */  
  uint16x8_t r_c_0_l_0_u16_l; // left column, first line, no need for higher part
  uint16x8_t r_c_0_l_1_u16_h, r_c_0_l_1_u16_l; // left column, second line
  uint16x8_t r_c_0_l_2_u16_l; // left column, third line, no need for higher part
  
  uint16x8_t r_c_1_l_0_u16_h, r_c_1_l_0_u16_l; // central column, first line
  uint16x8_t r_c_1_l_1_u16_h, r_c_1_l_1_u16_l; // central column, second line
  uint16x8_t r_c_1_l_2_u16_h, r_c_1_l_2_u16_l; // central column, third line
  
  uint16x8_t r_c_2_l_0_u16_h, r_c_2_l_0_u16_l; // right column, first line
  uint16x8_t r_c_2_l_1_u16_h, r_c_2_l_1_u16_l; // right column, second line
  uint16x8_t r_c_2_l_2_u16_h, r_c_2_l_2_u16_l; // right column, third line

  // for storing the sum of the pixels on the higher part of the central column
  // used in variable reduction
  uint16x8_t r_c_1_l_1_u16_h_temp;

  // To store shiffted values for the reduction
  uint16x8_t r_left_l, r_left_h;
  uint16x8_t r_right_l, r_right_h; 

  // store the central column in the first iteration
  uint16x8_t r_c_1_l_1_u16_l_central, r_c_1_l_1_u16_h_central;
  uint16x8_t r_c_2_l_1_u16_l_central, r_c_2_l_1_u16_h_central;

  // loop over y (start from +1, end at -1 => no border)
  for (int i = y + 1; i < y + height - 1; i++) {
   // #############################################################
    //                      PROLOGUE
    // #############################################################
    /*
     * In order to start the variable rotation, we need to precompute
     * the left and central columns. The computation of the central column
     * is done in the same way as the right column in the main loop. 
     * Since the x-loop starts from the second pixel, the left column 
     * is composed by only the first pixel of each line. So, one 
     * possible strategy is to load to perform the exact same computation
     * as the central column, but starting from the first pixel. After that,
     * we can use vextq_u16 to shift the first pixel to the last position.
     * 
     * In this way, we can use simd instructions to perform the computation
     * also for the first 4 pixels of each line instead of using scalar
     * instructions.
     */ 
    {
      /**
       * MEMORY OPERATIONS: each vld1q_u8 instruction loads 4 int from memory
       * 6*4 integers are loaded in total.
       */
      r_c_0_l_0_u8 = vld1q_u8((uint8_t*)&cur_img(i - 1, x)); 
      r_c_0_l_1_u8 = vld1q_u8((uint8_t*)&cur_img(i + 0, x));
      r_c_0_l_2_u8 = vld1q_u8((uint8_t*)&cur_img(i + 1, x));
      
      r_c_1_l_0_u8 = vld1q_u8((uint8_t*)&cur_img(i - 1, x + 1));
      r_c_1_l_1_u8 = vld1q_u8((uint8_t*)&cur_img(i + 0, x + 1));
      r_c_1_l_2_u8 = vld1q_u8((uint8_t*)&cur_img(i + 1, x + 1));
      /*
      * please note that we need only the lower part of the 16 pixel for the 
      * left column. The higher part is not used in the computation but only
      * to store the first pixel for the computation in the loop.
      */
      // first line
      r_c_0_l_0_u16_l = vmovl_u8(vget_low_u8(r_c_0_l_0_u8));
      
      r_c_1_l_0_u16_l = vmovl_u8(vget_low_u8(r_c_1_l_0_u8));
      r_c_1_l_0_u16_h = vmovl_u8(vget_high_u8(r_c_1_l_0_u8));

      // second line
      r_c_0_l_1_u16_l = vmovl_u8(vget_low_u8(r_c_0_l_1_u8));
      
      r_c_1_l_1_u16_l = vmovl_u8(vget_low_u8(r_c_1_l_1_u8));
      r_c_1_l_1_u16_h = vmovl_u8(vget_high_u8(r_c_1_l_1_u8));

      // NOTE: in the first we need to store the central pixels 
      // otherwise the variable rotation will not work.
      r_c_1_l_1_u16_l_central = r_c_1_l_1_u16_l;
      r_c_1_l_1_u16_h_central = r_c_1_l_1_u16_h;

      // third line
      r_c_0_l_2_u16_l = vmovl_u8(vget_low_u8(r_c_0_l_2_u8));
      
      r_c_1_l_2_u16_l = vmovl_u8(vget_low_u8(r_c_1_l_2_u8));
      r_c_1_l_2_u16_h = vmovl_u8(vget_high_u8(r_c_1_l_2_u8));

      // reduction
      r_c_0_l_1_u16_l = vaddq_u16(r_c_0_l_2_u16_l,
                                    vaddq_u16(r_c_0_l_1_u16_l, r_c_0_l_0_u16_l)); // lower part
      
      r_c_1_l_1_u16_l = vaddq_u16(r_c_1_l_2_u16_l,
                                    vaddq_u16(r_c_1_l_1_u16_l, r_c_1_l_0_u16_l)); // lower part
      r_c_1_l_1_u16_h = vaddq_u16(r_c_1_l_2_u16_h,
                                    vaddq_u16(r_c_1_l_1_u16_h, r_c_1_l_0_u16_h)); // higher part


      // move the first bit to the last position using vextq_u16
      r_c_0_l_1_u16_h = vextq_u16(r_c_0_l_1_u16_h, r_c_0_l_1_u16_l, 4);    
    }
    // #############################################################
    //                      END PROLOGUE
    // #############################################################

    // loop over x (start from +1, end at -1 => no border)
    for (int j = x + 1; j < x + width; j+=4) {

      // 3. Memory interliving
      /* 
       * Use vld1q_u8 instructions to perform an interleved load of the four pixel components.
       * For all the three lines of the right column-group, we load from memory 4 pixels,
       * leaving interleaved the colors. Now each register is composed by 4 pixels.
       */
      /**
       * MEMORY OPERATIONS: each vld1q_u8 instruction loads 4 int from memory
       * 3*4 integers are loaded in total.
       */
      r_c_2_l_0_u8 = vld1q_u8((uint8_t*)&cur_img(i - 1, j + 4)); // [[ r1 g1 b1 a1 ] [r2 g2 b2 a2] [r3 g3 b3 a3] [ r4 g4 b4 a4 ]]
      r_c_2_l_1_u8 = vld1q_u8((uint8_t*)&cur_img(i + 0, j + 4));
      r_c_2_l_2_u8 = vld1q_u8((uint8_t*)&cur_img(i + 1, j + 4));
      
      // 4. 
      /*
       * Promote the 8-bit components into 16-bit components to perform the accumulation
       * First we extract the lower and higher part of the 8-bit components using the 
       * vget_low_u8 and vget_high_u8.
       * Then we promote them to 16-bit components using the vmovl_u8 instruction.
       *  
       */
      // first line
      r_c_2_l_0_u16_l = vmovl_u8(vget_low_u8(r_c_2_l_0_u8));
      r_c_2_l_0_u16_h = vmovl_u8(vget_high_u8(r_c_2_l_0_u8));

      // second line
      r_c_2_l_1_u16_l = vmovl_u8(vget_low_u8(r_c_2_l_1_u8));
      r_c_2_l_1_u16_h = vmovl_u8(vget_high_u8(r_c_2_l_1_u8));

      // third line
      r_c_2_l_2_u16_l = vmovl_u8(vget_low_u8(r_c_2_l_2_u8));
      r_c_2_l_2_u16_h = vmovl_u8(vget_high_u8(r_c_2_l_2_u8));

      // save the central line before overwriting
      r_c_2_l_1_u16_l_central = r_c_2_l_1_u16_l;
      r_c_2_l_1_u16_h_central = r_c_2_l_1_u16_h;

      // 5. Compute the reduction of the second column
      /*
       * Accumulate the color component of the right column-group.
       */
      /**
       * ARITHMETIC OPERATIONS: each vaddq_u16 instruction performs 4 additions
       * 4*4 additions are performed in total.
       */
      r_c_2_l_1_u16_l = vaddq_u16(r_c_2_l_2_u16_l,
                                              vaddq_u16(r_c_2_l_1_u16_l, r_c_2_l_0_u16_l)); // lower part
      r_c_2_l_1_u16_h = vaddq_u16(r_c_2_l_2_u16_h,
                                              vaddq_u16(r_c_2_l_1_u16_h, r_c_2_l_0_u16_h)); // higher part

      // 6. left-right pattern
      /*
        * Perform the left-right pattern to compute the sum of the pixel components. 
        * Now, since colors are interleaved, we need to shift 4 colors when preparing the
        * left and right part of the sum. 
        */
      r_left_l  = vextq_u16(r_c_0_l_1_u16_h, r_c_1_l_1_u16_l, 4);  // [lh15, cl0, cl1, cl2, cl3, cl4, cl5, cl6]
      r_right_l = vextq_u16(r_c_1_l_1_u16_l, r_c_1_l_1_u16_h, 4); // [cl1, cl2, cl3, cl4, cl5, cl6, cl7, ch8]

      r_left_h  = vextq_u16(r_c_1_l_1_u16_l, r_c_1_l_1_u16_h, 4); //[cl7, ch8, ch9, ch10, ch11, ch12, ch13, ch14]
      r_right_h = vextq_u16(r_c_1_l_1_u16_h, r_c_2_l_1_u16_l, 4); // [ch9, ch10, ch11, ch12, ch13, ch14, ch15, rl0]

      // store the previous value before overwrite 
      r_c_1_l_1_u16_h_temp = r_c_1_l_1_u16_h;

      // vertical sum
      /**
       * ARITHMETIC OPERATIONS: each vaddq_u16 instruction performs 4 additions
       * 4*4 additions are performed in total.
       */
      r_c_1_l_1_u16_l = vaddq_u16(vaddq_u16(r_left_l, r_c_1_l_1_u16_l), r_right_l); // sum of the lower part
      r_c_1_l_1_u16_h = vaddq_u16(vaddq_u16(r_left_h, r_c_1_l_1_u16_h), r_right_h); // sum of the higher part

      // remove the central line
      /**
       * ARITHMETIC OPERATIONS: each vsubq_u16 instruction performs 4 subtractions
       * 2*4 subtractions are performed in total.
       */
      r_c_1_l_1_u16_l = vsubq_u16(r_c_1_l_1_u16_l, r_c_1_l_1_u16_l_central);
      r_c_1_l_1_u16_h = vsubq_u16(r_c_1_l_1_u16_h, r_c_1_l_1_u16_h_central);

      // division by 8 using just a vectorized shift operation
      /**
       * LOGICAL OPERATIONS: each vshrq_n_u16 instruction performs 4 shifts
       * 2*4 shifts are performed in total.
       */
      r_c_1_l_1_u16_l = vshrq_n_u16(r_c_1_l_1_u16_l, 3);
      r_c_1_l_1_u16_h = vshrq_n_u16(r_c_1_l_1_u16_h, 3);

      // 11. convert back to uint8x16_t
      r_sum_u8 = vcombine_u8(vqmovn_u16(r_c_1_l_1_u16_l), vqmovn_u16(r_c_1_l_1_u16_h));

      // 12. store 
      // use vst1 to store back the data
      /**
       * MEMORY OPERATIONS: each vst1q_u8 instruction stores 4 int to memory
       * 1*4 integers are stored in total.
       */
      vst1q_u8((uint8_t*)&next_img(i, j), r_sum_u8);

      // 13. variable rotation
      // copy the lowest part of the middle column into the highest part of the left column ( we should pass the sum of the pixels)
      // col 0 <- col 1
      r_c_0_l_1_u16_h = r_c_1_l_1_u16_h_temp; 

      // pass the central column
      r_c_1_l_1_u16_l_central = r_c_2_l_1_u16_l_central;
      r_c_1_l_1_u16_h_central = r_c_2_l_1_u16_h_central;

      // col 1 <- col 2
      r_c_1_l_1_u16_l = r_c_2_l_1_u16_l;
      r_c_1_l_1_u16_h = r_c_2_l_1_u16_h;

    }
  }

  // left-right borders size
  uint32_t bsize = 1;
  // compute the borders
  compute_borders(x, y, width, height, bsize);

  return 0;
}

#endif /* __ARM_NEON__ */