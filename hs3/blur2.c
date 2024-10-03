#include "easypap.h"

#include <omp.h>

#define __ARM_NEON__

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

void compute_borders(int x, int y, int width, int height, int bsize) {
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
  uint8x16x4_t ra4_c_0_l_0_u8, ra4_c_0_l_0_u8, ra4_c_0_l_0_u8;
  uint8x16x4_t ra4_c_1_l_0_u8, ra4_c_1_l_1_u8, ra4_c_1_l_2_u8;
  uint8x16x4_t ra4_c_2_l_0_u8, ra4_c_2_l_1_u8, ra4_c_2_l_2_u8;
  uint8x16x4_t ra4_sum_u8;

  /**
   * Contains the deinterlived pixel colors of the right group column
   * extended to 16 bits. Two array, one for the lower and on for higher part, for each
   * line and for each column.
   */
  uint16_t zero = 0;
  uint16x8x4_t ra4_c_0_l_1_u16_h; // left column, only the higher part of the cetral row of the pixels is needed
  
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
    //PROLOGO-----------------
    // another strategy could be doing the scalar sum and then using conversion 
    ra4_c_1_l_0_u8 = vld4q_u8((uint8_t*)&cur_img(i - 1, x));
    ra4_c_1_l_1_u8 = vld4q_u8((uint8_t*)&cur_img(i + 0, x));
    ra4_c_1_l_2_u8 = vld4q_u8((uint8_t*)&cur_img(i + 1, x));
    
    ra4_c_1_l_0_u8 = vld4q_u8((uint8_t*)&cur_img(i - 1, x + 1));
    ra4_c_1_l_1_u8 = vld4q_u8((uint8_t*)&cur_img(i + 0, x + 1));
    ra4_c_1_l_2_u8 = vld4q_u8((uint8_t*)&cur_img(i + 1, x + 1));
    


    for(int index=0; index<4; index++){
      // first line
      ra4_c_0_l_0_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_0_l_0_u8.val[index]));
      ra4_c_0_l_0_u16_h.val[index] = vmovl_u8(vget_high_u8(ra4_c_0_l_0_u8.val[index]));

      ra4_c_1_l_0_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_1_l_0_u8.val[index]));
      ra4_c_1_l_0_u16_h.val[index] = vmovl_u8(vget_high_u8(ra4_c_1_l_0_u8.val[index]));

      // second line
      ra4_c_0_l_1_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_0_l_1_u8.val[index]));
      ra4_c_0_l_1_u16_h.val[index] = vmovl_u8(vget_high_u8(ra4_c_0_l_1_u8.val[index]));
      
      ra4_c_1_l_1_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_1_l_1_u8.val[index]));
      ra4_c_1_l_1_u16_h.val[index] = vmovl_u8(vget_high_u8(ra4_c_1_l_1_u8.val[index]));

      // third line
      ra4_c_0_l_2_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_0_l_2_u8.val[index]));
      ra4_c_0_l_2_u16_h.val[index] = vmovl_u8(vget_high_u8(ra4_c_0_l_2_u8.val[index]));

      ra4_c_1_l_2_u16_l.val[index] = vmovl_u8(vget_low_u8(ra4_c_1_l_2_u8.val[index]));
      ra4_c_1_l_2_u16_h.val[index] = vmovl_u8(vget_high_u8(ra4_c_1_l_2_u8.val[index]));

      // reduction
      ra4_c_0_l_1_u16_l.val[index] = vaddq_u16(ra4_c_0_l_2_u16_l.val[index],
                                              vaddq_u16(ra4_c_1_l_1_u16_l.val[index], ra4_c_0_l_0_u16_l.val[index])); // lower part
      ra4_c_0_l_1_u16_h.val[index] = vaddq_u16(ra4_c_0_l_2_u16_h.val[index],
                                              vaddq_u16(ra4_c_1_l_1_u16_h.val[index], ra4_c_0_l_0_u16_h.val[index])); // higher part

      ra4_c_1_l_1_u16_l.val[index] = vaddq_u16(ra4_c_1_l_2_u16_l.val[index],
                                              vaddq_u16(ra4_c_1_l_1_u16_l.val[index], ra4_c_1_l_0_u16_l.val[index])); // lower part
      ra4_c_1_l_1_u16_h.val[index] = vaddq_u16(ra4_c_1_l_2_u16_h.val[index],
                                              vaddq_u16(ra4_c_1_l_1_u16_h.val[index], ra4_c_1_l_0_u16_h.val[index])); // higher part

      // move the first bit to the last position
      ra4_c_0_l_1_u16_l.val[index] = vextq_u16(ra4_c_0_l_1_u16_l.val[index], ra4_c_0_l_1_u16_l.val[index], 1);
      ra4_c_0_l_1_u16_h.val[index] = vextq_u16(ra4_c_0_l_1_u16_h.val[index], ra4_c_0_l_1_u16_h.val[index], 1);    
    }
    

    //-------------------------------------------------------------------
    
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

int blur2_do_tile_urrot1_neon_div9_u16 (int x, int y, int width, int height) {
  // TODO
  return 0;
}

int blur2_do_tile_urrot2_neon_div9_f32 (int x, int y, int width, int height) {
  // TODO
  return 0;
}

int blur2_do_tile_urrot2_neon_div9_u16 (int x, int y, int width, int height) {
  // TODO
  return 0;
}

int blur2_do_tile_urrot2_neon_div8_u16 (int x, int y, int width, int height) {
  // TODO
  return 0;
}

#endif /* __ARM_NEON__ */
