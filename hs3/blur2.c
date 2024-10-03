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

    uint16x8x4_t r_c_0_h;
    uint16x8x4_t r_c_1_h, r_c_1_l;
    uint16x8x4_t r_c_2_h, r_c_2_l; // here we need both 
    
    uint16x8x4_t r_c_2_l_0_h, r_c_2_l_0_l;
    uint16x8x4_t r_c_2_l_1_h, r_c_2_l_1_l;
    uint16x8x4_t r_c_2_l_2_h, r_c_2_l_2_l;

    uint16x8x4_t left_l, left_h;
    uint16x8x4_t right_l, right_h; 
    uint16x8x4_t sum_l, sum_h; 

    uint8x16x4_t sum;

    uint32x4x4_t sum_l_l, sum_l_h, sum_h_l, sum_h_h;

    float32x4x4_t sumf_l_l, sumf_l_h, sumf_h_l, sumf_h_h;

    // loop over x (start from +1, end at -1 => no border)
    for (int j = x + 1; j < x + width - 1; j+16) {
      uint16_t c_2_r = 0, c_2_g = 0, c_2_b = 0, c_2_a = 0; // col 2 -> 4 color components {r,g,b,a}

      //point 2
      /*uint8x16_t r_c_2_l_0 = vld1q_u8((uint8_t*)&cur_img(i - 1, j + 4));
      uint8x16_t r_c_2_l_1 = vld1q_u8((uint8_t*)&cur_img(i + 0, j + 4));
      uint8x16_t r_c_2_l_2 = (uint8x16_t)vld1q_u8((uint8_t*)&cur_img(i + 1, j + 4));*/

      // point3
      uint8x16x4_t r_c_2_l_0_4 = vld4q_u8((uint8_t*)&cur_img(i - 1, j)); // [[ r x 16 ] [ g x 16 ] [ b x 16 ] [ a x 16 ]]
      uint8x16x4_t r_c_2_l_1_4 = vld4q_u8((uint8_t*)&cur_img(i + 0, j));
      uint8x16x4_t r_c_2_l_2_4 = vld4q_u8((uint8_t*)&cur_img(i + 1, j));
      
      float32_t p = 9;
      float32x4_t r_nine = vld1q_dup_f32(&p);

      for(int index=0; index<4; index++){
        r_c_2_l_0_l.val[index] = vmovl_u8(vget_low_u8(r_c_2_l_0_4.val[index]));
        r_c_2_l_0_h.val[index] = vmovl_u8(vget_high_u8(r_c_2_l_0_4.val[index]));

        r_c_2_l_1_l.val[index] = vmovl_u8(vget_low_u8(r_c_2_l_1_4.val[index]));
        r_c_2_l_1_h.val[index] = vmovl_u8(vget_high_u8(r_c_2_l_1_4.val[index]));

        r_c_2_l_2_l.val[index] = vmovl_u8(vget_low_u8(r_c_2_l_2_4.val[index]));
        r_c_2_l_2_h.val[index] = vmovl_u8(vget_high_u8(r_c_2_l_2_4.val[index]));

        // reduction
        r_c_2_l.val[index] = vaddq_u16(r_c_2_l_2_l.val[index],vaddq_u16(r_c_2_l_1_l.val[index], r_c_2_l_0_l.val[index])); // line 0 + line 1 low part
        r_c_2_h.val[index] = vaddq_u16(r_c_2_l_2_h.val[index],vaddq_u16(r_c_2_l_1_h.val[index], r_c_2_l_0_h.val[index])); // line 0 + line 1 high part

        // 6. left-right pattern
        left_l.val[index]  = vextq_u16(r_c_0_h.val[index], r_c_1_l.val[index], 7);  // [15,0,1,2,3,4,5,6]
        right_l.val[index] = vextq_u16(r_c_1_l.val[index], r_c_1_h.val[index], 1); // [1,2,3,4,5,6,7,8]

        left_h.val[index]  = vextq_u16(r_c_1_l.val[index], r_c_1_h.val[index], 7); // [7, 8, 9, 10, 11, 12, 13, 14] 
        right_h.val[index] = vextq_u16(r_c_1_h.val[index], r_c_2_l.val[index], 1); // [9, 10, 11, 12, 13, 14, 15, 0]

        sum_l.val[index] =  vaddq_u16(vaddq_u16(left_l.val[index], r_c_1_l.val[index]), right_l.val[index]);
        sum_h.val[index] =  vaddq_u16(vaddq_u16(left_h.val[index], r_c_1_h.val[index]), right_h.val[index]);

        // 7. promotion to uint32x4_t
        sum_l_l.val[index] = vmovl_u16(vget_low_u16(sum_l.val[index]));
        sum_l_h.val[index] = vmovl_u16(vget_high_u16(sum_l.val[index]));
        sum_h_l.val[index] = vmovl_u16(vget_low_u16(sum_h.val[index]));
        sum_h_h.val[index] = vmovl_u16(vget_high_u16(sum_h.val[index]));
      
        // 7. convert to float32x4_t
        sumf_l_l.val[index] = vcvtq_n_f32_u32(sum_l_l.val[index], 10);
        sumf_l_h.val[index] = vcvtq_n_f32_u32(sum_l_h.val[index], 10);
        sumf_h_l.val[index] = vcvtq_n_f32_u32(sum_h_l.val[index], 10);
        sumf_h_h.val[index] = vcvtq_n_f32_u32(sum_h_h.val[index], 10);
        
        // 8. divison by 9
        sumf_l_l.val[index] = vdivq_f32(sumf_l_l.val[index], r_nine);
        sumf_l_h.val[index] = vdivq_f32(sumf_l_h.val[index], r_nine);
        sumf_h_l.val[index] = vdivq_f32(sumf_h_l.val[index], r_nine);
        sumf_h_h.val[index] = vdivq_f32(sumf_h_h.val[index], r_nine);

        // 9. convert back to uint32x4_t
        sum_l_l.val[index] = vcvtq_n_u32_f32(sumf_l_l.val[index], 10);
        sum_l_h.val[index] = vcvtq_n_u32_f32(sumf_l_h.val[index], 10);
        sum_h_l.val[index] = vcvtq_n_u32_f32(sumf_h_l.val[index], 10);
        sum_h_h.val[index] = vcvtq_n_u32_f32(sumf_h_h.val[index], 10);

        // 10. convert back to uint16x8_t 
        sum_l.val[index] = vcombine_u16(vqmovn_u32(sum_l_l.val[index]), vqmovn_u32(sum_l_h.val[index]));
        sum_h.val[index] = vcombine_u16(vqmovn_u32(sum_h_l.val[index]), vqmovn_u32(sum_h_h.val[index]));

        // 11. convert back to uint8x16_t
        sum.val[index] = vcombine_u8(vqmovn_u16(sum_l.val[index]), vqmovn_u16(sum_h.val[index]));
       
      }

      // 12. store
      //vst4q_u8((uint8_t*)&next_img(i, j), sum);
      
      // 13. variable rotation
      //r_c_0_h = r_c_1_h;
      //r_c_1_h = r_c_2_h;

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

//We extend the simulated division by 9 operation in Code 2 to a vectorized version 
//that can handle 128 bits (i.e. 8 16-bit integers) in one operation.
uint16x8_t urrot1_neon_div9_u16(uint16x8_t n) {
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


int blur2_do_tile_urrot1_neon_div9_u16 (int x, int y, int width, int height) {
  // TODO
  return 0;
}

int blur2_do_tile_urrot2_neon_div9_f32 (int x, int y, int width, int height) {
  for (int i = y + 1; i < y + height - 1; i++) {
        
        uint16x8_t c_0_l, c_0_h, c_1_l, c_1_h, c_2_l, c_2_h;
        uint16x8_t left_l, left_h, right_l, right_h;
        uint16x8_t sum_l, sum_h;
        uint32x4_t sum_l_l, sum_l_h, sum_h_l, sum_h_h;
        float32x4_t sumf_l_l, sumf_l_h, sumf_h_l, sumf_h_h;

        // loop over x (start from +1, end at -1 => no border)
        for (int j = x + 1; j < x + width - 1; j += 16) {
            // Use the vld1q_u8 instructions
            uint8x16x4_t r_c_2_l_0_4 = vld4q_u8((const uint8_t*)&cur_img(i - 1, j));
            uint8x16x4_t r_c_2_l_1_4 = vld4q_u8((const uint8_t*)&cur_img(i + 0, j));
            uint8x16x4_t r_c_2_l_2_4 = vld4q_u8((const uint8_t*)&cur_img(i + 1, j));

            //Promote the 8-bit components into 16-bit components
            c_2_l = vaddq_u16(vmovl_u8(vget_low_u8(r_c_2_l_0_4.val[0])),
                              vaddq_u16(vmovl_u8(vget_low_u8(r_c_2_l_1_4.val[0])),
                                        vmovl_u8(vget_low_u8(r_c_2_l_2_4.val[0]))));
            c_2_h = vaddq_u16(vmovl_u8(vget_high_u8(r_c_2_l_0_4.val[0])),
                              vaddq_u16(vmovl_u8(vget_high_u8(r_c_2_l_1_4.val[0])),
                                        vmovl_u8(vget_high_u8(r_c_2_l_2_4.val[0]))));

            // 3. left-right
            left_l = vextq_u16(c_0_l, c_1_l, 7);
            right_l = vextq_u16(c_1_l, c_2_l, 1);
            left_h = vextq_u16(c_0_h, c_1_h, 7);
            right_h = vextq_u16(c_1_h, c_2_h, 1);

            sum_l = vaddq_u16(vaddq_u16(left_l, c_1_l), right_l);
            sum_h = vaddq_u16(vaddq_u16(left_h, c_1_h), right_h);

            // 4. promote the results to uint32x4_t registers.
            sum_l_l = vmovl_u16(vget_low_u16(sum_l));
            sum_l_h = vmovl_u16(vget_high_u16(sum_l));
            sum_h_l = vmovl_u16(vget_low_u16(sum_h));
            sum_h_h = vmovl_u16(vget_high_u16(sum_h));

            sumf_l_l = vcvtq_f32_u32(sum_l_l);
            sumf_l_h = vcvtq_f32_u32(sum_l_h);
            sumf_h_l = vcvtq_f32_u32(sum_h_l);
            sumf_h_h = vcvtq_f32_u32(sum_h_h);

            // 5. Perform the division by 9 on 32-bit floating-point
            float32x4_t r_nine = vdupq_n_f32(9.0f);
            sumf_l_l = vdivq_f32(sumf_l_l, r_nine);
            sumf_l_h = vdivq_f32(sumf_l_h, r_nine);
            sumf_h_l = vdivq_f32(sumf_h_l, r_nine);
            sumf_h_h = vdivq_f32(sumf_h_h, r_nine);

            // 6. Convert back the float32x4_t resulting registers into uint32x4_t registers.
            sum_l_l = vcvtq_u32_f32(sumf_l_l);
            sum_l_h = vcvtq_u32_f32(sumf_l_h);
            sum_h_l = vcvtq_u32_f32(sumf_h_l);
            sum_h_h = vcvtq_u32_f32(sumf_h_h);

            // 7. Convert back the uint32x4_t registers into uint16x8_t registers.
            sum_l = vcombine_u16(vqmovn_u32(sum_l_l), vqmovn_u32(sum_l_h));
            sum_h = vcombine_u16(vqmovn_u32(sum_h_l), vqmovn_u32(sum_h_h));

            // 8. Convert back the uint16x8_t registers into uint8x16_t registers
            uint8x16_t result = vcombine_u8(vqmovn_u16(sum_l), vqmovn_u16(sum_h));

            // 9. store
            vst4q_u8((uint8_t*)&next_img(i, j), result);

            // 10
            c_0_l = c_1_l; c_0_h = c_1_h;
            c_1_l = c_2_l; c_1_h = c_2_h;
        }
    }

    uint32_t bsize = 1;
    compute_borders(x, y, width, height, bsize);

    return 0;
}
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
