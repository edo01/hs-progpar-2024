
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

    uint8x16_t r_c_0_l_0, r_c_0_l_1, r_c_0_l_2;
    uint8x16_t r_c_1_l_0, r_c_1_l_1, r_c_1_l_2;

    // loop over x (start from +1, end at -1 => no border)
    for (int j = x + 1; j < x + width - 1; j++) {
      uint16_t c_2_r = 0, c_2_g = 0, c_2_b = 0, c_2_a = 0; // col 2 -> 4 color components {r,g,b,a}

      // read 3 pixels of column 2
      unsigned c_0_2 = cur_img(i - 1, j + 1);
      unsigned c_1_2 = cur_img(i + 0, j + 1);
      unsigned c_2_2 = cur_img(i + 1, j + 1);

      uint8x16_t r_c_2_l_0 = (uint8x16_t)vld1q_u8((uint8_t*)&c_0_2);
      uint8x16_t r_c_2_l_1 = (uint8x16_t)vld1q_u8((uint8_t*)&c_1_2);
      uint8x16_t r_c_2_l_2 = (uint8x16_t)vld1q_u8((uint8_t*)&c_2_2);


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
  // TODO
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