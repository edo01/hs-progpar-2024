
#include "easypap.h"

#include <omp.h>


/**
 * Optimization O0
 * DENVER 2)  default_nb time: 8826.458(CHANGEME) ms VS optim1 time:  5180.396(CHANGEME) ms 
 * CORTEX-A57)  default_nb time: 8826.458 (CHANGEME) ms VS optim1 time: 20361.103(CHANGEME)  ms 
 * Explication:
 * 
 * 
 * 

 * 
 * 
*/

///////////////////////////// Sequential version (tiled)
// Suggested cmdline(s):
// ./run -l data/img/1024.png -k blur -v seq -si
//
int blur_do_tile_optim4(int x, int y, int width, int height)
{
  for (int i = y+1; i < y + height-1; i++) {

    // Precompute for the first column
    unsigned r0 = 0, g0 = 0, b0 = 0, a0 = 0;
    unsigned r1 = 0, g1 = 0, b1 = 0, a1 = 0;
    unsigned r2 = 0, g2 = 0, b2 = 0, a2 = 0;

    unsigned c;
    c = cur_img(i-1, x-1);
    r0 += (uint8_t)c; g0 += (uint8_t)(c >> 8); b0 += (uint8_t)(c >> 16); a0 += (uint8_t)(c >> 24);

    c = cur_img(i-1, x);
    r1 += (uint8_t)c; g1 += (uint8_t)(c >> 8); b1 += (uint8_t)(c >> 16); a1 += (uint8_t)(c >> 24);

    c = cur_img(i-1, x+1);
    r2 += (uint8_t)c; g2 += (uint8_t)(c >> 8); b2 += (uint8_t)(c >> 16); a2 += (uint8_t)(c >> 24);

    c = cur_img(i, x-1);
    r0 += (uint8_t)c; g0 += (uint8_t)(c >> 8); b0 += (uint8_t)(c >> 16); a0 += (uint8_t)(c >> 24);

    c = cur_img(i, x);
    r1 += (uint8_t)c; g1 += (uint8_t)(c >> 8); b1 += (uint8_t)(c >> 16); a1 += (uint8_t)(c >> 24);

    c = cur_img(i, x+1);
    r2 += (uint8_t)c; g2 += (uint8_t)(c >> 8); b2 += (uint8_t)(c >> 16); a2 += (uint8_t)(c >> 24);

    c = cur_img(i+1, x-1);
    r0 += (uint8_t)c; g0 += (uint8_t)(c >> 8); b0 += (uint8_t)(c >> 16); a0 += (uint8_t)(c >> 24);

    c = cur_img(i+1, x);
    r1 += (uint8_t)c; g1 += (uint8_t)(c >> 8); b1 += (uint8_t)(c >> 16); a1 += (uint8_t)(c >> 24);

    c = cur_img(i+1, x+1);
    r2 += (uint8_t)c; g2 += (uint8_t)(c >> 8); b2 += (uint8_t)(c >> 16); a2 += (uint8_t)(c >> 24);

    for (int j = x + 1; j < x + width - 1; j++) {
      // Compute the average for the current pixel
      unsigned r = (r0 + r1 + r2) / 9;
      unsigned g = (g0 + g1 + g2) / 9;
      unsigned b = (b0 + b1 + b2) / 9;
      unsigned a = (a0 + a1 + a2) / 9;

      next_img(i, j) = ezv_rgba(r, g, b, a);

      // Shift the columns to the left 
      r0 = r1; g0 = g1; b0 = b1; a0 = a1;
      r1 = r2; g1 = g2; b1 = b2; a1 = a2;
      // Compute values for the next column (right)
      c = cur_img(i - 1, j + 2);
      r2 = (c >> 24) & 0xff; g2 = (c >> 16) & 0xff; b2 = (c >> 8) & 0xff; a2 = c & 0xff;

      c = cur_img(i, j + 2);
      r2 += (c >> 24) & 0xff; g2 += (c >> 16) & 0xff; b2 += (c >> 8) & 0xff; a2 += c & 0xff;

      c = cur_img(i + 1, j + 2);
      r2 += (c >> 24) & 0xff; g2 += (c >> 16) & 0xff; b2 += (c >> 8) & 0xff; a2 += c & 0xff;
    }
  }
  return 0;
}


///////////////////////////// Sequential version (tiled)
// Suggested cmdline(s):
// ./run -l data/img/1024.png -k blur -v seq
//
unsigned blur_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    do_tile (0, 0, DIM, DIM);

    swap_images ();
  }

  return 0;
}

///////////////////////////// Tiled sequential version (tiled)
// Suggested cmdline(s):
// ./run -l data/img/1024.png -k blur -v tiled -ts 32 -m si
//
unsigned blur_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H);

    swap_images ();
  }

  return 0;
}
