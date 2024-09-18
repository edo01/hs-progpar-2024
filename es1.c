
#include "easypap.h"

#include <omp.h>

/**
 * Compiler optimization O0
 *
 * Denver 2 time: 9139.340 ms
 * Cortex-A57 time: 29570.333 ms
 *
 *
 * Compiler optimization O1
 *
 * Denver 2 time: 3203.847 ms
 * Cortex-A57 time: 3863.578 ms
 *
 * Compiler optimization O2
 *
 * Denver 2 time: 4003.764 ms
 * Cortex-A57 time: 3378.642 ms
 *
 *
 * Compiler optimization O3
 *
 * Denver 2 time optimization: 2867.745 ms
 * Cortex-A57 time optimization: 3888.405 ms
 *
 */

///////////////////////////// Sequential version (tiled)
// Suggested cmdline(s):
// ./run -l data/img/1024.png -k blur -v seq -si
//
int blur_do_tile_default (int x, int y, int width, int height)
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
