
#include "easypap.h"

#include <omp.h>

/**
 * compiler optimization O0
 * time:  8826.458 ms
 */


///////////////////////////// Sequential version (tiled)
// Suggested cmdline(s):
// ./run -l data/img/1024.png -k blur -v seq -si
//
int blur_do_tile_default (int x, int y, int width, int height)
{
 //printf("%d - %d , %d - %d", x, y, width,height );
  for (int i = y+1; i < y + height-1; i++){
    for (int j = x+1; j < x + width-1; j++) {
      unsigned r = 0, g = 0, b = 0, a = 0, n = 0;

	for (int yloc = i-1; yloc <= i+1; yloc++){
	  for (int xloc = j-1; xloc <= j+1; xloc++) {
	      unsigned c = cur_img(yloc, xloc);
	      r += ezv_c2r (c);
	      g += ezv_c2g (c);
	      b += ezv_c2b (c);
	      a += ezv_c2a (c);
	      n += 1;
	    }
	}

	  r /= n;
	  g /= n;
	  b /= n;
	  a /= n;
	  //printf("miao\n");
	  next_img(i, j) = ezv_rgba (r, g, b, a);
      }
     }
  for(int i =0; i<height; i++){	
      next_img(i,0) = cur_img(i, 0);	
      next_img(i,DIM-1) = cur_img(i, DIM-1);	
  } 
  for(int j =0; j<width; j++){	
      next_img(0,j) = cur_img(0, j);	
      next_img(DIM-1,j) = cur_img(DIM-1,j);	
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
