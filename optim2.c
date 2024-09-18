
#include "easypap.h"

#include <omp.h>

// we removed n 
/**
 * Optimization O0
 * DENVER 2)  default_nb time: 8826.458 ms VS optim1 time:  6171.679 ms 
 * CORTEX-A57)  default_nb time: 8826.458 (CHANGEME) ms VS optim1 time: 25721.413 ms 
 * since we eliminated another loop, the code run faster than optim1. It is evident
 * that the arch benifts from the elimination of the branch from the assembly.
 We are accessing column major. Due to cache miss cross-row accesses 
 * in the vertical direction, we can observe that is slower than the optim1 [TO CHECK]
*/

///////////////////////////// Sequential version (tiled)
// Suggested cmdline(s):
// ./run -l data/img/1024.png -k blur -v seq -si
//
int blur_do_tile_default (int x, int y, int width, int height)
{
  for (int i = y+1; i < y + height-1; i++){
    for (int j = x+1; j < x + width-1; j++) {
      unsigned r = 0, g = 0, b = 0, a = 0;
      unsigned c;
	//for (int yloc = i-1; yloc <= i+1; yloc++){
	// i-1, j-1
	c = cur_img(i-1, j-1);
	r += ezv_c2r (c); g += ezv_c2g (c); b += ezv_c2b (c); a += ezv_c2a (c);

	// i-1, j
	c = cur_img(i-1, j);
	r += ezv_c2r (c); g += ezv_c2g (c); b += ezv_c2b (c); a += ezv_c2a (c);
		
	// i-1, j+1
	c = cur_img(i-1, j+1);
	r += ezv_c2r (c); g += ezv_c2g (c); b += ezv_c2b (c); a += ezv_c2a (c);	
	
	// i, j-1
	c = cur_img(i, j-1);
	r += ezv_c2r (c); g += ezv_c2g (c); b += ezv_c2b (c); a += ezv_c2a (c);

	// i, j
	c = cur_img(i, j);
	r += ezv_c2r (c); g += ezv_c2g (c); b += ezv_c2b (c); a += ezv_c2a (c);
		
	// i, j+1
	c = cur_img(i, j+1);
	r += ezv_c2r (c); g += ezv_c2g (c); b += ezv_c2b (c); a += ezv_c2a (c);
	
	// i+1, j-1
	c = cur_img(i+1, j-1);
	r += ezv_c2r (c); g += ezv_c2g (c); b += ezv_c2b (c); a += ezv_c2a (c);

	// i+1, j
	c = cur_img(i+1, j);
	r += ezv_c2r (c); g += ezv_c2g (c); b += ezv_c2b (c); a += ezv_c2a (c);
		
	// i+1, j+1
	c = cur_img(i+1, j+1);
	r += ezv_c2r (c); g += ezv_c2g (c); b += ezv_c2b (c); a += ezv_c2a (c);

	  r /= 9;
	  g /= 9;
	  b /= 9;
	  a /= 9;
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
