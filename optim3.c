
#include "easypap.h"

#include <omp.h>

// we removed n, the image blur algorithm uses a 3x3 neighborhood template. 
//If it is obtained by using the variable n, each addition operation will increase the overhead.
/**
 * Optimization O0
 * DENVER 2)  default_nb time: 8826.458 ms VS optim1 time:  5180.396 ms 
 * CORTEX-A57)  default_nb time: 8826.458 (CHANGEME) ms VS optim1 time: 20361.103  ms 
 * In this version we can observe better performances thanks to the inlining.
 * Explication:
 * Each time a function is called, the program needs to save the context, jump to the function, etc.,
 * which will bring a certain amount of time overhead. 
 * Inline functions insert function code directly at the call point.
 * 
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
	r += (uint8_t)c; g += (uint8_t)(c >> 8); b += (uint8_t)(c >> 16); a += (uint8_t)(c >> 24);

	// i-1, j
	c = cur_img(i-1, j);
	r += (uint8_t)c; g += (uint8_t)(c >> 8); b += (uint8_t)(c >> 16); a += (uint8_t)(c >> 24);
		
	// i-1, j+1
	c = cur_img(i-1, j+1);
	r += (uint8_t)c; g += (uint8_t)(c >> 8); b += (uint8_t)(c >> 16); a += (uint8_t)(c >> 24);	
	
	// i, j-1
	c = cur_img(i, j-1);
	r += (uint8_t)c; g += (uint8_t)(c >> 8); b += (uint8_t)(c >> 16); a += (uint8_t)(c >> 24);

	// i, j
	c = cur_img(i, j);
	r += (uint8_t)c; g += (uint8_t)(c >> 8); b += (uint8_t)(c >> 16); a += (uint8_t)(c >> 24);
		
	// i, j+1
	c = cur_img(i, j+1);
	r += (uint8_t)c; g += (uint8_t)(c >> 8); b += (uint8_t)(c >> 16); a += (uint8_t)(c >> 24);
	
	// i+1, j-1
	c = cur_img(i+1, j-1);
	r += (uint8_t)c; g += (uint8_t)(c >> 8); b += (uint8_t)(c >> 16); a += (uint8_t)(c >> 24);

	// i+1, j
	c = cur_img(i+1, j);
	r += (uint8_t)c; g += (uint8_t)(c >> 8); b += (uint8_t)(c >> 16); a += (uint8_t)(c >> 24);
		
	// i+1, j+1
	c = cur_img(i+1, j+1);
	r += (uint8_t)c; g += (uint8_t)(c >> 8); b += (uint8_t)(c >> 16); a += (uint8_t)(c >> 24);

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
