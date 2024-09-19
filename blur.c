
#include "easypap.h"

#include <omp.h>

// Sequential version
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

// default_nb - REMOVE BORDERS
/**
 * compiler optimization O0
 * time:  8826.458 ms
 */
int blur_do_tile_default_nb (int x, int y, int width, int height)
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

// optim1 - LOOP UNROLLING ONLY X DIRECTION
// we removed n 
/**
 * Optimization O0
 * DENVER 2)  default_nb time: 8826.458 ms VS optim1 time: 6932.951 ms 
 * CORTEX-A57)  default_nb time: 8826.458 ms VS optim1 time: 25708.885 ms 
 * We are accessing row major (in every version we are doing like that)
 */
int blur_do_tile_optim1 (int x, int y, int width, int height)
{
  for (int i = y+1; i < y + height-1; i++){
    for (int j = x+1; j < x + width-1; j++) {
      unsigned r = 0, g = 0, b = 0, a = 0, n = 0;

	for (int yloc = i-1; yloc <= i+1; yloc++){
	  //for (int xloc = j-1; xloc <= j+1; xloc++) {
	      unsigned c = cur_img(yloc, j-1);
	      r += ezv_c2r (c); g += ezv_c2g (c); b += ezv_c2b (c); a += ezv_c2a (c);
	      //n += 1;
	    
	      c = cur_img(yloc, j);
	      r += ezv_c2r (c); g += ezv_c2g (c); b += ezv_c2b (c); a += ezv_c2a (c);
	      //n += 1;

	      c = cur_img(yloc, j+1);
	      r += ezv_c2r (c); g += ezv_c2g (c); b += ezv_c2b (c); a += ezv_c2a (c);
	}

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


// optim2 - LOOP UNROLLING X AND Y DIRECTION
/**
 * Optimization O0
 * DENVER 2)  default_nb time: 8826.458 ms VS optim1 time:  6171.679 ms 
 * CORTEX-A57)  default_nb time: 8826.458 (CHANGEME) ms VS optim1 time: 25721.413 ms 
 * since we eliminated another loop, the code run faster than optim1. It is evident
 * that the arch benifts from the elimination of the branch from the assembly.
 We are accessing column major. Due to cache miss cross-row accesses 
 * in the vertical direction, we can observe that is slower than the optim1 [TO CHECK]
*/
int blur_do_tile_loopxy (int x, int y, int width, int height)
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

// optim3 - INLINE FUNCTION CALLS
/**
 * Optimization O0
 * DENVER 2)  default_nb time: 8826.458 ms VS optim1 time:  5180.396 ms 
 * CORTEX-A57)  default_nb time: 8826.458 (CHANGEME) ms VS optim1 time: 20361.103  ms 
 * since we eliminated another loop, the code run faster than optim1. It is evident
 * that the arch benifts from the elimination of the branch from the assembly.
 We are accessing column major. Due to cache miss cross-row accesses 
 * in the vertical direction, we can observe that is slower than the optim1 [TO CHECK]

 * In this version we can observe better performances thanks to the inlining.
 * 
*/
int blur_do_tile_optim3 (int x, int y, int width, int height)
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

// we removed n 
/**
 * Optimization O0
 * DENVER 2)  default_nb time: 8826.458(CHANGEME) ms VS optim1 time:  5180.396(CHANGEME) ms 
 * CORTEX-A57)  default_nb time: 8826.458 (CHANGEME) ms VS optim1 time: 20361.103(CHANGEME)  ms 
 * Explication:
 * In optim3, every time a pixel is processed, all pixels in its 3x3 neighborhood are revisited, 
   and for two adjacent pixels, they partially overlap.
 * 
 * 

 * 
 * 
*/
int blur_do_tile_optim4(int x, int y, int width, int height)
{
  for (int i = y+1; i < y + height-1; i++) {

    // compute for the first column
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
      // Compute the average 
      unsigned r = (r0 + r1 + r2) / 9;
      unsigned g = (g0 + g1 + g2) / 9;
      unsigned b = (b0 + b1 + b2) / 9;
      unsigned a = (a0 + a1 + a2) / 9;

      next_img(i, j) = ezv_rgba(r, g, b, a);

      // Shift the columns to the left 
      r0 = r1; g0 = g1; b0 = b1; a0 = a1;
      r1 = r2; g1 = g2; b1 = b2; a1 = a2;
      // Compute values for the next column 
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
