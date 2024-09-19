
#include "easypap.h"

#include <omp.h>

// Sequential version - no optimization
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
/*
* Optimization carried out:
* - Remove if conditions inside the loop
* - Remove variable declarations inside the loop
* - Remove computation of n
*
* By removing the borders, we can remove the if conditions that check if the pixel is at the edge of the image.
* In this, we can remove the if conditions that check if the pixel is at the edge of the image, reducing the 
* number of branches in the code. 
* 
* Especially in the Cortex-A57, the branch prediction is only two levels deep, and the branch prediction is not
* accurate when there are too many branch is nested. For the Cortex-A57, a branch misprediction will cost 
* 16-19 cycles, which is a significant overhead considering the number of pixels in the image.
* 
* 
* MISSING RESULTS!!!
*/
int blur_do_tile_default_nb (int x, int y, int width, int height)
{
  // moving variables declaration outside the loop
  unsigned r = 0, g = 0, b = 0, a = 0;
  unsigned c;

  /* 
    * We remove the borders by starting from y+1 and x+1 and 
    * ending at y+height-1 and x+width-1 
  */
  for (int i = y+1; i < y + height-1; i++){
    for (int j = x+1; j < x + width-1; j++){
      r = 0, g = 0, b = 0, a = 0;

      for (int yloc = i-1; yloc <= i+1; yloc++){
        for (int xloc = j-1; xloc <= j+1; xloc++) {
            c = cur_img(yloc, xloc);
            r += ezv_c2r (c); g += ezv_c2g (c); 
            b += ezv_c2b (c); a += ezv_c2a (c);
          }
      }

      // take the average
      r /= 9: g /= 9; b /= 9; a /= 9;
      next_img(i, j) = ezv_rgba (r, g, b, a);
    }
  }

  // Deal the borders
  for(int i = y; i < height; i++){	
      next_img(i, x) = cur_img(i, x);	
      next_img(i, DIM-1) = cur_img(i, DIM-1);	
  } 

  // Attention NOT COALESCED if easypap store the image in row-major (not cache friendly access)
  for(int j = x; j < width; j++){	
      next_img(y, j) = cur_img(y, j);	
      next_img(DIM-1,j) = cur_img(DIM-1,j);	
  } 

  return 0;
}

// optim1 - LOOP UNROLLING ONLY X DIRECTION
/*
 * Optimization carried out:
 * - Loop unrolling in the x direction
 * 
 * In this version, we unrolled the loop in the x direction. Unrolling the loop produces 
 * more instructions, but it also reduces the number of jumps in the code. Note that
 * we don't need to add an epilogue since the loop is completely unrolled.
 *
 * MISSING RESULTS!!!
 */
/**
 * Optimization O0
 * DENVER 2)  default_nb time: 8826.458 ms VS optim1 time: 6932.951 ms 
 * CORTEX-A57)  default_nb time: 8826.458 ms VS optim1 time: 25708.885 ms 
 * We are accessing row major (in every version we are doing like that)
 */
int blur_do_tile_optim1 (int x, int y, int width, int height)
{
  unsigned r = 0, g = 0, b = 0, a = 0;
  unsigned c;

  for (int i = y+1; i < y + height-1; i++){
    for (int j = x+1; j < x + width-1; j++) {
      r = 0, g = 0, b = 0, a = 0;

	    for (int yloc = i-1; yloc <= i+1; yloc++){
	      c = cur_img(yloc, j-1);
	      r += ezv_c2r (c); g += ezv_c2g (c); b += ezv_c2b (c); a += ezv_c2a (c);
	    
	      c = cur_img(yloc, j);
	      r += ezv_c2r (c); g += ezv_c2g (c); b += ezv_c2b (c); a += ezv_c2a (c);

	      c = cur_img(yloc, j+1);
	      r += ezv_c2r (c); g += ezv_c2g (c); b += ezv_c2b (c); a += ezv_c2a (c);
	    }

      // no need for an epilogue since the loop is completely unrolled
	    r /= 9; g /= 9; b /= 9; a /= 9;
    
	    next_img(i, j) = ezv_rgba (r, g, b, a);
    }
  }

  // Deal the borders
  for(int i = y; i < height; i++){	
      next_img(i, x) = cur_img(i, x);	
      next_img(i, DIM-1) = cur_img(i, DIM-1);	
  } 

  // Attention NOT COALESCED if easypap store the image in row-major (not cache friendly access)
  for(int j = x; j < width; j++){	
      next_img(y, j) = cur_img(y, j);	
      next_img(DIM-1,j) = cur_img(DIM-1,j);	
  } 

  return 0;
}


// optim2 - LOOP UNROLLING X AND Y DIRECTION
/*
 * Optimization carried out:
 * - Loop unrolling in the y direction
 * 
 * In this version, we unrolled the loop in the y direction.
 * We can observe that the code is faster than optim1 since we eliminated 
 * completely the loop in the y direction. In this way, we have 
 * removed completely the branches inside the loop, leaving only two
 * loops in the code. Theoretically, the code on the Cortex-A57 should
 * perform better since the branch prediction is only two levels deep.
 *
 * MISSING RESULTS!!!
 */
/**
 * Optimization O0
 * DENVER 2)  default_nb time: 8826.458 ms VS optim1 time:  6171.679 ms 
 * CORTEX-A57)  default_nb time: 8826.458 (CHANGEME) ms VS optim1 time: 25721.413 ms 
*/
int blur_do_tile_optim2 (int x, int y, int width, int height)
{
  unsigned r = 0, g = 0, b = 0, a = 0;
  unsigned c;

  for (int i = y+1; i < y + height-1; i++){
    for (int j = x+1; j < x + width-1; j++) {
      r = 0, g = 0, b = 0, a = 0;
      
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

      r /= 9; g /= 9; b /= 9; a /= 9;
      
      next_img(i, j) = ezv_rgba (r, g, b, a);
    }
  }

  for(int i = x; i<height; i++){	
      next_img(i,x) = cur_img(i, x);	
      next_img(i,DIM-1) = cur_img(i, DIM-1);	
  } 
  for(int j =0; j<width; j++){	
      next_img(y,j) = cur_img(y, j);	
      next_img(DIM-1,j) = cur_img(DIM-1,j);	
  } 
  return 0;
}

// optim3 - INLINE FUNCTION CALLS
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
      next_img(i, j) = ezv_rgba (r, g, b, a);
    }
  }
  for(int i = x ; i<height; i++){	
      next_img(i,x) = cur_img(i, x);	
      next_img(i,DIM-1) = cur_img(i, DIM-1);	
  } 
  for(int j = y; j<width; j++){	
      next_img(y,j) = cur_img(y, j);	
      next_img(DIM-1,j) = cur_img(DIM-1,j);	
  } 
  return 0;
}

// optim4 - VARIABLES ROTATION
/**
 * Optimization O0
 * DENVER 2)  default_nb time: 8826.458(CHANGEME) ms VS optim1 time:  2547.382(CHANGEME) ms 
 * CORTEX-A57)  default_nb time: 8826.458 (CHANGEME) ms VS optim1 time: 8651.694(CHANGEME)  ms 
 * Explication:
 * In optim3, every time a pixel is processed, all pixels in its 3x3 neighborhood are revisited, 
   and for two adjacent pixels, they partially overlap.
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
    c = cur_img(i-1, x);
    r0 += (uint8_t)c; g0 += (uint8_t)(c >> 8); b0 += (uint8_t)(c >> 16); a0 += (uint8_t)(c >> 24); 

    c = cur_img(i-1, x+1);
    r1 += (uint8_t)c; g1 += (uint8_t)(c >> 8); b1 += (uint8_t)(c >> 16); a1 += (uint8_t)(c >> 24);

    c = cur_img(i-1, x+2);
    r2 += (uint8_t)c; g2 += (uint8_t)(c >> 8); b2 += (uint8_t)(c >> 16); a2 += (uint8_t)(c >> 24);

    c = cur_img(i, x);
    r0 += (uint8_t)c; g0 += (uint8_t)(c >> 8); b0 += (uint8_t)(c >> 16); a0 += (uint8_t)(c >> 24);

    c = cur_img(i, x+1);
    r1 += (uint8_t)c; g1 += (uint8_t)(c >> 8); b1 += (uint8_t)(c >> 16); a1 += (uint8_t)(c >> 24);

    c = cur_img(i, x+2);
    r2 += (uint8_t)c; g2 += (uint8_t)(c >> 8); b2 += (uint8_t)(c >> 16); a2 += (uint8_t)(c >> 24);

    c = cur_img(i+1, x);
    r0 += (uint8_t)c; g0 += (uint8_t)(c >> 8); b0 += (uint8_t)(c >> 16); a0 += (uint8_t)(c >> 24);

    c = cur_img(i+1, x+1);
    r1 += (uint8_t)c; g1 += (uint8_t)(c >> 8); b1 += (uint8_t)(c >> 16); a1 += (uint8_t)(c >> 24);

    c = cur_img(i+1, x+2);
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
      r2 = (uint8_t)c; g2 = (uint8_t)(c >> 8); b2 = (uint8_t)(c >> 16) & 0xff; a2 = (uint8_t)(c >> 24);

      c = cur_img(i, j + 2);
      r2 += (uint8_t)c; g2 += (uint8_t)(c >> 8); b2 += (uint8_t)(c >> 16) & 0xff; a2 += (uint8_t)(c >> 24);

      c = cur_img(i + 1, j + 2);
      r2 += (uint8_t)c; g2 += (uint8_t)(c >> 8); b2 += (uint8_t)(c >> 16) & 0xff; a2 += (uint8_t)(c >> 24);
    }
  }

  for(int i =x; i<height; i++){	
      next_img(i,x) = cur_img(i, x);	
      next_img(i,DIM-1) = cur_img(i, DIM-1);	
  } 
  for(int j =y; j<width; j++){	
      next_img(y,j) = cur_img(y, j);	
      next_img(DIM-1,j) = cur_img(DIM-1,j);	
  }
  return 0;
}


/**
 * Optimization O0
 * DENVER 2)  default_nb time: 8826.458(CHANGEME) ms VS optim5 time: 5802.546(CHANGEME) ms 
 * CORTEX-A57)  default_nb time: 8826.458 (CHANGEME) ms VS optim5 time: 21753.905(CHANGEME)  ms 
 * For pixels at the edge of the image, we use the logic of blur_do_tile_default. For non-border pixels, 
 we keep the optimization method of optim4 and use variable rotation technology to reduce redundant calculations.
 * Explication:
 * 
 * 
 * 

 * 
 * 
*/
int blur_do_tile_optim5(int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++) {
    for (int j = x; j < x + width; j++) {

      // Handle the edge case as in default
      if (i == 0 || j == 0 || i == DIM - 1 || j == DIM - 1) {
        unsigned r = 0, g = 0, b = 0, a = 0, n = 0;
        int i_d = (i > 0) ? i - 1 : i;
        int i_f = (i < DIM - 1) ? i + 1 : i;
        int j_d = (j > 0) ? j - 1 : j;
        int j_f = (j < DIM - 1) ? j + 1 : j;

        // Loop through neighboring pixels for boundary
        for (int yloc = i_d; yloc <= i_f; yloc++) {
          for (int xloc = j_d; xloc <= j_f; xloc++) {
            unsigned c = cur_img(yloc, xloc);
            r += ezv_c2r(c);
            g += ezv_c2g(c);
            b += ezv_c2b(c);
            a += ezv_c2a(c);
            n++;
          }
        }
        r /= n;
        g /= n;
        b /= n;
        a /= n;
        next_img(i, j) = ezv_rgba(r, g, b, a);
      } else {
        // Non-edge pixels: optimized logic from optim4 
        unsigned r0 = 0, g0 = 0, b0 = 0, a0 = 0;
        unsigned r1 = 0, g1 = 0, b1 = 0, a1 = 0;
        unsigned r2 = 0, g2 = 0, b2 = 0, a2 = 0;

        unsigned c;
        c = cur_img(i - 1, j - 1);
        r0 += (uint8_t)c; g0 += (uint8_t)(c >> 8); b0 += (uint8_t)(c >> 16); a0 += (uint8_t)(c >> 24);
        c = cur_img(i - 1, j);
        r1 += (uint8_t)c; g1 += (uint8_t)(c >> 8); b1 += (uint8_t)(c >> 16); a1 += (uint8_t)(c >> 24);
        c = cur_img(i - 1, j + 1);
        r2 += (uint8_t)c; g2 += (uint8_t)(c >> 8); b2 += (uint8_t)(c >> 16); a2 += (uint8_t)(c >> 24);

        c = cur_img(i, j - 1);
        r0 += (uint8_t)c; g0 += (uint8_t)(c >> 8); b0 += (uint8_t)(c >> 16); a0 += (uint8_t)(c >> 24);
        c = cur_img(i, j);
        r1 += (uint8_t)c; g1 += (uint8_t)(c >> 8); b1 += (uint8_t)(c >> 16); a1 += (uint8_t)(c >> 24);
        c = cur_img(i, j + 1);
        r2 += (uint8_t)c; g2 += (uint8_t)(c >> 8); b2 += (uint8_t)(c >> 16); a2 += (uint8_t)(c >> 24);

        c = cur_img(i + 1, j - 1);
        r0 += (uint8_t)c; g0 += (uint8_t)(c >> 8); b0 += (uint8_t)(c >> 16); a0 += (uint8_t)(c >> 24);
        c = cur_img(i + 1, j);
        r1 += (uint8_t)c; g1 += (uint8_t)(c >> 8); b1 += (uint8_t)(c >> 16); a1 += (uint8_t)(c >> 24);
        c = cur_img(i + 1, j + 1);
        r2 += (uint8_t)c; g2 += (uint8_t)(c >> 8); b2 += (uint8_t)(c >> 16); a2 += (uint8_t)(c >> 24);
        
        unsigned r = (r0 + r1 + r2) / 9;
        unsigned g = (g0 + g1 + g2) / 9;
        unsigned b = (b0 + b1 + b2) / 9;
        unsigned a = (a0 + a1 + a2) / 9;

        next_img(i, j) = ezv_rgba(r, g, b, a);
      }
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