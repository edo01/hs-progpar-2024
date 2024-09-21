
#include "easypap.h"

#include <omp.h>

// default - no optimization
/**
 * #############  O0  ###############
 * D2: 9698.822   ms
 * CA57: 29565.17 ms
 * #############  O1  ###############
 * D2: 3701.235   ms
 * CA57: 3743.223 ms
 * #############  O2  ###############
 * D2: 4266.211   ms
 * CA57: 3332.947 ms
 * #############  O3  ###############
 * D2: 3181.235   ms
 * CA57: 3802.015 ms
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
* #############  O0  ###############
* D2: 16294.070   ms
* CA57: 28222.70  ms
* #############  O1  ###############
* D2: 3649.656    ms
* CA57: 3605.934  ms
* #############  O2  ###############
* D2: 2836.506    ms
* CA57: 3303.200  ms
* #############  O3  ###############
* D2: 887.191     ms
* CA57: 1169.399  ms
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
      r /= 9; g /= 9; b /= 9; a /= 9;
      next_img(i, j) = ezv_rgba (r, g, b, a);
    }
  }

  // Manage the borders
  // Attention NOT COALESCED if easypap stores the image in row-major (not cache friendly access)
  for(int i = y; i < y + height; i++){	
      next_img(i, x) = cur_img(i, x);	
      next_img(i, DIM-1) = cur_img(i, DIM-1);	
  } 

  // coalesced access
  for(int j = x; j < x + width; j++){	
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
 * #############  O0  ###############
 * D2: 25624.676  ms
 * CA57: 25442.64 ms
 * #############  O1  ###############
 * D2: 2159.928   ms
 * CA57: 3061.945 ms
 * #############  O2  ###############
 * D2: 2190.614   ms
 * CA57: 2839.399 ms
 * #############  O3  ###############
 * D2: 860.914    ms
 * CA57: 1193.382 ms
 *
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

  // Manage the borders
  // Attention NOT COALESCED if easypap stores the image in row-major (not cache friendly access)
  for(int i = y; i < y + height; i++){	
      next_img(i, x) = cur_img(i, x);	
      next_img(i, DIM-1) = cur_img(i, DIM-1);	
  } 

  // coalesced access
  for(int j = x; j < x + width; j++){	
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
 * loops in the code. Theoretically, when running the code on the Cortex-A57 
 * it should perform better since the branch prediction is only two levels deep. 
 * This doesn't happen, and it is due to the presence of function calls inside the loop that
 * still limit the performance of the code.
 * 
 * #############  O0  ###############
 * D2: 12694.874  ms
 * CA57: 25197.32 ms
 * #############  O1  ###############
 * D2: 1474.978   ms
 * CA57: 1952.566 ms
 * #############  O2  ###############
 * D2: 1724.298   ms
 * CA57: 1644.741 ms
 * #############  O3  ###############
 * D2: 893.004    ms
 * CA57: 1170.463 ms
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

  // Manage the borders
  // Attention NOT COALESCED if easypap stores the image in row-major (not cache friendly access)
  for(int i = y; i < y + height; i++){	
      next_img(i, x) = cur_img(i, x);	
      next_img(i, DIM-1) = cur_img(i, DIM-1);	
  } 

  // coalesced access
  for(int j = x; j < x + width; j++){	
      next_img(y, j) = cur_img(y, j);	
      next_img(DIM-1,j) = cur_img(DIM-1,j);	
  } 
  
  return 0;
}

// optim3 - INLINE FUNCTION CALLS
/** 
 * Optimization carried out:
 * - Inline functions inside the loop
 * 
 * In this version, we inline the function calls inside the loop.
 * Each time a function is called, the program needs to save the context, jump and return from the function, etc.,
 * This will bring a certain amount of time overhead, that can be avoided by inlining the function calls.
 * 
 * Here we can see a significant performance when running the code on the Cortex-A57. Removing the function calls
 * in
 *
 * #############  O0  ###############
 * D2: 11670.198  ms
 * CA57: 20069.54 ms
 * #############  O1  ###############
 * D2: 1475.827   ms
 * CA57: 1951.636 ms
 * #############  O2  ###############
 * D2: 1208.733   ms
 * CA57: 1599.101 ms
 * #############  O3  ###############
 * D2: 883.137    ms
 * CA57: 1188.670 ms
*/
int blur_do_tile_optim3 (int x, int y, int width, int height)
{
  unsigned r = 0, g = 0, b = 0, a = 0;
  unsigned c;
  for (int i = y+1; i < y + height-1; i++){
    for (int j = x+1; j < x + width-1; j++) {
      r = 0, g = 0, b = 0, a = 0;

      /**
       * We mantained the same order of the access to the pixels as in the previous version
       * in order to access coalesced memory in the x-direction.
       */

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

  // Manage the borders
  // Attention NOT COALESCED if easypap stores the image in row-major (not cache friendly access)
  for(int i = y; i < y + height; i++){	
      next_img(i, x) = cur_img(i, x);	
      next_img(i, DIM-1) = cur_img(i, DIM-1);	
  } 

  // coalesced access
  for(int j = x; j < x + width; j++){	
      next_img(y, j) = cur_img(y, j);	
      next_img(DIM-1,j) = cur_img(DIM-1,j);	
  } 
  
  return 0;
}

// optim4 - VARIABLES ROTATION
/**
 * Optimization carried out:
 * - Variables rotation
 * 
 * In this version, we rotate the variables in the x direction.
 * Rotating the variables allows us to reduce the number of redundant calculations and memory accesses.
 * In our case, when the kernel slides to the right, the first and the seocond column of the 3x3 neighborhood
 * overlap with the second and the third column of the previous 3x3 neighborhood.
 * 
 * Please note that the memory accesses are non-coalesced when considering the same iteration of the loop in the
 * y direction. But, when the kernel slides to the right, data are likely to be in the cache due to the 
 * previous access on the same row. 
 */
/**
 * #############  O0  ###############
 * D2: 5482.415  ms
 * CA57: 8605.900  ms
 * #############  O1  ###############
 * D2: 786.345  ms
 * CA57: 1328.467  ms
 * #############  O2  ###############
 * D2: 764.912  ms
 * CA57: 1028.392  ms
 * #############  O3  ###############
 * D2: 750.870  ms
 * CA57: 1139.087  ms
 * 
*/
int blur_do_tile_optim4(int x, int y, int width, int height)
{

  unsigned r0 = 0, g0 = 0, b0 = 0, a0 = 0;
  unsigned r1 = 0, g1 = 0, b1 = 0, a1 = 0;
  unsigned r2 = 0, g2 = 0, b2 = 0, a2 = 0;
  unsigned r, g, b, a;

  unsigned c;

  for (int i = y+1; i < y + height-1; i++) {
    // compute for the first pixel
    r0 = 0, g0 = 0, b0 = 0, a0 = 0;
    r1 = 0, g1 = 0, b1 = 0, a1 = 0;
    r2 = 0, g2 = 0, b2 = 0, a2 = 0;

    c = cur_img(i-1, y);
    r0 += (uint8_t)c; g0 += (uint8_t)(c >> 8); b0 += (uint8_t)(c >> 16); a0 += (uint8_t)(c >> 24); 

    c = cur_img(i-1, y+1);
    r1 += (uint8_t)c; g1 += (uint8_t)(c >> 8); b1 += (uint8_t)(c >> 16); a1 += (uint8_t)(c >> 24);

    c = cur_img(i-1, y+2);
    r2 += (uint8_t)c; g2 += (uint8_t)(c >> 8); b2 += (uint8_t)(c >> 16); a2 += (uint8_t)(c >> 24);

    c = cur_img(i, y);
    r0 += (uint8_t)c; g0 += (uint8_t)(c >> 8); b0 += (uint8_t)(c >> 16); a0 += (uint8_t)(c >> 24);

    c = cur_img(i, y+1);
    r1 += (uint8_t)c; g1 += (uint8_t)(c >> 8); b1 += (uint8_t)(c >> 16); a1 += (uint8_t)(c >> 24);

    c = cur_img(i, y+2);
    r2 += (uint8_t)c; g2 += (uint8_t)(c >> 8); b2 += (uint8_t)(c >> 16); a2 += (uint8_t)(c >> 24);

    c = cur_img(i+1, y);
    r0 += (uint8_t)c; g0 += (uint8_t)(c >> 8); b0 += (uint8_t)(c >> 16); a0 += (uint8_t)(c >> 24);

    c = cur_img(i+1, x+1);
    r1 += (uint8_t)c; g1 += (uint8_t)(c >> 8); b1 += (uint8_t)(c >> 16); a1 += (uint8_t)(c >> 24);

    c = cur_img(i+1, x+2);
    r2 += (uint8_t)c; g2 += (uint8_t)(c >> 8); b2 += (uint8_t)(c >> 16); a2 += (uint8_t)(c >> 24);

    for (int j = x + 1; j < x + width - 1; j++) {
      // Compute the average 
      r = (r0 + r1 + r2) / 9;
      g = (g0 + g1 + g2) / 9;
      b = (b0 + b1 + b2) / 9;
      a = (a0 + a1 + a2) / 9;

      next_img(i, j) = ezv_rgba(r, g, b, a);

      // Shift the columns to the left 
      r0 = r1; g0 = g1; b0 = b1; a0 = a1;
      r1 = r2; g1 = g2; b1 = b2; a1 = a2;
      // Compute values for the next column 
      c = cur_img(i - 1, j + 2);
      r2 = (uint8_t)c; g2 = (uint8_t)(c >> 8); b2 = (uint8_t)(c >> 16); a2 = (uint8_t)(c >> 24);

      c = cur_img(i, j + 2);
      r2 += (uint8_t)c; g2 += (uint8_t)(c >> 8); b2 += (uint8_t)(c >> 16); a2 += (uint8_t)(c >> 24);

      c = cur_img(i + 1, j + 2);
      r2 += (uint8_t)c; g2 += (uint8_t)(c >> 8); b2 += (uint8_t)(c >> 16); a2 += (uint8_t)(c >> 24);
    }
  }

  // Manage the borders
  // Attention NOT COALESCED if easypap stores the image in row-major (not cache friendly access)
  for(int i = y; i < y + height; i++){	
      next_img(i, x) = cur_img(i, x);	
      next_img(i, DIM-1) = cur_img(i, DIM-1);	
  } 

  // coalesced access
  for(int j = x; j < x + width; j++){	
      next_img(y, j) = cur_img(y, j);	
      next_img(DIM-1,j) = cur_img(DIM-1,j);	
  } 
  return 0;
}


// optim5 - VARIABLES ROTATION with border management
/**
 * In this version we restore the border management in the loop
 * in order to have a complete version of the code.
 * 
 * #############  O0  ###############
 * D2: 2591.172  ms
 * CA52: 8666.370  ms
 * #############  O1  ###############
 * D2: 893.937  ms
 * CA52: 1329.319  ms
 * #############  O2  ###############
 * D2: 802.893  ms
 * CA52: 1095.065  ms
 * #############  O3  ###############
 * D2: 755.031  ms
 * CA52: 1146.858  ms
 */
 
int blur_do_tile_optim5(int x, int y, int width, int height)
{

  unsigned r0 = 0, g0 = 0, b0 = 0, a0 = 0;
  unsigned r1 = 0, g1 = 0, b1 = 0, a1 = 0;
  unsigned r2 = 0, g2 = 0, b2 = 0, a2 = 0;
  unsigned r = 0, g = 0, b = 0, a = 0;

  unsigned c;
  int j_d, j_f, n = 0;

  for (int i = y+1; i < y + height-1; i++) {
    // compute for the first pixel

    c = cur_img(i-1, y);
    r0 = (uint8_t)c; g0 = (uint8_t)(c >> 8); b0 = (uint8_t)(c >> 16); a0 = (uint8_t)(c >> 24); 

    c = cur_img(i-1, y+1);
    r1 = (uint8_t)c; g1 = (uint8_t)(c >> 8); b1 = (uint8_t)(c >> 16); a1 = (uint8_t)(c >> 24);

    c = cur_img(i-1, y+2);
    r2 = (uint8_t)c; g2 = (uint8_t)(c >> 8); b2 = (uint8_t)(c >> 16); a2 = (uint8_t)(c >> 24);

    c = cur_img(i, y);
    r0 += (uint8_t)c; g0 += (uint8_t)(c >> 8); b0 += (uint8_t)(c >> 16); a0 += (uint8_t)(c >> 24);

    c = cur_img(i, y+1);
    r1 += (uint8_t)c; g1 += (uint8_t)(c >> 8); b1 += (uint8_t)(c >> 16); a1 += (uint8_t)(c >> 24);

    c = cur_img(i, y+2);
    r2 += (uint8_t)c; g2 += (uint8_t)(c >> 8); b2 += (uint8_t)(c >> 16); a2 += (uint8_t)(c >> 24);

    c = cur_img(i+1, y);
    r0 += (uint8_t)c; g0 += (uint8_t)(c >> 8); b0 += (uint8_t)(c >> 16); a0 += (uint8_t)(c >> 24);

    c = cur_img(i+1, x+1);
    r1 += (uint8_t)c; g1 += (uint8_t)(c >> 8); b1 += (uint8_t)(c >> 16); a1 += (uint8_t)(c >> 24);

    c = cur_img(i+1, x+2);
    r2 += (uint8_t)c; g2 += (uint8_t)(c >> 8); b2 += (uint8_t)(c >> 16); a2 += (uint8_t)(c >> 24);

    for (int j = x + 1; j < x + width - 1; j++) {
      // Compute the average 
      r = (r0 + r1 + r2) / 9;
      g = (g0 + g1 + g2) / 9;
      b = (b0 + b1 + b2) / 9;
      a = (a0 + a1 + a2) / 9;

      next_img(i, j) = ezv_rgba(r, g, b, a);

      // Shift the columns to the left 
      r0 = r1; g0 = g1; b0 = b1; a0 = a1;
      r1 = r2; g1 = g2; b1 = b2; a1 = a2;
      // Compute values for the next column 
      c = cur_img(i - 1, j + 2);
      r2 = (uint8_t)c; g2 = (uint8_t)(c >> 8); b2 = (uint8_t)(c >> 16); a2 = (uint8_t)(c >> 24);

      c = cur_img(i, j + 2);
      r2 += (uint8_t)c; g2 += (uint8_t)(c >> 8); b2 += (uint8_t)(c >> 16); a2 += (uint8_t)(c >> 24);

      c = cur_img(i + 1, j + 2);
      r2 += (uint8_t)c; g2 += (uint8_t)(c >> 8); b2 += (uint8_t)(c >> 16); a2 += (uint8_t)(c >> 24);
    }
  }

  
  // Manage the borders
  for(int i = y+1; i < y + height-1; i++){	
    r = 0, g = 0, b = 0, a = 0;
    // first pixel of the row
    for (int yloc = 0; yloc <= 1; yloc++){
      for (int xloc = i-1; xloc <= i+1; xloc++) {
        c = cur_img (xloc, yloc);
        r += ezv_c2r (c);
        g += ezv_c2g (c);
        b += ezv_c2b (c);
        a += ezv_c2a (c);
      }
    }
    r /= 6; g /= 6; b /= 6; a /= 6;
    // set the first pixel of the row
    next_img (i, x) = ezv_rgba (r, g, b, a);	

    r = 0, g = 0, b = 0, a = 0;
    // Finally we compute the last pixel of the row
    for (int yloc = DIM-2; yloc <= DIM-1; yloc++){
      for (int xloc = i-1; xloc <= i+1; xloc++) {
        c = cur_img (xloc, yloc);
        r += ezv_c2r (c);
        g += ezv_c2g (c);
        b += ezv_c2b (c);
        a += ezv_c2a (c);
      }
    }
    r /= 6; g /= 6; b /= 6; a /= 6;
    // set the last pixel of the row
    next_img (i, DIM-1) = ezv_rgba (r, g, b, a);

  } 

  // manage the first and last row
  for(int j = x; j < x + width; j++){	
    r = 0, g = 0, b = 0, a = 0;
    n = 0;
    j_d = (j > 0) ? j - 1 : j;
    j_f = (j < DIM - 1) ? j + 1 : j;

    next_img(y, j) = cur_img(y, j);	
    next_img(DIM-1,j) = cur_img(DIM-1,j);	

    // first row
    for (int yloc = j_d; yloc <= j_f; yloc++){
      for (int xloc = 0; xloc <= 1; xloc++) {
        c = cur_img (xloc, yloc);
        r += ezv_c2r (c); g += ezv_c2g (c); b += ezv_c2b (c); a += ezv_c2a (c);
        n++;
      }
    }
    // we must use n because the number of neighbors is variable
    r /= n; g /= n; b /= n; a /= n; 
    next_img (y, j) = ezv_rgba (r, g, b, a);

    r = 0, g = 0, b = 0, a = 0;
    n = 0;
    // last row
    for (int yloc = j_d; yloc <= j_f; yloc++){
      for (int xloc = DIM-2; xloc <= DIM-1; xloc++) {
        c = cur_img (xloc, yloc);
        r += ezv_c2r (c); g += ezv_c2g (c); b += ezv_c2b (c); a += ezv_c2a (c);
        n++;
      }
    }
    r /= n; g /= n; b /= n; a /= n;
    next_img (DIM-1, j) = ezv_rgba (r, g, b, a);
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