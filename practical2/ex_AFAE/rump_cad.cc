//include cadna
#include <cadna.h>
#include <math.h>
#include <stdio.h>
/**
 * ------------------------------------------
 * |  Polynomial function of two variables  |
 * ------------------------------------------
 * 
 * ----------------------------------------------------------------
 * CADNA_C 3.1.11 software
 * Self-validation detection: ON
 * Mathematical instabilities detection: ON
 * Branching instabilities detection: ON
 * Intrinsic instabilities detection: ON
 * Cancellation instabilities detection: ON
 * ----------------------------------------------------------------
 * ------------------------------------------
 * |  Polynomial function of two variables  |
 * ------------------------------------------
 * res= @.0
 * ----------------------------------------------------------------
 * CADNA_C 3.1.11 software
 * There is  1 numerical instability
 * 1 LOSS(ES) OF ACCURACY DUE TO CANCELLATION(S)
 * ----------------------------------------------------------------
 */
int main()
{
  // initialize Cadna
  cadna_init(-1);

  printf("------------------------------------------\n");
  printf("|  Polynomial function of two variables  |\n");
  printf("------------------------------------------\n");

  float_st x = 77617.;
  float_st y = 33096.;
  float_st res;

  res=333.75*y*y*y*y*y*y+x*x*(11.*x*x*y*y-y*y*y*y*y*y-121.*y*y*y*y-2.0)   
    +5.5*y*y*y*y*y*y*y*y+x/(2.*y);

  printf("res=%s\n", strp(res));

  // finalize Cadna
  cadna_end();

  return 0;
}
