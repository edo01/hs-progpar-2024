#include <math.h>
#include <stdio.h>
/**
 * ------------------------------------------
 * |  Polynomial function of two variables  |
 * ------------------------------------------
 * res = 7.08931176699221e+29 (NOT CORRECT)
 */

int main()
{
  printf("------------------------------------------\n");
  printf("|  Polynomial function of two variables  |\n");
  printf("------------------------------------------\n");

  float x = 77617.;
  float y = 33096.;
  float res;

  res=333.75*y*y*y*y*y*y+x*x*(11.*x*x*y*y-y*y*y*y*y*y-121.*y*y*y*y-2.0)   
    +5.5*y*y*y*y*y*y*y*y+x/(2.*y);

  printf("res=%.14e\n",res);
  return 0;
}
