#include <math.h>
#include <stdio.h>
/**
 * ------------------------------------------
 * |  Polynomial function of two variables  |
 * ------------------------------------------
 * res=1.17260394005318e+00
 *
 * We can see that increasing the precision reduces the error,
 * but the result is still not correct. 
 */
int main()
{
  printf("------------------------------------------\n");
  printf("|  Polynomial function of two variables  |\n");
  printf("------------------------------------------\n");

  double x = 77617.;
  double y = 33096.;
  double res;

  res=333.75*y*y*y*y*y*y+x*x*(11.*x*x*y*y-y*y*y*y*y*y-121.*y*y*y*y-2.0)   
    +5.5*y*y*y*y*y*y*y*y+x/(2.*y);

  printf("res=%.14e\n",res);
  return 0;
}
