#include <stdio.h>
#include <math.h>

/**
 * -----------------------------
 * |  Logistic iteration       |
 * -----------------------------
 * i= 50 x= 3.875950637036066e-01
 * i=100 x= 4.564074671150947e-01
 * i=150 x= 4.062664991348743e-01
 * i=200 x= 5.761024866343428e-01
 * last iterate:
 * i=200 x= 5.761024866343428e-01
 */

int main (){
	
  double x,a;
  int i=0;
  printf("-----------------------------\n");
  printf("|  Logistic iteration       |\n");
  printf("-----------------------------\n");
  
  x=0.6;
  a=3.6;

  do {
         x=a*x*(1-x);	
         //x=a*0.25-a*pow((x-0.5),2);
	 i=i+1;
         if (!(i%50)) printf("i=%3d x=%22.15e\n",i,x);
      }  
  while (i<200);
  printf("last iterate:\ni=%d x=%22.15e\n",i,x);
  return 0;
}
