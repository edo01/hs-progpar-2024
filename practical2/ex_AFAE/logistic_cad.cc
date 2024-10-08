#include <cadna.h>
#include <stdio.h>
#include <math.h>

/**
 * ----------------------------------------------------------------
 * CADNA_C 3.1.11 software
 * Self-validation detection: ON
 * Mathematical instabilities detection: ON
 * Branching instabilities detection: ON
 * Intrinsic instabilities detection: ON
 * Cancellation instabilities detection: ON
 * ----------------------------------------------------------------
 * -----------------------------
 * |  Logistic iteration       |
 * -----------------------------
 * i= 50 x= 0.3875950636E+000
 * i=100 x= 0.45640E+000
 * i=150 x= 0.4E+000
 * i=200 x= @.0
 * last iterate:
 * i=200 x= @.0
 * ----------------------------------------------------------------
 * CADNA_C 3.1.11 software
 * 
 * CRITICAL WARNING: the self-validation detects major problem(s).
 * The results are NOT guaranteed.
 * 
 * There are 10 numerical instabilities
 * 10 UNSTABLE MULTIPLICATION(S)
 * ----------------------------------------------------------------
 * 
 */

int main (){
	cadna_init(-1);

  double_st x,a;
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
         if (!(i%50)) printf("i=%3d x=%s\n",i,strp(x));
      }  
  while (i<200);
  printf("last iterate:\ni=%d x=%s\n",i,strp(x));

  cadna_end();
  
  return 0;
}
