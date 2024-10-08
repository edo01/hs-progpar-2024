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
 * ------------------------------------------------------
 * | Solving a linear system using Gaussian elimination |
 * | with partial pivoting                              |
 * ------------------------------------------------------
 * xsol(0) =  0.99E+000 (exact solution: xsol(0)=  0.1000000E+001)
 * xsol(1) =  0.100E+001 (exact solution: xsol(1)=  0.1000000E+001)
 * xsol(2) =  0.999999E-008 (exact solution: xsol(2)=  0.1000000E-007)
 * xsol(3) =  0.100000E+001 (exact solution: xsol(3)=  0.1000000E+001)
 * ----------------------------------------------------------------
 * CADNA_C 3.1.11 software
 * There are 3 numerical instabilities
 * 1 UNSTABLE BRANCHING(S)
 * 1 UNSTABLE INTRINSIC FUNCTION(S)
 * 1 LOSS(ES) OF ACCURACY DUE TO CANCELLATION(S)
 * ----------------------------------------------------------------
 */

#define IDIM 4

int main()
{
  cadna_init(-1);

  float_st a[IDIM][IDIM+1]={
    {  21.0, 130.0,       0.0,    2.1,  153.1},
    {  13.0,  80.0,   4.74e+8,  752.0, 849.74},
    {   0.0,  -0.4, 3.9816e+8,    4.2, 7.7816},
    {   0.0,   0.0,       1.7, 9.0E-9, 2.6e-8}};

  float_st  xsol[IDIM]={1., 1., 1.e-8,1.};

  printf("------------------------------------------------------\n");
  printf("| Solving a linear system using Gaussian elimination |\n");
  printf("| with partial pivoting                              |\n");
  printf("------------------------------------------------------\n");
  
  int i,j,k,ll;
  float_st pmax, aux;

  for(i=0; i<IDIM-1;i++){
    
    pmax = 0.0;
    for(j=i; j<IDIM;j++){
      if(fabsf(a[j][i])>pmax){
	pmax = fabsf(a[j][i]);
	ll = j;
      }
    }
    if (ll!=i) {
      for(j=i; j<IDIM+1;j++){ 
	aux = a[i][j];
	a[i][j] = a[ll][j];
	a[ll][j] = aux;
      }
    }
    aux = a[i][i];
    for(j=i+1; j<IDIM+1;j++)
      a[i][j] = a[i][j]/aux;
    
    for (k=i+1;k<IDIM;k++){
      aux = a[k][i];
      for(j=i+1;j<IDIM+1;j++)
	a[k][j]=a[k][j] - aux*a[i][j];
    }
  }

  a[IDIM-1][IDIM] = a[IDIM-1][IDIM]/a[IDIM-1][IDIM-1];
  
  for(i=IDIM-2;i>=0;i--)
    for(j=i+1;j<IDIM;j++)
      a[i][IDIM] = a[i][IDIM] - a[i][j]*a[j][IDIM];
  
  for( i=0;i<IDIM;i++)
    printf("xsol(%d) = %s (exact solution: xsol(%d)= %s)\n",
	   i,strp(a[i][IDIM]),i,strp(xsol[i]));

  cadna_end();

  return 0;  
}











