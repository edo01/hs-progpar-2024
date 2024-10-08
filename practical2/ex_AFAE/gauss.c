#include <stdio.h>
#include <math.h>

/**
 * ------------------------------------------------------
 * | Solving a linear system using Gaussian elimination |
 * | with partial pivoting                              |
 * ------------------------------------------------------
 * xsol(0) = +6.261988e+01 (exact solution: xsol(0)= +1.000000e+00)
 * xsol(1) = -8.953979e+00 (exact solution: xsol(1)= +1.000000e+00)
 * xsol(2) = +0.000000e+00 (exact solution: xsol(2)= +1.000000e-08)
 * xsol(3) = +9.999999e-01 (exact solution: xsol(3)= +1.000000e+00)
 */

#define IDIM 4

int main()
{
  float a[IDIM][IDIM+1]={
    {  21.0, 130.0,       0.0,    2.1,  153.1},
    {  13.0,  80.0,   4.74e+8,  752.0, 849.74},
    {   0.0,  -0.4, 3.9816e+8,    4.2, 7.7816},
    {   0.0,   0.0,       1.7, 9.0E-9, 2.6e-8}};

  float  xsol[IDIM]={1., 1., 1.e-8,1.};

  printf("------------------------------------------------------\n");
  printf("| Solving a linear system using Gaussian elimination |\n");
  printf("| with partial pivoting                              |\n");
  printf("------------------------------------------------------\n");
  
  int i,j,k,ll;
  float pmax, aux;

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
    printf("xsol(%d) = %+.6e (exact solution: xsol(%d)= %+.6e)\n",
	   i,a[i][IDIM],i,xsol[i]);

  return 0;  
}











