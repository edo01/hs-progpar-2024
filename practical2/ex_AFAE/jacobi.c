#include <math.h>
#include <stdio.h>

/**
 * -----------------------------------------------------------
 * |                 Jacobi iteration                        |
 * -----------------------------------------------------------
 * niter = 1000
 * eps = 1.e-4
 * x_sol( 0) = +1.699764e+00 (correct value: +1.700000e+00), error( 0) = -7.934570e-04
 * x_sol( 1) = -4.746890e+03 (correct value: -4.746890e+03), error( 1) = -1.464844e-03
 * x_sol( 2) = +5.023014e+01 (correct value: +5.023000e+01), error( 2) = -4.882812e-04
 * x_sol( 3) = -2.453199e+02 (correct value: -2.453200e+02), error( 3) = -1.708984e-03
 * x_sol( 4) = +4.778291e+03 (correct value: +4.778290e+03), error( 4) = +4.394531e-03
 * x_sol( 5) = -7.572982e+01 (correct value: -7.573000e+01), error( 5) = -9.765625e-04
 * x_sol( 6) = +3.495429e+03 (correct value: +3.495430e+03), error( 6) = +9.765625e-04
 * x_sol( 7) = +4.350384e+00 (correct value: +4.350000e+00), error( 7) = +4.882812e-04
 * x_sol( 8) = +4.529805e+02 (correct value: +4.529800e+02), error( 8) = +4.882812e-04
 * x_sol( 9) = -2.759486e+00 (correct value: -2.760000e+00), error( 9) = +2.929688e-03
 * x_sol(10) = +8.239240e+03 (correct value: +8.239240e+03), error(10) = +3.417969e-03
 * x_sol(11) = +3.460304e+00 (correct value: +3.460000e+00), error(11) = +0.000000e+00
 * x_sol(12) = +9.999997e+02 (correct value: +1.000000e+03), error(12) = -5.371094e-03
 * x_sol(13) = -5.000281e+00 (correct value: -5.000000e+00), error(13) = -9.765625e-04
 * x_sol(14) = +3.642400e+03 (correct value: +3.642400e+03), error(14) = -9.765625e-04
 * x_sol(15) = +7.353594e+02 (correct value: +7.353600e+02), error(15) = -2.441406e-04
 * x_sol(16) = +1.700638e+00 (correct value: +1.700000e+00), error(16) = +3.417969e-03
 * x_sol(17) = -2.349171e+03 (correct value: -2.349170e+03), error(17) = -1.953125e-03
 * x_sol(18) = -8.247520e+03 (correct value: -8.247520e+03), error(18) = +3.234863e-03
 * x_sol(19) = +9.843569e+03 (correct value: +9.843570e+03), error(19) = -3.906250e-03
 */

static int  nrand;

float random1()
{
  nrand = (nrand*5363 + 143) % 1387;
  return((float)(2.0*nrand/1387.0 - 1.0));
}

int main()
{
  const float eps = 1.e-4; 
  const int ndim = 20, niter = 1000;

  float a[ndim][ndim];
  float b[ndim];
  float x[ndim];
  float y[ndim];
  float xsol[]={  1.7, -4746.89, 50.23, -245.32,
		       4778.29, -75.73, 3495.43, 4.35,
		       452.98, -2.76, 8239.24, 3.46,
		       1000.0, -5.0, 3642.4, 735.36,
		       1.7, -2349.17, -8247.52, 9843.57 };
  int i, j, k;
  float aux, anorm;
  nrand = 23;
  printf("-----------------------------------------------------------\n");
  printf("|                 Jacobi iteration                        |\n");
  printf("-----------------------------------------------------------\n");
  
 
  for(i=0;i<ndim;i++){
    for (j=0;j<ndim;j++){
          a[i][j] = random1();
    }
    a[i][i] = a[i][i] + 4.500002;
  }
  for (i=0;i<ndim;i++){
    aux = 0.0;
    for (j=0;j<ndim;j++)
      aux = aux + a[i][j]*xsol[j];
    b[i] = aux;
    y[i] = 10.0;
  }

  for (i=1;i<=niter;i++){
    anorm = 0.0;
    for (j=0;j<ndim;j++)  x[j] = y[j];
        
    for (j=0;j<ndim;j++){
      aux = b[j];
      for (k=0;k<ndim;k++) 
	if (k!=j) aux = aux - a[j][k]*x[k];
      y[j] = aux/a[j][j];
      if (fabsf(x[j]-y[j])>anorm) anorm = fabsf(x[j]-y[j]);
    }
    if (anorm<eps) break;
  }
  printf("niter = %d\n",i-1);
  printf("eps = %+.1e\n",eps);

  for (i=0;i<ndim;i++){
    aux = -b[i];
    for (j=0;j<ndim;j++)
      aux = aux + a[i][j]*y[j];
    printf("x_sol(%2d) = %+.6e (correct value: %+.6e), error(%2d) = %+.6e\n",i,y[i],xsol[i],i,aux);    
  }
  return 0;
}
