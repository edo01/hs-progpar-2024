#include <cadna.h>
#include <math.h>
#include <stdio.h>

/**
 * ----------------------------------------------------------------
 * CADNA_C 3.1.11 software
 * Self-validation detection: ON
 * Mathematical instabilities detection: ON
 * Branching instabilities detection: ON
 * Intrinsic instabilities detection: ON
 * Cancellation instabilities detection: ON
 * ----------------------------------------------------------------
 * -----------------------------------------------------------
 * |                 Jacobi iteration                        |
 * -----------------------------------------------------------
 * anorm =  0.994084E+004
 * anorm =  0.640232E+004
 * anorm =  0.482850E+004
 * anorm =  0.22048E+004
 * anorm =  0.15047E+004
 * anorm =  0.11195E+004
 * anorm =  0.6281E+003
 * anorm =  0.30117E+003
 * anorm =  0.1980E+003
 * anorm =  0.11084E+003
 * anorm =  0.96123E+002
 * anorm =  0.4052E+002
 * anorm =  0.1861E+002
 * anorm =  0.1322E+002
 * anorm =  0.8839E+001
 * anorm =  0.528E+001
 * anorm =  0.317E+001
 * anorm =  0.23E+001
 * anorm =  0.184E+001
 * anorm =  0.72E+000
 * anorm =  0.47E+000
 * anorm =  0.23E+000
 * anorm =  0.18E+000
 * anorm =  0.92E-001
 * anorm =  0.8E-001
 * anorm =  0.6E-001
 * anorm =  0.2E-001
 * anorm =  0.1E-001
 * anorm =  @.0
 * anorm =  @.0
 * anorm =  @.0
 * anorm =  0.0000000E+000
 * niter = 31
 * eps =  0.1000000E-003
 * x_sol( 0) =  0.169E+001 (correct value:  0.1700000E+001), error( 0) =  @.0
 * x_sol( 1) = -0.474689E+004 (correct value: -0.4746889E+004), error( 1) =  @.0
 * x_sol( 2) =  0.5023E+002 (correct value:  0.5023000E+002), error( 2) =  @.0
 * x_sol( 3) = -0.24532E+003 (correct value: -0.2453199E+003), error( 3) =  @.0
 * x_sol( 4) =  0.477829E+004 (correct value:  0.4778290E+004), error( 4) =  @.0
 * x_sol( 5) = -0.75729E+002 (correct value: -0.7572999E+002), error( 5) =  @.0
 * x_sol( 6) =  0.349543E+004 (correct value:  0.3495430E+004), error( 6) =  @.0
 * x_sol( 7) =  0.434E+001 (correct value:  0.4350000E+001), error( 7) =  @.0
 * x_sol( 8) =  0.45297E+003 (correct value:  0.4529799E+003), error( 8) =  @.0
 * x_sol( 9) = -0.275E+001 (correct value: -0.2759999E+001), error( 9) =  @.0
 * x_sol(10) =  0.823923E+004 (correct value:  0.8239240E+004), error(10) =  @.0
 * x_sol(11) =  0.345E+001 (correct value:  0.3460000E+001), error(11) =  @.0
 * x_sol(12) =  0.100000E+004 (correct value:  0.1000000E+004), error(12) =  @.0
 * x_sol(13) = -0.499E+001 (correct value: -0.5000000E+001), error(13) =  @.0
 * x_sol(14) =  0.364239E+004 (correct value:  0.3642400E+004), error(14) =  @.0
 * x_sol(15) =  0.73535E+003 (correct value:  0.7353600E+003), error(15) =  @.0
 * x_sol(16) =  0.17E+001 (correct value:  0.1700000E+001), error(16) =  @.0
 * x_sol(17) = -0.234916E+004 (correct value: -0.2349169E+004), error(17) =  @.0
 * x_sol(18) = -0.824751E+004 (correct value: -0.8247519E+004), error(18) =  @.0
 * x_sol(19) =  0.9843569E+004 (correct value:  0.9843570E+004), error(19) =  @.0
 * ----------------------------------------------------------------
 * CADNA_C 3.1.11 software
 * There are 376 numerical instabilities
 * 111 UNSTABLE BRANCHING(S)
 * 35 UNSTABLE INTRINSIC FUNCTION(S)
 * 230 LOSS(ES) OF ACCURACY DUE TO CANCELLATION(S)
 * ----------------------------------------------------------------
 */

static int  nrand;

float random1()
{
  nrand = (nrand*5363 + 143) % 1387;
  return((float)(2.0*nrand/1387.0 - 1.0));
}

int main()
{
  cadna_init(-1);

  const float_st eps = 1.e-4; // check me
  const int ndim = 20, niter = 1000;

  float_st a[ndim][ndim];
  float_st b[ndim];
  float_st x[ndim];
  float_st y[ndim];
  float_st xsol[]={  1.7, -4746.89, 50.23, -245.32,
		       4778.29, -75.73, 3495.43, 4.35,
		       452.98, -2.76, 8239.24, 3.46,
		       1000.0, -5.0, 3642.4, 735.36,
		       1.7, -2349.17, -8247.52, 9843.57 };
  int i, j, k;
  float_st aux, anorm;
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

    // print anorm
    printf("anorm = %s\n", strp(anorm));

    if (anorm<eps) break;
  }
  printf("niter = %d\n",i-1);
  printf("eps = %s\n", strp(eps));

  for (i=0;i<ndim;i++){
    aux = -b[i];
    for (j=0;j<ndim;j++)
      aux = aux + a[i][j]*y[j];
    printf("x_sol(%2d) = %s (correct value: %s), error(%2d) = %s\n",i,strp(y[i]),strp(xsol[i]),i,strp(aux));    
  }
  
  cadna_end();
  return 0;
}
