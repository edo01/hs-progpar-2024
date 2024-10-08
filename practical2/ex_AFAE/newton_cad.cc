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
 * -------------------------------------------------------------
 * |  Computation of a root of a polynomial by Newton's method |
 * -------------------------------------------------------------
 * x(  1) =  0.464864864864864E+000, diff =  0.35135135135135E-001
 * x(  2) =  0.446871334005381E+000, diff =  0.17993530859483E-001
 * x(  3) =  0.43776082261777E+000, diff =  0.9110511387604E-002
 * x(  4) =  0.43317613556077E+000, diff =  0.458468705700E-002
 * x(  5) =  0.43087630369577E+000, diff =  0.229983186499E-002
 * x(  6) =  0.4297244989611E+000, diff =  0.11518047345E-002
 * x(  7) =  0.4291481222774E+000, diff =  0.5763766836E-003
 * x(  8) =  0.4288598150904E+000, diff =  0.2883071870E-003
 * x(  9) =  0.428715631752E+000, diff =  0.144183338E-003
 * x( 10) =  0.428643532642E+000, diff =  0.720991094E-004
 * x( 11) =  0.428607481227E+000, diff =  0.36051415E-004
 * x( 12) =  0.42858945505E+000, diff =  0.1802617E-004
 * x( 13) =  0.42858044185E+000, diff =  0.901320E-005
 * x( 14) =  0.42857593522E+000, diff =  0.450662E-005
 * x( 15) =  0.42857368190E+000, diff =  0.225332E-005
 * x( 16) =  0.42857255523E+000, diff =  0.11266E-005
 * x( 17) =  0.4285719919E+000, diff =  0.5633E-006
 * x( 18) =  0.428571710E+000, diff =  0.281E-006
 * x( 19) =  0.428571569E+000, diff =  0.140E-006
 * x( 20) =  0.428571499E+000, diff =  0.70E-007
 * x( 21) =  0.428571463E+000, diff =  0.35E-007
 * x( 22) =  0.428571446E+000, diff =  0.1E-007
 * x( 23) =  0.42857143E+000, diff =  0.8E-008
 * x( 24) =  0.42857143E+000, diff =  @.0
 * ----------------------------------------------------------------
 * CADNA_C 3.1.11 software
 * There are 45 numerical instabilities
 * 45 LOSS(ES) OF ACCURACY DUE TO CANCELLATION(S)
 * ----------------------------------------------------------------
 * 
 */

int main()
{
  cadna_init(-1);

  int i, nmax=100;
  double_st y, x, diff, eps=1.e-12; 
  printf("-------------------------------------------------------------\n");
  printf("|  Computation of a root of a polynomial by Newton's method |\n");
  printf("-------------------------------------------------------------\n");
  
  y = 0.5;
  for(i = 1;i<=nmax;i++){
    x = y;
    y = x-(1.47*pow(x,3)+1.19*pow(x,2)-1.83*x+0.45)/
      (4.41*pow(x,2)+2.38*x-1.83);
    diff = fabs(x-y);
    printf("x(%3d) = %s, diff = %s\n", i, strp(y), strp(diff));     
    if (x==y) break;
  }

  cadna_end();
  return 0;
}
