#include <cadna.h>
#include <math.h>
#include <stdio.h>

/**
 * 
 * ----------------------------------------------------------------
 * CADNA_C 3.1.11 software
 * Self-validation detection: ON
 * Mathematical instabilities detection: ON
 * Branching instabilities detection: ON
 * Intrinsic instabilities detection: ON
 * Cancellation instabilities detection: ON
 * ----------------------------------------------------------------
 * -------------------------------------
 * | A second order recurrent sequence |
 * -------------------------------------
 * U(2) =  0.55901639344262E+001
 * U(3) =  0.5633431085044E+001
 * U(4) =  0.56746486205E+001
 * U(5) =  0.5713329052E+001
 * U(6) =  0.574912092E+001
 * U(7) =  0.57818109E+001
 * U(8) =  0.581131E+001
 * U(9) =  0.58376E+001
 * U(10) =  0.5860E+001
 * U(11) =  0.588E+001
 * U(12) =  0.59E+001
 * U(13) =  @.0
 * U(14) =  @.0
 * U(15) =  @.0
 * U(16) =  0.9E+002
 * U(17) =  0.99E+002
 * U(18) =  0.999E+002
 * U(19) =  0.99999E+002
 * U(20) =  0.999999E+002
 * U(21) =  0.9999999E+002
 * U(22) =  0.99999999E+002
 * U(23) =  0.999999999E+002
 * U(24) =  0.99999999999E+002
 * U(25) =  0.999999999999E+002
 * U(26) =  0.9999999999999E+002
 * U(27) =  0.99999999999999E+002
 * U(28) =  0.100000000000000E+003
 * U(29) =  0.100000000000000E+003
 * U(30) =  0.100000000000000E+003
 * ----------------------------------------------------------------
 * CADNA_C 3.1.11 software
 * 
 * CRITICAL WARNING: the self-validation detects major problem(s).
 * The results are NOT guaranteed.
 * 
 * There are 8 numerical instabilities
 * 6 UNSTABLE DIVISION(S)
 * 2 UNSTABLE MULTIPLICATION(S)
 * ----------------------------------------------------------------
 * 
 */

int main()
{
  cadna_init(-1);

  int i;
  double_st a,b,c;

  printf("-------------------------------------\n");
  printf("| A second order recurrent sequence |\n");
  printf("-------------------------------------\n");

  a = 5.5;
  b = 61./11.;
  for(i=2;i<=30;i++){
    c = b;
    b = 111. - 1130./b + 3000./(a*b);
    a = c;
    printf("U(%d) = %s\n",i, strp(b));
  }

  cadna_end();

  return 0;
}

