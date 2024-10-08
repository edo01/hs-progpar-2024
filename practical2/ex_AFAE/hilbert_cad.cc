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
 * ----------------------------------------------------------
 * |      determinant of Hilbert’s matrix of size 11         |
 * ----------------------------------------------------------
 * Pivot   0   =  0.100000000000000E+001
 * Pivot   1   =  0.833333333333333E-001
 * Pivot   2   =  0.55555555555555E-002
 * Pivot   3   =  0.3571428571428E-003
 * Pivot   4   =  0.22675736961E-004
 * Pivot   5   =  0.143154905E-005
 * Pivot   6   =  0.90097492E-007
 * Pivot   7   =  0.5659970E-008
 * Pivot   8   =  0.35513E-009
 * Pivot   9   =  0.2226E-010
 * Pivot  10   =  0.13E-011
 * Determinant =  0.30E-064
 * ----------------------------------------------------------------
 * CADNA_C 3.1.11 software
 * No instability detected
 * ----------------------------------------------------------------
 */

int main()
{
  // initialize Cadna
  cadna_init(-1);

  double_st amat[11][11];
  int i,j,k;
  double_st aux, det;

  printf("----------------------------------------------------------\n");
  printf("|      determinant of Hilbert's matrix of size 11         |\n");
  printf("----------------------------------------------------------\n");

  for(i=1;i<=11;i++)
    for(j=1;j<=11;j++)
      amat[i-1][j-1] = 1./(double_st)(i+j-1);
  
  det = 1.;

  for(i=0;i<10;i++){
    printf("Pivot %3d   = %s\n",i, strp(amat[i][i]));
    det = det*amat[i][i];
    aux = 1./amat[i][i];
    for(j=i+1;j<11;j++)
      amat[i][j] = amat[i][j]*aux;
    
    for(j=i+1;j<11;j++){
      aux = amat[j][i];
      for(k=i+1;k<11;k++)
	amat[j][k] = amat[j][k] - aux*amat[i][k];
    }
  }
  printf("Pivot %3d   = %s\n",i, strp(amat[i][i]));
  det = det*amat[i][i];
  printf("Determinant = %s\n", strp(det));

  // finalize Cadna
  cadna_end();
  
  return 0;
}
