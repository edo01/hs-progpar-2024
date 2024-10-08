#include <stdio.h>
#include <math.h>
/**
 * ----------------------------------------------------------
 * Pivot   0   = +1.000000000000000e+00
 * Pivot   1   = +8.333333333333331e-02
 * Pivot   2   = +5.555555555555522e-03
 * Pivot   3   = +3.571428571428736e-04
 * Pivot   4   = +2.267573696146732e-05
 * Pivot   5   = +1.431549050481817e-06
 * Pivot   6   = +9.009749236431395e-08
 * Pivot   7   = +5.659970607161749e-09
 * Pivot   8   = +3.551362553328898e-10
 * Pivot   9   = +2.226656943069665e-11
 * Pivot  10   = +1.398301799864147e-12
 * Determinant = +3.026439382718219e-65
 */

int main()
{
  double amat[11][11];
  int i,j,k;
  double aux, det;

  printf("----------------------------------------------------------\n");
  printf("|      determinant of Hilbert’s matrix of size 11         |\n");
  printf("----------------------------------------------------------\n");

  for(i=1;i<=11;i++)
    for(j=1;j<=11;j++)
      amat[i-1][j-1] = 1./(double)(i+j-1);
  
  det = 1.;

  for(i=0;i<10;i++){
    printf("Pivot %3d   = %+.15e\n",i,amat[i][i]);
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
  printf("Pivot %3d   = %+.15e\n",i,amat[i][i]);
  det = det*amat[i][i];
  printf("Determinant = %+.15e\n",det);
  return 0;
}
