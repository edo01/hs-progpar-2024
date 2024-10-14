#include "easypap.h"

static float *tgrid0;
static float *tgrid1;

unsigned heat_to_rgb_legacy (float h) // 0.0 = cold, 1.0 = hot
{
  int i;
  float f, p, q, t;
  float v = 1.0;
  float s = 1.0;

  if (s == 0) {
    // achromatic (grey)
    int c = v * 255;
    return ezv_rgba (c, c, c, 255);
  }

  h = (1.0 - h) * 4; // sector 0.0 to 4.0
  if (h == 4.0)
    h = 3.99999;
  i = h;
  f = h - i; // factorial part of h
  p = v * (1 - s);
  q = v * (1 - s * f);
  t = v * (1 - s * (1 - f));

  switch (i) {
  case 0:
    return ezv_rgba (v * 255, t * 255, p * 255, 255);
  case 1:
    return ezv_rgba (q * 255, v * 255, p * 255, 255);
  case 2:
    return ezv_rgba (p * 255, v * 255, t * 255, 255);
  default:
    return ezv_rgba (p * 255, q * 255, v * 255, 255);
  }
}

// ============================================================================
// ======================================================================== SEQ
// ============================================================================

void heat_init() {
  tgrid0 = (float *)malloc(sizeof(float) * DIM * DIM);
  tgrid1 = (float *)malloc(sizeof(float) * DIM * DIM);

  for (unsigned i = 0; i < DIM * DIM; i++) {
    tgrid0[i] = 0.f;
    tgrid1[i] = 0.f;
  }

  unsigned ss = DIM / 2;  // square size
  unsigned shift = DIM / 16;

  // put an hot spot in the top left corner
  for (unsigned i = 0; i < ss; i++)
    for (unsigned j = 0; j < ss; j++)
      tgrid0[(j + shift) * DIM + i + shift] = 1.f;
}

///////////////////////////// Sequential version (tiled)
// Suggested cmdline(s):
// ./run -k heat -v seq -si
//
int heat_do_tile_default(int x, int y, int width, int height) {
  float c,l,r,b,t;
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++) {
      c =                      tgrid0[(i + 0) * DIM + (j + 0)];
      l = (j - 1) >=       0 ? tgrid0[(i + 0) * DIM + (j - 1)] : tgrid0[  (i + 0) * DIM + (DIM - 1)];
      r = (j + 1) < (int)DIM ? tgrid0[(i + 0) * DIM + (j + 1)] : tgrid0[(  i + 0) * DIM + (      0)];
      b = (i - 1) >=       0 ? tgrid0[(i - 1) * DIM + (j + 0)] : tgrid0[(DIM - 1) * DIM + (  j + 0)];
      t = (i + 1) < (int)DIM ? tgrid0[(i + 1) * DIM + (j + 0)] : tgrid0[(      0) * DIM + (  j + 0)];
      tgrid1[i * DIM + j] = (c + l + r + t + b) * 0.2f;
    }

  return 0;
}

///////////////////////////// Sequential version (tiled)
// Suggested cmdline(s):
// ./run -k heat -v seq -r 1000
// ./run -k heat -v seq -i 10000 --no-display
//
unsigned heat_compute_seq(unsigned nb_iter) {
  for (unsigned it = 1; it <= nb_iter; it++) {
    do_tile(0, 0, DIM, DIM, 0);

    float *tmp = tgrid1;
    tgrid1 = tgrid0;
    tgrid0 = tmp;
  }

  for (unsigned i = 0; i < DIM; i++)
    for (unsigned j = 0; j < DIM; j++)
      cur_img(i, j) = heat_to_rgb_legacy(tgrid0[i * DIM + j]);

  return 0;
}

// ============================================================================
// ================================================================== BORDER V2
// ============================================================================

void heat_init_seq_bv2() {
  // point 1
  tgrid0 = (float *)malloc(sizeof(float)* (DIM + 2) * ( DIM + 2));
  tgrid1 = (float *)malloc(sizeof(float) * (DIM + 2) * (DIM + 2));
  
  for (unsigned i = 0; i < (DIM+2)*(DIM+2); i++) {
    tgrid0[i] = 0.f;
    tgrid1[i] = 0.f;
  }

  // point 2
  unsigned ss = DIM / 2;  // square size
  unsigned shift = DIM / 16;
  for (unsigned i = 0; i < ss; i++)
    for (unsigned j = 0; j < ss; j++)
      // we skip the first column and row
      tgrid0[((j+1) + shift) * (DIM + 2) + (i+1) + shift] = 1.f;
}

int heat_do_tile_bv2(int x, int y, int width, int height) {
  float c,l,r,b,t;
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++) {
      // point 3
      c = tgrid0[(i + 1 + 0) * (DIM + 2) + (j + 1 + 0)];
      l = tgrid0[(i + 1 + 0) * (DIM + 2) + (j + 1 - 1)];
      r = tgrid0[(i + 1 + 0) * (DIM + 2) + (j + 1 + 1)];
      b = tgrid0[(i + 1 - 1) * (DIM + 2) + (j + 1 + 0)];
      t = tgrid0[(i + 1 + 1) * (DIM + 2) + (j + 1 + 0)];
      tgrid1[(i + 1) * (DIM + 2) + (j + 1)] = (c + l + r + t + b) * 0.2f;
    }
  return 0;
 }

// // ./run -k heat -v seq_bv2 -wt bv2 -r 1000
// // ./run -k heat -v seq_bv2 -wt bv2 -i 10000 --no-display
unsigned heat_compute_seq_bv2(unsigned nb_iter) {
  for (unsigned it = 1; it <= nb_iter; it++) {
    // copie des bords
    for (int i = 0; i < (int)DIM; i++) {
      tgrid0[       0  * (DIM + 2) +   i + 1] = tgrid0[   DIM  * (DIM + 2) + i + 1]; // top   <= bot
      tgrid0[(DIM + 1) * (DIM + 2) +   i + 1] = tgrid0[     1  * (DIM + 2) + i + 1]; // bot   <= top
      tgrid0[  (i + 1) * (DIM + 2)       + 0] = tgrid0[(i + 1) * (DIM + 2) +   DIM]; // left  <= right
      tgrid0[  (i + 1) * (DIM + 2) + DIM + 1] = tgrid0[(i + 1) * (DIM + 2) +     1]; // right <= left
    }

    do_tile(0, 0, DIM, DIM, 0);
    float *tmp = tgrid1;
    tgrid1 = tgrid0;
    tgrid0 = tmp;
  }

  for (unsigned i = 0; i < DIM; i++)
    for (unsigned j = 0; j < DIM; j++)
      cur_img(i, j) = heat_to_rgb_legacy(tgrid0[(i+1) * (DIM+2) + (j+1)]);
  return 0;

}

// ============================================================================
// ===================================================================== MPI v0
// ============================================================================
#ifdef ENABLE_MPI

#include <mpi.h>

static int mpi_y = -1;
static int mpi_h = -1;
static int mpi_rank = -1;
static int mpi_size = -1;

void heat_init_mpi_v0(void) {
  heat_init_seq_bv2();

  easypap_check_mpi();  // check if MPI was correctly configured

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  mpi_y = mpi_rank * (DIM / mpi_size);
  mpi_h = (DIM / mpi_size);

  PRINT_DEBUG('M', "In charge of slice [%d-%d]\n", mpi_y,
               mpi_y + mpi_h - 1);
}

// ./run -k heat -v mpi_v0 --mpirun "-np 2" -wt bv2 -r 1000 --debug-flags M
// ./run -k heat -v mpi_v0 --mpirun "-np 8" -wt bv2 -i 10000 --no-display
unsigned heat_compute_mpi_v0(unsigned nb_iter) {
  for (unsigned it = 1; it <= nb_iter; it++) {
    // copie des bords verticaux
    for (int i = mpi_y; i < mpi_y + mpi_h; i++) {
      tgrid0[(i + 1) * (DIM + 2)       + 0] = tgrid0[(i + 1) * (DIM + 2) + DIM]; // left  <= right
      tgrid0[(i + 1) * (DIM + 2) + DIM + 1] = tgrid0[(i + 1) * (DIM + 2) +   1]; // right <= left
    }

    do_tile(0, mpi_y, DIM, mpi_h, mpi_rank);

    float *tmp = tgrid1;
    tgrid1 = tgrid0;
    tgrid0 = tmp;

    /**
     * We exchange the order of the send and the recv in order to avoid
     * deadlock.
     */
    if(mpi_rank == 0) {
      // we send the first line to the last process
      MPI_Send(&tgrid0[(DIM+2) + 1], DIM, MPI_FLOAT, mpi_size - 1, 0, MPI_COMM_WORLD);
      // we receive the last line from the last process
      MPI_Recv(&tgrid0[1], DIM, MPI_FLOAT, mpi_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    }else if(mpi_rank == mpi_size - 1) {
      // we receive the first line from the first process
      MPI_Recv(&tgrid0[(DIM+1)*(DIM+2)+1], DIM, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // we send the last line to the first process
      MPI_Send(&tgrid0[(DIM)*(DIM+2) + 1], DIM, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    
  }

  for (int i = mpi_y; i < mpi_y + mpi_h; i++)
    for (int j = 0; j < (int)DIM; j++)
      cur_img(i, j) = heat_to_rgb_legacy(tgrid0[(i + 1) * (DIM + 2) + (j + 1)]);

  return 0;
}

// ============================================================================
// ===================================================================== MPI v1
// ============================================================================

void heat_init_mpi_v1(void) {
  // heat_init_seq_bv2();
  // heat_init_mpi_v0();
}

unsigned heat_compute_mpi_v1(unsigned nb_iter) {
  for (unsigned it = 1; it <= nb_iter; it++) {
    // copie des bords verticaux
    for (int i = mpi_y; i < mpi_y + mpi_h; i++) {
      tgrid0[(i + 1) * (DIM + 2)       + 0] = tgrid0[(i + 1) * (DIM + 2) + DIM]; // left  <= right
      tgrid0[(i + 1) * (DIM + 2) + DIM + 1] = tgrid0[(i + 1) * (DIM + 2) +   1]; // right <= left
    }

    do_tile(0, mpi_y, DIM, mpi_h, 0);

    float *tmp = tgrid1;
    tgrid1 = tgrid0;
    tgrid0 = tmp;

    /* TODO */
  }

  /* TODO */

  return 0;
}

// ============================================================================
// ===================================================================== MPI v2
// ============================================================================

void heat_init_mpi_v2(void) {
  // heat_init_seq_bv2();
  // heat_init_mpi_v0();
}

unsigned heat_compute_mpi_v2(unsigned nb_iter) {
  for (unsigned it = 1; it <= nb_iter; it++) {
    // copie des bords verticaux
    for (int i = mpi_y; i < mpi_y + mpi_h; i++) {
      tgrid0[(i + 1) * (DIM + 2)       + 0] = tgrid0[(i + 1) * (DIM + 2) + DIM]; // left  <= right
      tgrid0[(i + 1) * (DIM + 2) + DIM + 1] = tgrid0[(i + 1) * (DIM + 2) +   1]; // right <= left
    }

    do_tile(0, mpi_y, DIM, mpi_h, 0);

    float *tmp = tgrid1;
    tgrid1 = tgrid0;
    tgrid0 = tmp;

    /* TODO */
  }

  /* TODO */

  return 0;
}

#endif