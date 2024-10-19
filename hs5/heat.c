/**
##############################################################
##############################################################
##############################################################

AUTHORS: MENGQIAN XU (21306077), EDOARDO CARRA' (21400562)
BOARD ID: Q

##############################################################
##############################################################
##############################################################
*/
#include "easypap.h"

static float *tgrid0;
static float *tgrid1;

unsigned heat_to_rgb_legacy(float h) // 0.0 = cold, 1.0 = hot
{
  int i;
  float f, p, q, t;
  float v = 1.0;
  float s = 1.0;

  if (s == 0)
  {
    // achromatic (grey)
    int c = v * 255;
    return ezv_rgba(c, c, c, 255);
  }

  h = (1.0 - h) * 4; // sector 0.0 to 4.0
  if (h == 4.0)
    h = 3.99999;
  i = h;
  f = h - i; // factorial part of h
  p = v * (1 - s);
  q = v * (1 - s * f);
  t = v * (1 - s * (1 - f));

  switch (i)
  {
  case 0:
    return ezv_rgba(v * 255, t * 255, p * 255, 255);
  case 1:
    return ezv_rgba(q * 255, v * 255, p * 255, 255);
  case 2:
    return ezv_rgba(p * 255, v * 255, t * 255, 255);
  default:
    return ezv_rgba(p * 255, q * 255, v * 255, 255);
  }
}

// ============================================================================
// ======================================================================== SEQ
// ============================================================================

void heat_init()
{
  tgrid0 = (float *)malloc(sizeof(float) * DIM * DIM);
  tgrid1 = (float *)malloc(sizeof(float) * DIM * DIM);

  for (unsigned i = 0; i < DIM * DIM; i++)
  {
    tgrid0[i] = 0.f;
    tgrid1[i] = 0.f;
  }

  unsigned ss = DIM / 2; // square size
  unsigned shift = DIM / 16;

  // put an hot spot in the top left corner
  for (unsigned i = 0; i < ss; i++)
    for (unsigned j = 0; j < ss; j++)
      tgrid0[(j + shift) * DIM + i + shift] = 1.f;
}

/** point 1.2
 * // O1
 * D2: 82781.298  ms
 * CA57: 38132.908  ms
 * 
 * // O2
 * D2: 28449.044  ms
 * CA57: 27993.255  ms
 *
 * // O3
 * D2: 19039.446  ms
 * CA57: 24118.710  ms
 * 
 * We run the default version of the heat kernel on the D2 and CA57 architectures,
 * using different optimization level. As expected, we can see that as we
 * increase the optimization level, the performance of the kernel increases.
 * 
 */
int heat_do_tile_default(int x, int y, int width, int height)
{
  float c, l, r, b, t;
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
    {
      c = tgrid0[(i + 0) * DIM + (j + 0)];
      l = (j - 1) >= 0 ? tgrid0[(i + 0) * DIM + (j - 1)] : tgrid0[(i + 0) * DIM + (DIM - 1)];
      r = (j + 1) < (int)DIM ? tgrid0[(i + 0) * DIM + (j + 1)] : tgrid0[(i + 0) * DIM + (0)];
      b = (i - 1) >= 0 ? tgrid0[(i - 1) * DIM + (j + 0)] : tgrid0[(DIM - 1) * DIM + (j + 0)];
      t = (i + 1) < (int)DIM ? tgrid0[(i + 1) * DIM + (j + 0)] : tgrid0[(0) * DIM + (j + 0)];
      tgrid1[i * DIM + j] = (c + l + r + t + b) * 0.2f;
    }

  return 0;
}

unsigned heat_compute_seq(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
  {
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

void heat_init_seq_bv2()
{
  // point 1
  tgrid0 = (float *)malloc(sizeof(float) * (DIM + 2) * (DIM + 2));
  tgrid1 = (float *)malloc(sizeof(float) * (DIM + 2) * (DIM + 2));

  for (unsigned i = 0; i < (DIM + 2) * (DIM + 2); i++)
  {
    tgrid0[i] = 0.f;
    tgrid1[i] = 0.f;
  }

  // point 2
  unsigned ss = DIM / 2; // square size
  unsigned shift = DIM / 16;
  for (unsigned i = 0; i < ss; i++)
    for (unsigned j = 0; j < ss; j++)
      // we skip the first column and row
      tgrid0[((j + 1) + shift) * (DIM + 2) + (i + 1) + shift] = 1.f;
}

int heat_do_tile_bv2(int x, int y, int width, int height)
{
  float c, l, r, b, t;
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
    {
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

/** point 2.1.5
 * // O3
 * D2:   20905.157 ms
 * CA57: 21757.951 ms
 * 
 * In this new version of the kernel we modify the border handling in order 
 * to avoid the ternary operator inside the loop. In order to do so,
 * we add a border of size 1 to the grid, copyting the values of the
 * opposite border to the new border. This way, we can compute the 
 * tile without having to check if we are at the border of the grid.
 * 
 * We can observe a gain of almost 3 seconds on the Cortex-A57 architecture,
 * while on the Denver2 architecture the time is not significantly different.
 * This is in agreement with the discussion done in the first hands-on session,
 * in which we saw that the Cortex-A57 architecture achieve a more significant
 * speedup when we perform these kind of optimizations.
 */
unsigned heat_compute_seq_bv2(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
  {
    // copie des bords
    for (int i = 0; i < (int)DIM; i++)
    {
      tgrid0[0 * (DIM + 2) + i + 1] = tgrid0[DIM * (DIM + 2) + i + 1];         // top   <= bot
      tgrid0[(DIM + 1) * (DIM + 2) + i + 1] = tgrid0[1 * (DIM + 2) + i + 1];   // bot   <= top
      tgrid0[(i + 1) * (DIM + 2) + 0] = tgrid0[(i + 1) * (DIM + 2) + DIM];     // left  <= right
      tgrid0[(i + 1) * (DIM + 2) + DIM + 1] = tgrid0[(i + 1) * (DIM + 2) + 1]; // right <= left
    }

    do_tile(0, 0, DIM, DIM, 0);
    float *tmp = tgrid1;
    tgrid1 = tgrid0;
    tgrid0 = tmp;
  }

  for (unsigned i = 0; i < DIM; i++)
    for (unsigned j = 0; j < DIM; j++)
      cur_img(i, j) = heat_to_rgb_legacy(tgrid0[(i + 1) * (DIM + 2) + (j + 1)]);
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

void heat_init_mpi_v0(void)
{
  heat_init_seq_bv2();

  easypap_check_mpi(); // check if MPI was correctly configured

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  mpi_y = mpi_rank * (DIM / mpi_size);
  mpi_h = (DIM / mpi_size);

  PRINT_DEBUG('M', "In charge of slice [%d-%d]\n", mpi_y,
              mpi_y + mpi_h - 1);
}


/** point 2.2.5
 * // O3
 * Running the first version of the MPI kernel, we obtain a significant
 * speedup on both the Denver2 and Cortex-A57 architectures, respectivelly
 * x1.55 and x1.975. This is expected, as we are now using two processes
 * to compute the heat kernel, and in a theoric scenario we would obtain
 * a speedup of 2.
 * 
 * D2: 13442.008  ms
 * CA57: 11011.552  ms
 */
// For Denver2 ./run -k heat --no-display -r 500 -i 5000 --mpirun "--cpu-set 4,5 --bind-to core --report-bindings -n 2" -si -v mpi_v0 -wt bv2
// For CortexA57 ./run -k heat --no-display -r 500 -i 5000 --mpirun "--cpu-set 0,1 --bind-to core --report-bindings -n 2" -si -v mpi_v0 -wt bv2 
unsigned heat_compute_mpi_v0(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
  {
    // copie des bords verticaux
    for (int i = mpi_y; i < mpi_y + mpi_h; i++)
    {
      tgrid0[(i + 1) * (DIM + 2) + 0] = tgrid0[(i + 1) * (DIM + 2) + DIM];     // left  <= right
      tgrid0[(i + 1) * (DIM + 2) + DIM + 1] = tgrid0[(i + 1) * (DIM + 2) + 1]; // right <= left
    }

    do_tile(0, mpi_y, DIM, mpi_h, mpi_rank);

    float *tmp = tgrid1;
    tgrid1 = tgrid0;
    tgrid0 = tmp;

    // point 3
    /**
     * We exchange the order of the send and the recv in order to avoid
     * deadlock.
     */
    if (mpi_rank == 0)
    {
      // we send the first line of the tile to the last process
      MPI_Send(&tgrid0[(DIM + 2) + 1], DIM, MPI_FLOAT, mpi_size - 1, 0, MPI_COMM_WORLD);
      // we receive the last line of the tile from the last process
      MPI_Recv(&tgrid0[1], DIM, MPI_FLOAT, mpi_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // we send the last line of the tile to the last process
      MPI_Send(&tgrid0[mpi_h * (DIM + 2) + 1], DIM, MPI_FLOAT, mpi_size - 1, 0, MPI_COMM_WORLD);
      // we receive the first line of the tile from the last process
      MPI_Recv(&tgrid0[(mpi_h + 1) * (DIM + 2) + 1], DIM, MPI_FLOAT, mpi_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if (mpi_rank == mpi_size - 1)
    {
      // we receive the first line of the tile from the first process
      MPI_Recv(&tgrid0[(DIM + 1) * (DIM + 2) + 1], DIM, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // we send the last line of the tile to the first process
      MPI_Send(&tgrid0[(DIM) * (DIM + 2) + 1], DIM, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

      // we recv the last line of the tile from the first process
      MPI_Recv(&tgrid0[(mpi_y) * (DIM + 2) + 1], DIM, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // we send the first line of the tile to the first process
      MPI_Send(&tgrid0[(mpi_y + 1) * (DIM + 2) + 1], DIM, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
  }

  for (int i = mpi_y; i < mpi_y + mpi_h; i++)
    for (int j = 0; j < (int)DIM; j++)
      cur_img(i, j) = heat_to_rgb_legacy(tgrid0[(i + 1) * (DIM + 2) + (j + 1)]);

  // point 4
  MPI_Gather(&image[mpi_y * (DIM)], mpi_h * DIM, MPI_UNSIGNED, image, mpi_h * DIM, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  return 0;
}

// ============================================================================
// ===================================================================== MPI v1
// ============================================================================

void heat_init_mpi_v1(void)
{
  heat_init_seq_bv2();
  heat_init_mpi_v0();
}

/** point 2.3.3
 * 
 * In this version of the MPI kernel, we generalize to 2^n processes the
 * computation of the heat kernel. As the number of processes increases,
 * the size of the tile that each process has to compute decreases, but we
 * increase the number of communications between processes. In fact, every process 
 * has to send and receive the first and last line of the tile to the previous and
 * subsequent process. This increases the number of communications, but since 
 * we are using processes in the same node, the latency of the communication is
 * very low, and the performance of the kernel increases.
 * 
 * // O3
 *  
 * cpu,process,time
 * cortex,1,21234.540
 * cortex,2,10826.735
 * cortex,4,5694.706
 * denver,1,23449.509
 * denver,2,12391.556
 * full_system,4,6425.335
 * full_system,8,7338.345
 * 
 * Please note that when running the kernel on the full system using 4 processes,
 * the performance is not as good as when running the kernel on the Cortex-A57
 * architecture using 4 processes. This is due to the fact that the communication
 * between the different processors is slower than the communication between the different
 * processes in the same processor. Moreover, when using 8 processes on the full system,
 * we can see that the performance is worse than when using 4 processes. This is due
 * to the fact that the architecture supports at most 6 processes with no overload 
 * on the cores, and when we use 8 processes, the kernel has to share the cores with
 * other processes.
 */
unsigned heat_compute_mpi_v1(unsigned nb_iter)
{
  int dest, source;
  for (unsigned it = 1; it <= nb_iter; it++)
  {
    // copie des bords verticaux
    for (int i = mpi_y; i < mpi_y + mpi_h; i++)
    {
      tgrid0[(i + 1) * (DIM + 2) + 0] = tgrid0[(i + 1) * (DIM + 2) + DIM];     // left  <= right
      tgrid0[(i + 1) * (DIM + 2) + DIM + 1] = tgrid0[(i + 1) * (DIM + 2) + 1]; // right <= left
    }

    do_tile(0, mpi_y, DIM, mpi_h, mpi_rank);

    float *tmp = tgrid1;
    tgrid1 = tgrid0;
    tgrid0 = tmp;

    dest = (mpi_rank == mpi_size - 1) ? 0 : (mpi_rank + 1);
    source = (mpi_rank == 0) ? mpi_size - 1 : (mpi_rank - 1);
    // sending the last line of the process to the subsequent process 
    // and receiving the last line of the previous process
    MPI_Sendrecv(&tgrid0[(mpi_y + mpi_h) * (DIM + 2) + 1], DIM, MPI_FLOAT, dest, 0,
                  &tgrid0[(mpi_y) * (DIM + 2) + 1], DIM, MPI_FLOAT, source , 0,
                  MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    dest = (mpi_rank == 0) ? mpi_size - 1 : (mpi_rank - 1);
    source = (mpi_rank == mpi_size - 1) ? 0 : (mpi_rank + 1);
    // sending the first line of the process to the previous process
    // and receiving the first line of the subsequent process
    MPI_Sendrecv(&tgrid0[(mpi_y + 1) * (DIM + 2) + 1], DIM, MPI_FLOAT, dest, 0,
                  &tgrid0[(mpi_y + 1 + mpi_h) * (DIM + 2) + 1], DIM, MPI_FLOAT, source, 0,
                  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  for (int i = mpi_y; i < mpi_y + mpi_h; i++)
    for (int j = 0; j < (int)DIM; j++)
      cur_img(i, j) = heat_to_rgb_legacy(tgrid0[(i + 1) * (DIM + 2) + (j + 1)]);
  // point 4
  MPI_Gather(&image[mpi_y * (DIM)], mpi_h * DIM, MPI_UNSIGNED, image, mpi_h * DIM, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  return 0;
}

// ============================================================================
// ===================================================================== MPI v2
// ============================================================================

void heat_init_mpi_v2(void)
{
  heat_init_seq_bv2();
  heat_init_mpi_v0();
}

/** point 3.4.2
 * Non-blocking communication allows the process to continue its computation
 * while the communication is being performed. This way, we can overlap the
 * computation and the communication, and increase the performance of the kernel.
 * In this version of the kernel, the computation of the tile cannot be overlapped
 * with the communication, as we need the values of the first and last line of the
 * tile to do the computation. So we don't expect a significant speedup when using
 * non-blocking communication.
 * 
 * cpu,process,time
 * cortex,1,21272.572
 * cortex,2,11000.392
 * cortex,4,5639.650
 * denver,1,23492.910
 * denver,2,12394.926
 * full_system,4,6395.091
 * full_system,8,7340.817
 * 
 * As we can see, the time is very similar to the one obtained using blocking
 * communication. Moreover, since we are using one node, the latency of the
 * communication is very low, and the performance of the kernel is not significantly
 * affected by the use of non-blocking communication.
 */

unsigned heat_compute_mpi_v2(unsigned nb_iter)
{
  int dest, source;
  MPI_Request requests[4];

  for (unsigned it = 1; it <= nb_iter; it++)
  {
    // copie des bords verticaux
    for (int i = mpi_y; i < mpi_y + mpi_h; i++)
    {
      tgrid0[(i + 1) * (DIM + 2) + 0] = tgrid0[(i + 1) * (DIM + 2) + DIM];     // left  <= right
      tgrid0[(i + 1) * (DIM + 2) + DIM + 1] = tgrid0[(i + 1) * (DIM + 2) + 1]; // right <= left
    }

    do_tile(0, mpi_y, DIM, mpi_h, mpi_rank);

    float *tmp = tgrid1;
    tgrid1 = tgrid0;
    tgrid0 = tmp;

    dest = (mpi_rank == mpi_size - 1) ? 0 : (mpi_rank + 1);
    source = (mpi_rank == 0) ? mpi_size - 1 : (mpi_rank - 1);
    
    // sending the last line of the process to the subsequent process 
    // and receiving the last line of the previous process
    MPI_Isend(&tgrid0[(mpi_y + mpi_h) * (DIM + 2) + 1], DIM, MPI_FLOAT, dest, 0, 
              MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(&tgrid0[(mpi_y) * (DIM + 2) + 1], DIM, MPI_FLOAT, source , 0,
                  MPI_COMM_WORLD, &requests[1]);

    dest = (mpi_rank == 0) ? mpi_size - 1 : (mpi_rank - 1);
    source = (mpi_rank == mpi_size - 1) ? 0 : (mpi_rank + 1);
    // sending the first line of the process to the previous process
    // and receiving the first line of the subsequent process
    MPI_Isend(&tgrid0[(mpi_y + 1) * (DIM + 2) + 1], DIM, MPI_FLOAT, dest, 0,
              MPI_COMM_WORLD, &requests[2]);
    MPI_Irecv(&tgrid0[(mpi_y + 1 + mpi_h) * (DIM + 2) + 1], DIM, MPI_FLOAT, source, 0,
              MPI_COMM_WORLD, &requests[3]);

    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
  }

  for (int i = mpi_y; i < mpi_y + mpi_h; i++)
    for (int j = 0; j < (int)DIM; j++)
      cur_img(i, j) = heat_to_rgb_legacy(tgrid0[(i + 1) * (DIM + 2) + (j + 1)]);
  // point 4
  MPI_Gather(&image[mpi_y * (DIM)], mpi_h * DIM, MPI_UNSIGNED, image, mpi_h * DIM, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  return 0;
}

#endif