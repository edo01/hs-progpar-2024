#include <math.h>
#include <omp.h>

#include "global.h"
#include "img_data.h"
#include "cppdefs.h"
#include "mipp.h"

static void rotate(void);

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// --REMOVE MEEEE--
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
#define ENABLE_VECTO

#ifdef ENABLE_VECTO
#include <iostream>
static bool is_printed = false;
static void print_simd_info() {
  if (!is_printed) {
    std::cout << "SIMD infos:" << std::endl;
    std::cout << " - Instr. type:       " << mipp::InstructionType << std::endl;
    std::cout << " - Instr. full type:  " << mipp::InstructionFullType << std::endl;
    std::cout << " - Instr. version:    " << mipp::InstructionVersion << std::endl;
    std::cout << " - Instr. size:       " << mipp::RegisterSizeBit << " bits"
              << std::endl;
    std::cout << " - Instr. lanes:      " << mipp::Lanes << std::endl;
    std::cout << " - 64-bit support:    " << (mipp::Support64Bit ? "yes" : "no")
              << std::endl;
    std::cout << " - Byte/word support: " << (mipp::SupportByteWord ? "yes" : "no")
              << std::endl;
    auto ext = mipp::InstructionExtensions();
    if (ext.size() > 0) {
      std::cout << " - Instr. extensions: {";
      for (auto i = 0; i < (int)ext.size(); i++)
        std::cout << ext[i] << (i < ((int)ext.size() - 1) ? ", " : "");
      std::cout << "}" << std::endl;
    }
    std::cout << std::endl;
    is_printed = true;
  }
}
#endif

// Global variables
static float base_angle = 0.f;
static int color_a_r = 255, color_a_g = 255, color_a_b = 0, color_a_a = 255;
static int color_b_r = 0, color_b_g = 0, color_b_b = 255, color_b_a = 255;

// for 2.6 -> removing constansts
static mipp::Reg<float> r_1 = mipp::Reg<float>(1.f);
static mipp::Reg<float> r_const_2 = mipp::Reg<float>(0.273f);
static mipp::Reg<float> r_zero = mipp::Reg<float>(0.f);



// ----------------------------------------------------------------------------
// -------------------------------------------------------- INITIAL SEQ VERSION
// ----------------------------------------------------------------------------

// The image is a two-dimension array of size of DIM x DIM. Each pixel is of
// type 'unsigned' and store the color information following a RGBA layout (4
// bytes). Pixel at line 'l' and column 'c' in the current image can be accessed
// using cur_img (l, c).

// The kernel returns 0, or the iteration step at which computation has
// completed (e.g. stabilized).

// Computation of one pixel
static unsigned compute_color(int i, int j) {
  float atan2f_in1 = (float)DIM / 2.f - (float)i;
  float atan2f_in2 = (float)j - (float)DIM / 2.f;
  float angle = atan2f(atan2f_in1, atan2f_in2) + M_PI + base_angle;

  float ratio = fabsf((fmodf(angle, M_PI / 4.f) - (M_PI / 8.f)) / (M_PI / 8.f));

  int r = color_a_r * ratio + color_b_r * (1.f - ratio);
  int g = color_a_g * ratio + color_b_g * (1.f - ratio);
  int b = color_a_b * ratio + color_b_b * (1.f - ratio);
  int a = color_a_a * ratio + color_b_a * (1.f - ratio);

  return ezv_rgba(r, g, b, a);
}

static void rotate(void) {
  base_angle = fmodf(base_angle + (1.f / 180.f) * M_PI, M_PI);
}

///////////////////////////// Simple sequential version (seq)
/**
 * -O0
 *    D2:   17990.973 ms
 *    CA57: 21297.301 ms
 * -O1
 *    D2:   13539.644 ms
 *    CA57: 14678.651 ms
 * 
 * -O2
 *    D2:   14734.991 ms
 *    CA57: 13464.836 ms
 * 
 * -O3
 *    D2:   14424.248 ms
 *    CA57: 13434.660 ms
 *
 * -O3 -ffast-math
 *    D2: 11328.200  ms
 *    CA57: 11868.157  ms
 *
 */
EXTERN unsigned spin_compute_seq(unsigned nb_iter) {
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++)
      for (unsigned j = 0; j < DIM; j++)
        cur_img(i, j) = compute_color(i, j);

    rotate();  // Slightly increase the base angle
  }

  return 0;
}

// ----------------------------------------------------------------------------
// --------------------------------------------------------- APPROX SEQ VERSION
// ----------------------------------------------------------------------------

static float atanf_approx(float x) {
  return x * M_PI / 4.f + 0.273f * x * (1.f - fabsf(x));
}

static float atan2f_approx(float y, float x) {
  float ay = fabsf(y);
  float ax = fabsf(x);
  int invert = ay > ax;
  float z = invert ? ax / ay : ay / ax; // [0,1]
  float th = atanf_approx(z); // [0,pi/4]
  if (invert) th = M_PI_2 - th; // [0,pi/2]
  if (x < 0) th = M_PI - th; // [0,pi]
  if (y < 0) th = -th;
  return th;
}

static float fmodf_approx(float x, float y) {
  return x - trunc(x / y) * y;
}

// Computation of one pixel
static unsigned compute_color_approx(int i, int j) {
  float atan2f_in1 = (float)DIM / 2.f - (float)i;
  float atan2f_in2 = (float)j - (float)DIM / 2.f;
  float angle = atan2f_approx(atan2f_in1, atan2f_in2) + M_PI + base_angle;

  float ratio = fabsf((fmodf_approx(angle, M_PI / 4.f) - (M_PI / 8.f)) / (M_PI / 8.f));

  int r = color_a_r * ratio + color_b_r * (1.f - ratio);
  int g = color_a_g * ratio + color_b_g * (1.f - ratio);
  int b = color_a_b * ratio + color_b_b * (1.f - ratio);
  int a = color_a_a * ratio + color_b_a * (1.f - ratio);

  return ezv_rgba(r, g, b, a);
}


/*
 * D2: 4089.862  ms
 * CA57: 5423.462  ms
 */
EXTERN unsigned spin_compute_approx(unsigned nb_iter) {
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++)
      for (unsigned j = 0; j < DIM; j++)
        cur_img(i, j) = compute_color_approx(i, j);

    rotate();  // Slightly increase the base angle
  }

  return 0;
}

#ifdef ENABLE_VECTO
// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 0
// ----------------------------------------------------------------------------

/**
 * This optimization is the prelude to the SIMD versions. The function accepts
 * two MIPP registers of integers, r_i and r_j, and computes the color of the
 * pixels without leveraging SIMD instructions. For all the pixels in the 
 * registers, the value is extracted and the color is computed. 
 * 
 * We expect this version to be slower or equal to the approx version, 
 * since it is basically the same code but with the overhead of the MIPP.
 * 
 * 
 * D2: 8466.470  ms
 * CA57: 6398.785  ms
 * 
 */
static mipp::Reg<int> compute_color_simd_v0(mipp::Reg<int> r_i,
                                            mipp::Reg<int> r_j)
{
  int result[mipp::N<int>()]; // array of final pixels
  int i, j, r, g, b, a;
  float atan2f_in1, atan2f_in2, angle, ratio;

  for(int index=0; index<mipp::N<int>(); index++)
  {
    i = r_i[index];
    j = r_j[index];
    atan2f_in1 = (float)DIM / 2.f - (float)i;
    atan2f_in2 = (float)j - (float)DIM / 2.f;
    angle = atan2f_approx(atan2f_in1, atan2f_in2) + M_PI + base_angle;

    ratio = fabsf((fmodf_approx(angle, M_PI / 4.f) - (M_PI / 8.f)) / (M_PI / 8.f));

    r = color_a_r * ratio + color_b_r * (1.f - ratio);
    g = color_a_g * ratio + color_b_g * (1.f - ratio);
    b = color_a_b * ratio + color_b_b * (1.f - ratio);
    a = color_a_a * ratio + color_b_a * (1.f - ratio);

    result[index] = ezv_rgba(r, g, b, a);
  }
  return mipp::Reg<int>(result);
}

EXTERN unsigned spin_compute_simd_v0(unsigned nb_iter) {
  print_simd_info();
  int tab_j[mipp::N<int>()];

  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++)
      for (unsigned j = 0; j < DIM; j += mipp::N<float>()) {
        for (unsigned jj = 0; jj < mipp::N<float>(); jj++) tab_j[jj] = j + jj;
        int* img_out_ptr = (int*)&cur_img(i, j);
        mipp::Reg<int> r_result =
            compute_color_simd_v0(mipp::Reg<int>(i), mipp::Reg<int>(tab_j));

        r_result.store(img_out_ptr);
      }

    rotate();  // Slightly increase the base angle
  }

  return 0;
}

// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 1
// ----------------------------------------------------------------------------

// simd version of the fmodf_approx function
static inline mipp::Reg<float> fmodf_approx_simd(mipp::Reg<float> r_x,
                                                 mipp::Reg<float> r_y) {
  return r_x - mipp::trunc(r_x / r_y) * r_y;
}

// Computation of one pixel
static inline mipp::Reg<int> compute_color_simd_v1(mipp::Reg<int> r_i, 
                                                  mipp::Reg<int> r_j)
{
  int result[mipp::N<int>()];
  float angles[mipp::N<float>()];
  int i, j;
  float atan2f_in1, atan2f_in2;
  int r, g, b, a;

  mipp::Reg<float> r_ratio;

  for(int index=0; index<mipp::N<int>(); index++)
  {
    i = r_i[index];
    j = r_j[index];
    atan2f_in1 = (float)DIM / 2.f - (float)i;
    atan2f_in2 = (float)j - (float)DIM / 2.f;
    angles[index] = atan2f_approx(atan2f_in1, atan2f_in2) + M_PI + base_angle;
  }

  // simd version of the ratio computation
  r_ratio = mipp::abs((fmodf_approx_simd(mipp::Reg<float>(angles),
        mipp::Reg<float>(M_PI / 4.f)) - mipp::Reg<float>(M_PI / 8.f)) / (M_PI / 8.f));

  for(int index=0; index<mipp::N<int>(); index++)
  {
    r = color_a_r * r_ratio[index] + color_b_r * (1.f - r_ratio[index]);
    g = color_a_g * r_ratio[index] + color_b_g * (1.f - r_ratio[index]);
    b = color_a_b * r_ratio[index] + color_b_b * (1.f - r_ratio[index]);
    a = color_a_a * r_ratio[index] + color_b_a * (1.f - r_ratio[index]);

    result[index] = ezv_rgba(r, g, b, a);
  }
  return mipp::Reg<int>(result);
}

/*
* In this version, we start using SIMD instructions. In particular, we vectorize
* the function of the fmodf_approx_simd, and so of the computation of the ratio.
* The rest of the code is the same as the previous version, so we expect a
* slight improvement in performance compared to the previous version. If compared
* to the approx version, the overhead of creating and access the registers may
* still make this version slower.
*
* D2: 6156.411  ms 
* CA57: 4579.960  ms
* 
* This version shows a significant improvement compared to the previous one,
* decreasing the execution time by a factor of 1.375 on Denver2 and 1.397 on 
* Cortex-A57. But in the case of Denver2, the execution time is still higher
* than the approx version.

* Note: for readability, we used the overloaded operator -,+,*,/ instead of the
* functions mipp::sub, mipp::add, mipp::mul, mipp::div.
*/
EXTERN unsigned spin_compute_simd_v1(unsigned nb_iter) {
  int tab_j[mipp::N<int>()];
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++)
      for (unsigned j = 0; j < DIM; j += mipp::N<float>()) {
        for (unsigned jj = 0; jj < mipp::N<float>(); jj++) tab_j[jj] = j + jj;
        int* img_out_ptr = (int*)&cur_img(i, j);
        mipp::Reg<int> r_result =
            compute_color_simd_v1(mipp::Reg<int>(i), mipp::Reg<int>(tab_j));

        r_result.store(img_out_ptr);
      }

    rotate();  // Slightly increase the base angle
  }
  return 0;
}

// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 2
// ----------------------------------------------------------------------------

static inline mipp::Reg<int> rgba_simd(mipp::Reg<int> r_r, mipp::Reg<int> r_g,
                                       mipp::Reg<int> r_b, mipp::Reg<int> r_a) 
{
  // << and | are overloaded operators from mipp
  return r_r  | r_g << 8 | r_b << 16 | r_a << 24;
}

static inline mipp::Reg<int> compute_color_simd_v2(mipp::Reg<int> r_i,
                                                   mipp::Reg<int> r_j)
{
  float angles[mipp::N<float>()];
  int i, j;
  float atan2f_in1, atan2f_in2;

  mipp::Reg<int> r_r, r_g, r_b, r_a;
  mipp::Reg<float> r_ratio;


  for(int index=0; index<mipp::N<int>(); index++)
  {
    i = r_i[index];
    j = r_j[index];
    atan2f_in1 = (float)DIM / 2.f - (float)i;
    atan2f_in2 = (float)j - (float)DIM / 2.f;
    angles[index] = atan2f_approx(atan2f_in1, atan2f_in2) + M_PI + base_angle;
  }

  /**
   * Please note that in order to enhance the readability of the code, we avoided
   * the explicit use of the mipp constructors. Instead, we used the overloaded
   * operators +, -, *, /.
   */
  // simd version of the ratio computation
  r_ratio = mipp::abs((fmodf_approx_simd(mipp::Reg<float>(angles), mipp::Reg<float>(M_PI / 4.f)) -
    (M_PI / 8.f)) / (M_PI / 8.f));
  
  // simd version of the color computation
  r_r = mipp::cvt<float,int>(r_ratio * color_a_r + (mipp::Reg<float>(1.f) - r_ratio) * color_b_r);
  r_g = mipp::cvt<float,int>(r_ratio * color_a_g + (mipp::Reg<float>(1.f) - r_ratio) * color_b_g);
  r_b = mipp::cvt<float,int>(r_ratio * color_a_b + (mipp::Reg<float>(1.f) - r_ratio) * color_b_b);
  r_a = mipp::cvt<float,int>(r_ratio * color_a_a + (mipp::Reg<float>(1.f) - r_ratio) * color_b_a);

  return rgba_simd(r_r, r_g, r_b, r_a);
}

/**
 * In this version, we further optimize the code by vectorizing the computation
 * of the color of the pixels. Now almost 2/3 of the code is vectorized, and we
 * expect some performance improvement compared to the previous version. Still,
 * one third of the code is not vectorized, and the overhead of creating and
 * accessing the registers may still make this version slower than the approx.
 * 
 * D2: 5700.988  ms
 * CA57: 4384.575  ms
 * 
 * In the case of Denver2, the execution time is still higher than the approx
 * version, but the performance gap is reduced. On the other hand, the execution
 * time on Cortex-A57 is lower than the simd_v1 version.
 */
EXTERN unsigned spin_compute_simd_v2(unsigned nb_iter) {
  int tab_j[mipp::N<int>()];
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++)
      for (unsigned j = 0; j < DIM; j += mipp::N<float>()) {
        for (unsigned jj = 0; jj < mipp::N<float>(); jj++) tab_j[jj] = j + jj;
        int* img_out_ptr = (int*)&cur_img(i, j);
        mipp::Reg<int> r_result =
            compute_color_simd_v2(mipp::Reg<int>(i), mipp::Reg<int>(tab_j));

        r_result.store(img_out_ptr);
      }

    rotate();  // Slightly increase the base angle
  }
  return 0;
}

// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 3
// ----------------------------------------------------------------------------

// simd version of the atanf_approx function
static inline mipp::Reg<float> atanf_approx_simd(mipp::Reg<float> r_z) {
  return r_z * M_PI / 4.f + r_z * 0.273f * (mipp::Reg<float>(1.f) - mipp::abs(r_z));
}

static inline mipp::Reg<float> atan2f_approx_simd(mipp::Reg<float> r_y,
                                                  mipp::Reg<float> r_x) {
  mipp::Reg<float> r_ay = mipp::abs(r_y);
  mipp::Reg<float> r_ax = mipp::abs(r_x);

  // creating a mask to check if r_ay > r_ax
  mipp::Msk<mipp::N<float>()> invert = r_ay > r_ax;
  
  // conditional selection of the value following the mask
  mipp::Reg<float> r_z = mipp::blend(r_ax/r_ay, r_ay/r_ax, invert);

  mipp::Reg<float> r_th = atanf_approx_simd(r_z);
  r_th = mipp::blend(- r_th + M_PI_2, r_th, invert);
  r_th = mipp::blend(- r_th + M_PI, r_th, r_x < 0.f);
  r_th = mipp::blend(-r_th, r_th, r_y < 0.f);

  return r_th;
}

static inline mipp::Reg<int> compute_color_simd_v3(mipp::Reg<int> r_i,
                                                   mipp::Reg<int> r_j)
{
  mipp::Reg<int> r_r, r_g, r_b, r_a;
  mipp::Reg<float> r_atan2f_in1, r_atan2f_in2, r_angles, r_ratio;

  // simd version of the angle computation
  r_atan2f_in1 = mipp::Reg<float>(DIM / 2.f) - mipp::cvt<int,float>(r_i);
  r_atan2f_in2 = mipp::cvt<int,float>(r_j) - mipp::Reg<float>(DIM / 2.f);
  r_angles = atan2f_approx_simd(r_atan2f_in1, r_atan2f_in2) + mipp::Reg<float>(M_PI + base_angle);
  
  // simd version of the ratio computation
  r_ratio = mipp::abs((fmodf_approx_simd(r_angles, mipp::Reg<float>(M_PI / 4.f)) -
    (M_PI / 8.f)) / (M_PI / 8.f));
  
  // simd version of the color computation
  r_r = mipp::cvt<float,int>(r_ratio * color_a_r + (mipp::Reg<float>(1.f) - r_ratio) * color_b_r);
  r_g = mipp::cvt<float,int>(r_ratio * color_a_g + (mipp::Reg<float>(1.f) - r_ratio) * color_b_g);
  r_b = mipp::cvt<float,int>(r_ratio * color_a_b + (mipp::Reg<float>(1.f) - r_ratio) * color_b_b);
  r_a = mipp::cvt<float,int>(r_ratio * color_a_a + (mipp::Reg<float>(1.f) - r_ratio) * color_b_a);
  
  return rgba_simd(r_r, r_g, r_b, r_a);
}

/**
 * Finally, we vectorize the entire computation of the color of the pixels. Now
 * the entire code is vectorized, and we expect a significant performance
 * improvement compared to the previous versions, even compared to the approx version.
 * 
 * Both Denver2 and Cortex-A57 support NEONv2 extensions. Both have 128-bit instruction 
 * size, capable of processing 4 floats/integer at a time. Therefore, we expect a
 * theoretical speedup of 4x compared to the sequential version (only one fetching, decode, 
 * computate and write back phases instead of 4).
 * 
 * D2: 1061.368  ms
 * CA57: 2936.885  ms
 * 
 * The execution time on Denver2 meets the expectations, showing a speedup of almost exactly
 * 4x compared to the sequential version. On the other hand, the execution time on Cortex-A57
 * is higher than expected, showing a speedup of only 1.847x compared to the sequential version.
 * This is may be due to a different implementation of the NEONv2 extensions on the Cortex-A57.
 */
EXTERN unsigned spin_compute_simd_v3(unsigned nb_iter) {
  int tab_j[mipp::N<int>()];
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++)
      for (unsigned j = 0; j < DIM; j += mipp::N<float>()) {
        for (unsigned jj = 0; jj < mipp::N<float>(); jj++) tab_j[jj] = j + jj;
        int* img_out_ptr = (int*)&cur_img(i, j);
        mipp::Reg<int> r_result =
            compute_color_simd_v3(mipp::Reg<int>(i), mipp::Reg<int>(tab_j));

        r_result.store(img_out_ptr);
      }

    rotate();  // Slightly increase the base angle
  }
  return 0;
}

// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 4
// ----------------------------------------------------------------------------

// Computation of one pixel
/*static inline mipp::Reg<int> compute_color_simd_v4(mipp::Reg<int> r_i,
                                                   mipp::Reg<int> r_j)
{
  mipp::Reg<float> r_atan2f_in1 = mipp::Reg<float>(DIM / 2.f) - mipp::cvt<int,float>(r_i);
  mipp::Reg<float> r_atan2f_in2 = mipp::cvt<int,float>(r_j) - mipp::Reg<float>(DIM / 2.f);
  
  //atan2f_approx
  mipp::Reg<float> r_ay = mipp::abs(r_atan2f_in1);
  mipp::Reg<float> r_ax = mipp::abs(r_atan2f_in2);
  mipp::Msk<mipp::N<float>()> invert = r_ay > r_ax;

  mipp::Reg<float> r_z = mipp::blend(r_ax/r_ay, r_ay/r_ax, invert);

  mipp::Reg<float> r_th = r_z * M_PI / 4.f + r_z * 0.273f * (mipp::Reg<float>(1.f) - mipp::abs(r_z)); // [0,pi/4]
  r_th = mipp::blend(mipp::Reg<float>(M_PI_2) - r_th, r_th, invert); // [0,pi/2]
  r_th = mipp::blend(mipp::Reg<float>(M_PI) - r_th, r_th, r_atan2f_in2 < mipp::Reg<float>(0.f)); // [0,pi]
  r_th = mipp::blend(-r_th, r_th, r_atan2f_in1 < mipp::Reg<float>(0.f));

  mipp::Reg<float> angles = r_th + mipp::Reg<float>(M_PI + base_angle);
  mipp::Reg<float> ratio = mipp::abs((fmodf_approx_simd(mipp::Reg<float>(angles),
        mipp::Reg<float>(M_PI / 4.f)) - mipp::Reg<float>(M_PI / 8.f)) / (M_PI / 8.f));

  mipp::Reg<int> r_r = mipp::cvt<float,int>(ratio * color_a_r + (mipp::Reg<float>(1.f) - ratio) * color_b_r);
  mipp::Reg<int> r_g = mipp::cvt<float,int>(ratio * color_a_g + (mipp::Reg<float>(1.f) - ratio) * color_b_g);
  mipp::Reg<int> r_b = mipp::cvt<float,int>(ratio * color_a_b + (mipp::Reg<float>(1.f) - ratio) * color_b_b);
  mipp::Reg<int> r_a = mipp::cvt<float,int>(ratio * color_a_a + (mipp::Reg<float>(1.f) - ratio) * color_b_a);
  
  return rgba_simd(r_r, r_g, r_b, r_a);
}*/

// Computation of one pixel
static inline mipp::Reg<int> compute_color_simd_v4(mipp::Reg<int> r_i,
                                                   mipp::Reg<int> r_j)
{

  mipp::Reg<float> r_dim = mipp::Reg<float>(DIM/2.f); 
  mipp::Reg<float> r_pi_div_4 = mipp::Reg<float>(M_PI / 4.f);  
  mipp::Reg<float> r_pi_div_8 = mipp::Reg<float>(M_PI / 8.f); 
  mipp::Reg<float> r_pi = mipp::Reg<float>(M_PI); 
  mipp::Reg<float> r_pi_2 = mipp::Reg<float>(M_PI_2);
  
  mipp::Reg<float> r_base_angle = mipp::Reg<float>(base_angle);

  mipp::Reg<float> r_atan2f_in1 = r_dim - mipp::cvt<int,float>(r_i);
  mipp::Reg<float> r_atan2f_in2 = mipp::cvt<int,float>(r_j) - r_dim;

  // atan2f_approx_simd 
  mipp::Reg<float> r_abs_y = mipp::abs(r_atan2f_in1);
  mipp::Reg<float> r_abs_x = mipp::abs(r_atan2f_in2);
  mipp::Msk<mipp::N<float>()> invert_mask = r_abs_y > r_abs_x;

  mipp::Reg<float> r_z = mipp::blend(r_abs_x / r_abs_y, r_abs_y / r_abs_x, invert_mask);

  // atanf_approx_simd
  mipp::Reg<float> r_atan_approx = r_z * r_pi_div_4 + r_const_2 * r_z * (r_1 - mipp::abs(r_z));
  r_atan_approx = mipp::blend(r_pi_2 - r_atan_approx, r_atan_approx, invert_mask); 

  // x < 0 
  mipp::Msk<mipp::N<float>()> x_negative_mask = r_atan2f_in2 < r_zero;
  r_atan_approx = mipp::blend(r_pi - r_atan_approx, r_atan_approx, x_negative_mask); 

  // y < 0 
  mipp::Msk<mipp::N<float>()> y_negative_mask = r_atan2f_in1 < r_zero;
  r_atan_approx = mipp::blend(-r_atan_approx, r_atan_approx, y_negative_mask);

  mipp::Reg<float> r_angle = r_atan_approx + r_pi + r_base_angle;

  mipp::Reg<float> r_ratio = mipp::abs((fmodf_approx_simd(r_angle, r_pi_div_4) - r_pi_div_8) / r_pi_div_8);
  
  
  mipp::Reg<int> r_r = mipp::cvt<float,int>(r_ratio * color_a_r + (r_1 - r_ratio) * color_b_r);
  mipp::Reg<int> r_g = mipp::cvt<float,int>(r_ratio * color_a_g + (r_1 - r_ratio) * color_b_g);
  mipp::Reg<int> r_b = mipp::cvt<float,int>(r_ratio * color_a_b + (r_1 - r_ratio) * color_b_b);
  mipp::Reg<int> r_a = mipp::cvt<float,int>(r_ratio * color_a_a + (r_1 - r_ratio) * color_b_a);

  return rgba_simd(r_r, r_g, r_b, r_a);

}

/**
  * 
  * D2: 1068.010  ms
  * CA57: 2934.604  ms
  */
EXTERN unsigned spin_compute_simd_v4(unsigned nb_iter) {

  int tab_j[mipp::N<int>()];
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++)
      for (unsigned j = 0; j < DIM; j += mipp::N<float>()) {
        for (unsigned jj = 0; jj < mipp::N<float>(); jj++) tab_j[jj] = j + jj;
        int* img_out_ptr = (int*)&cur_img(i, j);
        mipp::Reg<int> r_result =
            compute_color_simd_v4(mipp::Reg<int>(i), mipp::Reg<int>(tab_j));

        r_result.store(img_out_ptr);
      }

    rotate();  // Slightly increase the base angle
  }
  return 0;
}

// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 5
// ----------------------------------------------------------------------------

EXTERN unsigned spin_compute_simd_v5(unsigned nb_iter) {
  mipp::Reg<float> r_dim = DIM / 2.f;  
  mipp::Reg<float> r_pi_div_4 = M_PI / 4.f;  
  mipp::Reg<float> r_pi_div_8 = M_PI / 8.f; 
  mipp::Reg<float> r_pi = M_PI; 

  for (unsigned it = 1; it <= nb_iter; it++) {
      for (unsigned i = 0; i < DIM; i++) {
          mipp::Reg<float> r_atan2f_in1 = r_dim - mipp::cvt<int,float>(mipp::Reg<int>(i));  

          for (unsigned j = 0; j < DIM; j += mipp::N<float>()) {
              int tab_j[mipp::N<int>()];
              for (unsigned jj = 0; jj < mipp::N<float>(); jj++) {
                  tab_j[jj] = j + jj;
              }
              mipp::Reg<float> r_atan2f_in2 = mipp::cvt<int,float>(mipp::Reg<int>(tab_j)) - r_dim;

              // atan2f_approx_simd 
              mipp::Reg<float> r_abs_y = mipp::abs(r_atan2f_in1);
              mipp::Reg<float> r_abs_x = mipp::abs(r_atan2f_in2);
              mipp::Msk<mipp::N<float>()> invert_mask = r_abs_y > r_abs_x;
              mipp::Reg<float> r_z = mipp::blend(r_abs_x / r_abs_y, r_abs_y / r_abs_x, invert_mask);
              mipp::Reg<float> r_atan_approx = mipp::fmadd(r_z, r_pi_div_4, mipp::Reg<float>(0.273f) * r_z * (mipp::Reg<float>(1.f) - mipp::abs(r_z)));

              r_atan_approx = mipp::blend(mipp::Reg<float>(M_PI_2) - r_atan_approx, r_atan_approx,invert_mask);
              mipp::Msk<mipp::N<float>()> x_negative_mask = r_atan2f_in2 < mipp::Reg<float>(0.f);
              r_atan_approx = mipp::blend(r_pi - r_atan_approx, r_atan_approx,x_negative_mask);
              mipp::Msk<mipp::N<float>()> y_negative_mask = r_atan2f_in1 < mipp::Reg<float>(0.f);
              r_atan_approx = mipp::blend(-r_atan_approx, r_atan_approx,y_negative_mask);

              mipp::Reg<float> r_angle = r_atan_approx + mipp::Reg<float>(M_PI + base_angle);
              mipp::Reg<float> r_mod_reg = fmodf_approx_simd(r_angle, r_pi_div_4);

              //ratio 
              mipp::Reg<float> r_ratio = mipp::abs((r_mod_reg - r_pi_div_8) / r_pi_div_8);

              mipp::Reg<int> r_r = mipp::cvt<float,int>(r_ratio * color_a_r + (r_1 - r_ratio) * color_b_r);
              mipp::Reg<int> r_g = mipp::cvt<float,int>(r_ratio * color_a_g + (r_1 - r_ratio) * color_b_g);
              mipp::Reg<int> r_b = mipp::cvt<float,int>(r_ratio * color_a_b + (r_1 - r_ratio) * color_b_b);
              mipp::Reg<int> r_a = mipp::cvt<float,int>(r_ratio * color_a_a + (r_1 - r_ratio) * color_b_a);
              
              mipp::Reg<int> r_result = rgba_simd(r_r, r_g, r_b, r_a);

              int* img_out_ptr = (int*)&cur_img(i, j);
              r_result.store(img_out_ptr);
          }
      }
      rotate(); 
  }
  return 0;
}

// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 6
// ----------------------------------------------------------------------------

//The step size of j will double, processing twice as many pixels each time through the loop.
EXTERN unsigned spin_compute_simd_v6u2(unsigned nb_iter) {
  mipp::Reg<float> r_dim = DIM / 2.f;
  mipp::Reg<float> r_pi_div_4 = M_PI / 4.f;
  mipp::Reg<float> r_pi_div_8 = M_PI / 8.f;
  mipp::Reg<float> r_pi = M_PI;
  mipp::Reg<float> r_1 = 1.f;

  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++) {
      mipp::Reg<float> r_atan2f_in1 = r_dim - mipp::cvt<int, float>(mipp::Reg<int>(i)); 
      for (unsigned j = 0; j < DIM; j += 2 * mipp::N<float>()) { // 2-order unrolling
        int tab_j0[mipp::N<int>()], tab_j1[mipp::N<int>()];
        for (unsigned jj = 0; jj < mipp::N<float>(); jj++) {
          tab_j0[jj] = j + jj;
          tab_j1[jj] = j + jj + mipp::N<float>();
        }

        //j0
        mipp::Reg<float> r_atan2f_in2_j0 = mipp::cvt<int, float>(mipp::Reg<int>(tab_j0)) - r_dim;
        mipp::Reg<float> r_abs_y_j0 = mipp::abs(r_atan2f_in1);
        mipp::Reg<float> r_abs_x_j0 = mipp::abs(r_atan2f_in2_j0);
        mipp::Msk<mipp::N<float>()> invert_mask_j0 = r_abs_y_j0 > r_abs_x_j0;
        mipp::Reg<float> r_z_j0 = mipp::blend(r_abs_x_j0 / r_abs_y_j0, r_abs_y_j0 / r_abs_x_j0, invert_mask_j0);
        
        mipp::Reg<float> r_atan_approx_j0 = mipp::fmadd(r_z_j0, r_pi_div_4, mipp::Reg<float>(0.273f) * r_z_j0 * (r_1 - mipp::abs(r_z_j0)));
        r_atan_approx_j0 = mipp::blend(mipp::Reg<float>(M_PI_2) - r_atan_approx_j0, r_atan_approx_j0, invert_mask_j0);
        mipp::Msk<mipp::N<float>()> x_negative_mask_j0 = r_atan2f_in2_j0 < mipp::Reg<float>(0.f);
        r_atan_approx_j0 = mipp::blend(r_pi - r_atan_approx_j0, r_atan_approx_j0, x_negative_mask_j0);
        mipp::Msk<mipp::N<float>()> y_negative_mask_j0 = r_atan2f_in1 < mipp::Reg<float>(0.f);
        r_atan_approx_j0 = mipp::blend(-r_atan_approx_j0, r_atan_approx_j0, y_negative_mask_j0);

       
        mipp::Reg<float> r_angle_j0 = r_atan_approx_j0 + mipp::Reg<float>(M_PI + base_angle);
        mipp::Reg<float> r_mod_reg_j0 = fmodf_approx_simd(r_angle_j0, r_pi_div_4);
        mipp::Reg<float> r_ratio_j0 = mipp::abs((r_mod_reg_j0 - r_pi_div_8) / r_pi_div_8);

        mipp::Reg<int> r_r_j0 = mipp::cvt<float, int>(r_ratio_j0 * color_a_r + (r_1 - r_ratio_j0) * color_b_r);
        mipp::Reg<int> r_g_j0 = mipp::cvt<float, int>(r_ratio_j0 * color_a_g + (r_1 - r_ratio_j0) * color_b_g);
        mipp::Reg<int> r_b_j0 = mipp::cvt<float, int>(r_ratio_j0 * color_a_b + (r_1 - r_ratio_j0) * color_b_b);
        mipp::Reg<int> r_a_j0 = mipp::cvt<float, int>(r_ratio_j0 * color_a_a + (r_1 - r_ratio_j0) * color_b_a);
        mipp::Reg<int> r_result_j0 = rgba_simd(r_r_j0, r_g_j0, r_b_j0, r_a_j0);

        //j1
        mipp::Reg<float> r_atan2f_in2_j1 = mipp::cvt<int, float>(mipp::Reg<int>(tab_j1)) - r_dim;
        mipp::Reg<float> r_abs_y_j1 = mipp::abs(r_atan2f_in1);
        mipp::Reg<float> r_abs_x_j1 = mipp::abs(r_atan2f_in2_j1);
        mipp::Msk<mipp::N<float>()> invert_mask_j1 = r_abs_y_j1 > r_abs_x_j1;
        mipp::Reg<float> r_z_j1 = mipp::blend(r_abs_x_j1 / r_abs_y_j1, r_abs_y_j1 / r_abs_x_j1, invert_mask_j1);
        mipp::Reg<float> r_atan_approx_j1 = mipp::fmadd(r_z_j1, r_pi_div_4, mipp::Reg<float>(0.273f) * r_z_j1 * (r_1 - mipp::abs(r_z_j1)));
        r_atan_approx_j1 = mipp::blend(mipp::Reg<float>(M_PI_2) - r_atan_approx_j1, r_atan_approx_j1, invert_mask_j1);
        mipp::Msk<mipp::N<float>()> x_negative_mask_j1 = r_atan2f_in2_j1 < mipp::Reg<float>(0.f);
        r_atan_approx_j1 = mipp::blend(r_pi - r_atan_approx_j1, r_atan_approx_j1, x_negative_mask_j1);
        mipp::Msk<mipp::N<float>()> y_negative_mask_j1 = r_atan2f_in1 < mipp::Reg<float>(0.f);
        r_atan_approx_j1 = mipp::blend(-r_atan_approx_j1, r_atan_approx_j1, y_negative_mask_j1);

        mipp::Reg<float> r_angle_j1 = r_atan_approx_j1 + mipp::Reg<float>(M_PI + base_angle);
        mipp::Reg<float> r_mod_reg_j1 = fmodf_approx_simd(r_angle_j1, r_pi_div_4);
        mipp::Reg<float> r_ratio_j1 = mipp::abs((r_mod_reg_j1 - r_pi_div_8) / r_pi_div_8);

        mipp::Reg<int> r_r_j1 = mipp::cvt<float, int>(r_ratio_j1 * color_a_r + (r_1 - r_ratio_j1) * color_b_r);
        mipp::Reg<int> r_g_j1 = mipp::cvt<float, int>(r_ratio_j1 * color_a_g + (r_1 - r_ratio_j1) * color_b_g);
        mipp::Reg<int> r_b_j1 = mipp::cvt<float, int>(r_ratio_j1 * color_a_b + (r_1 - r_ratio_j1) * color_b_b);
        mipp::Reg<int> r_a_j1 = mipp::cvt<float, int>(r_ratio_j1 * color_a_a + (r_1 - r_ratio_j1) * color_b_a);
        mipp::Reg<int> r_result_j1 = rgba_simd(r_r_j1, r_g_j1, r_b_j1, r_a_j1);

        
        int* img_out_ptr_j0 = (int*)&cur_img(i, j);
        int* img_out_ptr_j1 = (int*)&cur_img(i, j + mipp::N<float>());
        r_result_j0.store(img_out_ptr_j0);
        r_result_j1.store(img_out_ptr_j1);
      }
    }
    rotate();
  }
  return 0;
}


/* Implement similar to  2-order unrolling, add 2 `j` processing blocks
 * Expand to 4th order expansion according to the same logic
 */
EXTERN unsigned spin_compute_simd_v6u4(unsigned nb_iter) {
  mipp::Reg<float> r_dim = DIM / 2.f;
  mipp::Reg<float> r_pi_div_4 = M_PI / 4.f;
  mipp::Reg<float> r_pi_div_8 = M_PI / 8.f;
  mipp::Reg<float> r_pi = M_PI;
  mipp::Reg<float> r_1 = 1.f;

  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++) {
      mipp::Reg<float> r_atan2f_in1 = r_dim - mipp::cvt<int, float>(mipp::Reg<int>(i)); 

      for (unsigned j = 0; j < DIM; j += 4 * mipp::N<float>()) {
        int tab_j0[mipp::N<int>()], tab_j1[mipp::N<int>()], tab_j2[mipp::N<int>()], tab_j3[mipp::N<int>()];
        for (unsigned jj = 0; jj < mipp::N<float>(); jj++) {
          tab_j0[jj] = j + jj;
          tab_j1[jj] = j + jj + mipp::N<float>();
          tab_j2[jj] = j + jj + 2 * mipp::N<float>();
          tab_j3[jj] = j + jj + 3 * mipp::N<float>();
        }

        //j0
        mipp::Reg<float> r_atan2f_in2_j0 = mipp::cvt<int, float>(mipp::Reg<int>(tab_j0)) - r_dim;
        mipp::Reg<float> r_abs_y_j0 = mipp::abs(r_atan2f_in1);
        mipp::Reg<float> r_abs_x_j0 = mipp::abs(r_atan2f_in2_j0);
        mipp::Msk<mipp::N<float>()> invert_mask_j0 = r_abs_y_j0 > r_abs_x_j0;
        mipp::Reg<float> r_z_j0 = mipp::blend(r_abs_x_j0 / r_abs_y_j0, r_abs_y_j0 / r_abs_x_j0, invert_mask_j0);
        mipp::Reg<float> r_atan_approx_j0 = mipp::fmadd(r_z_j0, r_pi_div_4, mipp::Reg<float>(0.273f) * r_z_j0 * (r_1 - mipp::abs(r_z_j0)));
        r_atan_approx_j0 = mipp::blend(mipp::Reg<float>(M_PI_2) - r_atan_approx_j0, r_atan_approx_j0, invert_mask_j0);
        mipp::Msk<mipp::N<float>()> x_negative_mask_j0 = r_atan2f_in2_j0 < mipp::Reg<float>(0.f);
        r_atan_approx_j0 = mipp::blend(r_pi - r_atan_approx_j0, r_atan_approx_j0, x_negative_mask_j0);
        mipp::Msk<mipp::N<float>()> y_negative_mask_j0 = r_atan2f_in1 < mipp::Reg<float>(0.f);
        r_atan_approx_j0 = mipp::blend(-r_atan_approx_j0, r_atan_approx_j0, y_negative_mask_j0);

  
        mipp::Reg<float> r_angle_j0 = r_atan_approx_j0 + mipp::Reg<float>(M_PI + base_angle);
        mipp::Reg<float> r_mod_reg_j0 = fmodf_approx_simd(r_angle_j0, r_pi_div_4);
        mipp::Reg<float> r_ratio_j0 = mipp::abs((r_mod_reg_j0 - r_pi_div_8) / r_pi_div_8);
        mipp::Reg<int> r_r_j0 = mipp::cvt<float, int>(r_ratio_j0 * color_a_r + (r_1 - r_ratio_j0) * color_b_r);
        mipp::Reg<int> r_g_j0 = mipp::cvt<float, int>(r_ratio_j0 * color_a_g + (r_1 - r_ratio_j0) * color_b_g);
        mipp::Reg<int> r_b_j0 = mipp::cvt<float, int>(r_ratio_j0 * color_a_b + (r_1 - r_ratio_j0) * color_b_b);
        mipp::Reg<int> r_a_j0 = mipp::cvt<float, int>(r_ratio_j0 * color_a_a + (r_1 - r_ratio_j0) * color_b_a);
        mipp::Reg<int> r_result_j0 = rgba_simd(r_r_j0, r_g_j0, r_b_j0, r_a_j0);

        //j1
        mipp::Reg<float> r_atan2f_in2_j1 = mipp::cvt<int, float>(mipp::Reg<int>(tab_j1)) - r_dim;
        mipp::Reg<float> r_abs_y_j1 = mipp::abs(r_atan2f_in1);
        mipp::Reg<float> r_abs_x_j1 = mipp::abs(r_atan2f_in2_j1);
        mipp::Msk<mipp::N<float>()> invert_mask_j1 = r_abs_y_j1 > r_abs_x_j1;
        mipp::Reg<float> r_z_j1 = mipp::blend(r_abs_x_j1 / r_abs_y_j1, r_abs_y_j1 / r_abs_x_j1, invert_mask_j1);
        mipp::Reg<float> r_atan_approx_j1 = mipp::fmadd(r_z_j1, r_pi_div_4, mipp::Reg<float>(0.273f) * r_z_j1 * (r_1 - mipp::abs(r_z_j1)));
        r_atan_approx_j1 = mipp::blend(mipp::Reg<float>(M_PI_2) - r_atan_approx_j1, r_atan_approx_j1, invert_mask_j1);
        mipp::Msk<mipp::N<float>()> x_negative_mask_j1 = r_atan2f_in2_j1 < mipp::Reg<float>(0.f);
        r_atan_approx_j1 = mipp::blend(r_pi - r_atan_approx_j1, r_atan_approx_j1, x_negative_mask_j1);
        mipp::Msk<mipp::N<float>()> y_negative_mask_j1 = r_atan2f_in1 < mipp::Reg<float>(0.f);
        r_atan_approx_j1 = mipp::blend(-r_atan_approx_j1, r_atan_approx_j1, y_negative_mask_j1);

        mipp::Reg<float> r_angle_j1 = r_atan_approx_j1 + mipp::Reg<float>(M_PI + base_angle);
        mipp::Reg<float> r_mod_reg_j1 = fmodf_approx_simd(r_angle_j1, r_pi_div_4);
        mipp::Reg<float> r_ratio_j1 = mipp::abs((r_mod_reg_j1 - r_pi_div_8) / r_pi_div_8);
        mipp::Reg<int> r_r_j1 = mipp::cvt<float, int>(r_ratio_j1 * color_a_r + (r_1 - r_ratio_j1) * color_b_r);
        mipp::Reg<int> r_g_j1 = mipp::cvt<float, int>(r_ratio_j1 * color_a_g + (r_1 - r_ratio_j1) * color_b_g);
        mipp::Reg<int> r_b_j1 = mipp::cvt<float, int>(r_ratio_j1 * color_a_b + (r_1 - r_ratio_j1) * color_b_b);
        mipp::Reg<int> r_a_j1 = mipp::cvt<float, int>(r_ratio_j1 * color_a_a + (r_1 - r_ratio_j1) * color_b_a);
        mipp::Reg<int> r_result_j1 = rgba_simd(r_r_j1, r_g_j1, r_b_j1, r_a_j1);

        //j2
        mipp::Reg<float> r_atan2f_in2_j2 = mipp::cvt<int, float>(mipp::Reg<int>(tab_j2)) - r_dim;
        mipp::Reg<float> r_abs_y_j2 = mipp::abs(r_atan2f_in1);
        mipp::Reg<float> r_abs_x_j2 = mipp::abs(r_atan2f_in2_j2);
        mipp::Msk<mipp::N<float>()> invert_mask_j2 = r_abs_y_j2 > r_abs_x_j2;
        mipp::Reg<float> r_z_j2 = mipp::blend(r_abs_x_j2 / r_abs_y_j2, r_abs_y_j2 / r_abs_x_j2, invert_mask_j2);
        mipp::Reg<float> r_atan_approx_j2 = mipp::fmadd(r_z_j2, r_pi_div_4, mipp::Reg<float>(0.273f) * r_z_j2 * (r_1 - mipp::abs(r_z_j2)));
        r_atan_approx_j2 = mipp::blend(mipp::Reg<float>(M_PI_2) - r_atan_approx_j2, r_atan_approx_j2, invert_mask_j2);
        mipp::Msk<mipp::N<float>()> x_negative_mask_j2 = r_atan2f_in2_j2 < mipp::Reg<float>(0.f);
        r_atan_approx_j2 = mipp::blend(r_pi - r_atan_approx_j2, r_atan_approx_j2, x_negative_mask_j2);
        mipp::Msk<mipp::N<float>()> y_negative_mask_j2 = r_atan2f_in1 < mipp::Reg<float>(0.f);
        r_atan_approx_j2 = mipp::blend(-r_atan_approx_j2, r_atan_approx_j2, y_negative_mask_j2);

        mipp::Reg<float> r_angle_j2 = r_atan_approx_j2 + mipp::Reg<float>(M_PI + base_angle);
        mipp::Reg<float> r_mod_reg_j2 = fmodf_approx_simd(r_angle_j2, r_pi_div_4);
        mipp::Reg<float> r_ratio_j2 = mipp::abs((r_mod_reg_j2 - r_pi_div_8) / r_pi_div_8);
        mipp::Reg<int> r_r_j2 = mipp::cvt<float, int>(r_ratio_j2 * color_a_r + (r_1 - r_ratio_j2) * color_b_r);
        mipp::Reg<int> r_g_j2 = mipp::cvt<float, int>(r_ratio_j2 * color_a_g + (r_1 - r_ratio_j2) * color_b_g);
        mipp::Reg<int> r_b_j2 = mipp::cvt<float, int>(r_ratio_j2 * color_a_b + (r_1 - r_ratio_j2) * color_b_b);
        mipp::Reg<int> r_a_j2 = mipp::cvt<float, int>(r_ratio_j2 * color_a_a + (r_1 - r_ratio_j2) * color_b_a);
        mipp::Reg<int> r_result_j2 = rgba_simd(r_r_j2, r_g_j2, r_b_j2, r_a_j2);

        //j3
        mipp::Reg<float> r_atan2f_in2_j3 = mipp::cvt<int, float>(mipp::Reg<int>(tab_j3)) - r_dim;
        mipp::Reg<float> r_abs_y_j3 = mipp::abs(r_atan2f_in1);
        mipp::Reg<float> r_abs_x_j3 = mipp::abs(r_atan2f_in2_j3);
        mipp::Msk<mipp::N<float>()> invert_mask_j3 = r_abs_y_j3 > r_abs_x_j3;
        mipp::Reg<float> r_z_j3 = mipp::blend(r_abs_x_j3 / r_abs_y_j3, r_abs_y_j3 / r_abs_x_j3, invert_mask_j3);
        mipp::Reg<float> r_atan_approx_j3 = mipp::fmadd(r_z_j3, r_pi_div_4, mipp::Reg<float>(0.273f) * r_z_j3 * (r_1 - mipp::abs(r_z_j3)));
        r_atan_approx_j3 = mipp::blend(mipp::Reg<float>(M_PI_2) - r_atan_approx_j3, r_atan_approx_j3, invert_mask_j3);
        mipp::Msk<mipp::N<float>()> x_negative_mask_j3 = r_atan2f_in2_j3 < mipp::Reg<float>(0.f);
        r_atan_approx_j3 = mipp::blend(r_pi - r_atan_approx_j3, r_atan_approx_j3, x_negative_mask_j3);
        mipp::Msk<mipp::N<float>()> y_negative_mask_j3 = r_atan2f_in1 < mipp::Reg<float>(0.f);
        r_atan_approx_j3 = mipp::blend(-r_atan_approx_j3, r_atan_approx_j3, y_negative_mask_j3);

        mipp::Reg<float> r_angle_j3 = r_atan_approx_j3 + mipp::Reg<float>(M_PI + base_angle);
        mipp::Reg<float> r_mod_reg_j3 = fmodf_approx_simd(r_angle_j3, r_pi_div_4);
        mipp::Reg<float> r_ratio_j3 = mipp::abs((r_mod_reg_j3 - r_pi_div_8) / r_pi_div_8);
        mipp::Reg<int> r_r_j3 = mipp::cvt<float, int>(r_ratio_j3 * color_a_r + (r_1 - r_ratio_j3) * color_b_r);
        mipp::Reg<int> r_g_j3 = mipp::cvt<float, int>(r_ratio_j3 * color_a_g + (r_1 - r_ratio_j3) * color_b_g);
        mipp::Reg<int> r_b_j3 = mipp::cvt<float, int>(r_ratio_j3 * color_a_b + (r_1 - r_ratio_j3) * color_b_b);
        mipp::Reg<int> r_a_j3 = mipp::cvt<float, int>(r_ratio_j3 * color_a_a + (r_1 - r_ratio_j3) * color_b_a);
        mipp::Reg<int> r_result_j3 = rgba_simd(r_r_j3, r_g_j3, r_b_j3, r_a_j3);

        int* img_out_ptr_j0 = (int*)&cur_img(i, j);
        int* img_out_ptr_j1 = (int*)&cur_img(i, j + mipp::N<float>());
        int* img_out_ptr_j2 = (int*)&cur_img(i, j + 2 * mipp::N<float>());
        int* img_out_ptr_j3 = (int*)&cur_img(i, j + 3 * mipp::N<float>());
        r_result_j0.store(img_out_ptr_j0);
        r_result_j1.store(img_out_ptr_j1);
        r_result_j2.store(img_out_ptr_j2);
        r_result_j3.store(img_out_ptr_j3);
      }
    }
    rotate();
  }
  return 0;
}


#endif /* ENABLE_VECTO */
