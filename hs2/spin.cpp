#include <math.h>
#include <omp.h>

#include "global.h"
#include "img_data.h"
#include "cppdefs.h"
#include "mipp.h"

static void rotate(void);

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
// Suggested cmdline(s):
// ./run --size 1024 --kernel spin --variant seq
// or
// ./run -s 1024 -k spin -v seq
//
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

// Computation of one pixel
static mipp::Reg<int> compute_color_simd_v0(mipp::Reg<int> r_i,
                                            mipp::Reg<int> r_j)
{
  int result[mipp::N<int>()]; // array of final pixels
  for(int index=0; index<mipp::N<int>(); index++)
  {
    int i = r_i[index]; // changeme
    int j = r_j[index];
    float atan2f_in1 = (float)DIM / 2.f - (float)i;
    float atan2f_in2 = (float)j - (float)DIM / 2.f;
    float angle = atan2f_approx(atan2f_in1, atan2f_in2) + M_PI + base_angle;

    float ratio = fabsf((fmodf_approx(angle, M_PI / 4.f) - (M_PI / 8.f)) / (M_PI / 8.f));

    int r = color_a_r * ratio + color_b_r * (1.f - ratio);
    int g = color_a_g * ratio + color_b_g * (1.f - ratio);
    int b = color_a_b * ratio + color_b_b * (1.f - ratio);
    int a = color_a_a * ratio + color_b_a * (1.f - ratio);

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

static inline mipp::Reg<float> fmodf_approx_simd(mipp::Reg<float> r_x,
                                                 mipp::Reg<float> r_y) {
  // Use trunc to get the integer part of x/y
  return r_x - mipp::trunc(r_x / r_y) * r_y;
}

// Computation of one pixel
static inline mipp::Reg<int> compute_color_simd_v1(mipp::Reg<int> r_i,
                                                   mipp::Reg<int> r_j)
{
  float angles[mipp::N<float>()];

  for(int index=0; index<mipp::N<int>(); index++)
  {
    int i = r_i[index]; // changeme
    int j = r_j[index];
    float atan2f_in1 = (float)DIM / 2.f - (float)i;
    float atan2f_in2 = (float)j - (float)DIM / 2.f;
    angles[index] = atan2f_approx(atan2f_in1, atan2f_in2) + M_PI + base_angle;
  }

  mipp::Reg<float> ratio = mipp::abs((fmodf_approx_simd(mipp::Reg<float>(angles),
        mipp::Reg<float>(M_PI / 4.f)) - mipp::Reg<float>(M_PI / 8.f)) / mipp::Reg<float>(M_PI / 8.f));
  
  int result[mipp::N<int>()];
  for(int index=0; index<mipp::N<int>(); index++)
  {
    int r = color_a_r * ratio[index] + color_b_r * (1.f - ratio[index]);
    int g = color_a_g * ratio[index] + color_b_g * (1.f - ratio[index]);
    int b = color_a_b * ratio[index] + color_b_b * (1.f - ratio[index]);
    int a = color_a_a * ratio[index] + color_b_a * (1.f - ratio[index]);

    result[index] = ezv_rgba(r, g, b, a);
  }
  return mipp::Reg<int>(result);
}

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
                                       mipp::Reg<int> r_b, mipp::Reg<int> r_a) {
  return r_r  | r_g << 8 | r_b << 16 | r_a << 24;
}

// Computation of one pixel
static inline mipp::Reg<int> compute_color_simd_v2(mipp::Reg<int> r_i,
                                                   mipp::Reg<int> r_j)
{
  float angles[mipp::N<float>()];

  for(int index=0; index<mipp::N<int>(); index++)
  {
    int i = r_i[index]; // changeme
    int j = r_j[index];
    float atan2f_in1 = (float)DIM / 2.f - (float)i;
    float atan2f_in2 = (float)j - (float)DIM / 2.f;
    angles[index] = atan2f_approx(atan2f_in1, atan2f_in2) + M_PI + base_angle;
  }

  mipp::Reg<float> ratio = mipp::abs((fmodf_approx_simd(mipp::Reg<float>(angles),
        mipp::Reg<float>(M_PI / 4.f)) - mipp::Reg<float>(M_PI / 8.f)) / mipp::Reg<float>(M_PI / 8.f));
  
  mipp::Reg<int> r_r = mipp::cvt<float,int>(mipp::Reg<float>(color_a_r) * ratio + mipp::Reg<float>(color_b_r) * (mipp::Reg<float>(1.f) - ratio));
  mipp::Reg<int> r_g = mipp::cvt<float,int>(mipp::Reg<float>(color_a_g) * ratio + mipp::Reg<float>(color_b_g) * (mipp::Reg<float>(1.f) - ratio));
  mipp::Reg<int> r_b = mipp::cvt<float,int>(mipp::Reg<float>(color_a_b) * ratio + mipp::Reg<float>(color_b_b) * (mipp::Reg<float>(1.f) - ratio));
  mipp::Reg<int> r_a = mipp::cvt<float,int>(mipp::Reg<float>(color_a_a) * ratio + mipp::Reg<float>(color_b_a) * (mipp::Reg<float>(1.f) - ratio));
  
  return rgba_simd(r_r, r_g, r_b, r_a);
}

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

static inline mipp::Reg<float> atanf_approx_simd(mipp::Reg<float> r_z) {
  // return x * M_PI / 4.f + 0.273f * x * (1.f - fabsf(x));
  mipp::Reg<float> result = r_z * mipp::Reg<float>(M_PI / 4.f) + mipp::Reg<float>(0.273f) * r_z * (mipp::Reg<float>(1.f) - mipp::abs(r_z));
  return result;
}

static inline mipp::Reg<float> atan2f_approx_simd(mipp::Reg<float> r_y,
                                                  mipp::Reg<float> r_x) {
  mipp::Reg<float> abs_y = mipp::abs(r_y);
  mipp::Reg<float> abs_x = mipp::abs(r_x);
  
  mipp::Msk<mipp::N<float>()> invert_mask = abs_y > abs_x;
  mipp::Reg<float> z = mipp::blend(invert_mask, abs_x / abs_y, abs_y / abs_x);
  mipp::Reg<float> th = atanf_approx_simd(z); // [0, pi/4]
  th = mipp::blend(invert_mask, mipp::Reg<float>(M_PI_2) - th, th); // [0, pi/2]
  // if x < 0
  mipp::Msk<mipp::N<float>()> x_negative = r_x < mipp::Reg<float>(0.f);
  th = mipp::blend(x_negative, mipp::Reg<float>(M_PI) - th, th); // [0, pi]

  // if y < 0
  mipp::Msk<mipp::N<float>()> y_negative = r_y < mipp::Reg<float>(0.f);
  th = mipp::blend(y_negative, -th, th);

  return th;
}

// Computation of one pixel
static inline mipp::Reg<int> compute_color_simd_v3(mipp::Reg<int> r_i,
                                                   mipp::Reg<int> r_j)
{
  mipp::Reg<float> atan2f_in1 = mipp::Reg<float>(DIM / 2.f) - mipp::Reg<float>(r_i);
  mipp::Reg<float> atan2f_in2 = mipp::Reg<float>(r_j) - mipp::Reg<float>(DIM / 2.f);

  mipp::Reg<float> angle = atan2f_approx_simd(atan2f_in1, atan2f_in2) + mipp::Reg<float>(M_PI + base_angle);

  mipp::Reg<float> mod_reg = fmodf_approx_simd(angle, mipp::Reg<float>(M_PI / 4.f));

  mipp::Reg<float> ratio = mipp::abs((fmodf_approx_simd(mipp::Reg<float>(angles),
        mipp::Reg<float>(M_PI / 4.f)) - mipp::Reg<float>(M_PI / 8.f)) / mipp::Reg<float>(M_PI / 8.f));
  
  mipp::Reg<int> r_r = mipp::cvt<float,int>(mipp::Reg<float>(color_a_r) * ratio + mipp::Reg<float>(color_b_r) * (mipp::Reg<float>(1.f) - ratio));
  mipp::Reg<int> r_g = mipp::cvt<float,int>(mipp::Reg<float>(color_a_g) * ratio + mipp::Reg<float>(color_b_g) * (mipp::Reg<float>(1.f) - ratio));
  mipp::Reg<int> r_b = mipp::cvt<float,int>(mipp::Reg<float>(color_a_b) * ratio + mipp::Reg<float>(color_b_b) * (mipp::Reg<float>(1.f) - ratio));
  mipp::Reg<int> r_a = mipp::cvt<float,int>(mipp::Reg<float>(color_a_a) * ratio + mipp::Reg<float>(color_b_a) * (mipp::Reg<float>(1.f) - ratio));
  
  return rgba_simd(r_r, r_g, r_b, r_a);
}

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
static inline mipp::Reg<int> compute_color_simd_v4(mipp::Reg<int> r_i,
                                                   mipp::Reg<int> r_j)
{
  
  mipp::Reg<float> atan2f_in1 = mipp::Reg<float>(DIM / 2.f) - mipp::Reg<float>(r_i);
  mipp::Reg<float> atan2f_in2 = mipp::Reg<float>(r_j) - mipp::Reg<float>(DIM / 2.f);

  // atan2f_approx_simd 
  mipp::Reg<float> abs_y = mipp::abs(atan2f_in1);
  mipp::Reg<float> abs_x = mipp::abs(atan2f_in2);
  mipp::Msk<mipp::N<float>()> invert_mask = abs_y > abs_x;

  mipp::Reg<float> z = mipp::blend(invert_mask, abs_x / abs_y, abs_y / abs_x);

  // atanf_approx_simd
  mipp::Reg<float> atan_approx = z * mipp::Reg<float>(M_PI / 4.f) + mipp::Reg<float>(0.273f) * z * (mipp::Reg<float>(1.f) - mipp::abs(z));
  atan_approx = mipp::blend(invert_mask, mipp::Reg<float>(M_PI_2) - atan_approx, atan_approx); 

  // x < 0 
  mipp::Msk<mipp::N<float>()> x_negative_mask = atan2f_in2 < mipp::Reg<float>(0.f);
  atan_approx = mipp::blend(x_negative_mask, mipp::Reg<float>(M_PI) - atan_approx, atan_approx); 

  // y < 0 
  mipp::Msk<mipp::N<float>()> y_negative_mask = atan2f_in1 < mipp::Reg<float>(0.f);
  atan_approx = mipp::blend(y_negative_mask, -atan_approx, atan_approx);

  mipp::Reg<float> angle = atan_approx + mipp::Reg<float>(M_PI + base_angle);

  mipp::Reg<float> ratio = mipp::abs((fmodf_approx_simd(mipp::Reg<float>(angles),
  mipp::Reg<float>(M_PI / 4.f)) - mipp::Reg<float>(M_PI / 8.f)) / mipp::Reg<float>(M_PI / 8.f));
  
  mipp::Reg<int> r_r = mipp::cvt<float,int>(mipp::Reg<float>(color_a_r) * ratio + mipp::Reg<float>(color_b_r) * (mipp::Reg<float>(1.f) - ratio));
  mipp::Reg<int> r_g = mipp::cvt<float,int>(mipp::Reg<float>(color_a_g) * ratio + mipp::Reg<float>(color_b_g) * (mipp::Reg<float>(1.f) - ratio));
  mipp::Reg<int> r_b = mipp::cvt<float,int>(mipp::Reg<float>(color_a_b) * ratio + mipp::Reg<float>(color_b_b) * (mipp::Reg<float>(1.f) - ratio));
  mipp::Reg<int> r_a = mipp::cvt<float,int>(mipp::Reg<float>(color_a_a) * ratio + mipp::Reg<float>(color_b_a) * (mipp::Reg<float>(1.f) - ratio));
  
  return rgba_simd(r_r, r_g, r_b, r_a);
}

}

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
    mipp::Reg<float> r_dim = mipp::Reg<float>(DIM / 2.f);  
    mipp::Reg<float> r_pi_div_4 = mipp::Reg<float>(M_PI / 4.f);  
    mipp::Reg<float> r_pi_div_8 = mipp::Reg<float>(M_PI / 8.f); 
    mipp::Reg<float> r_pi = mipp::Reg<float>(M_PI);  

    for (unsigned it = 1; it <= nb_iter; it++) {
        for (unsigned i = 0; i < DIM; i++) {
            mipp::Reg<float> r_atan2f_in1 =r_dim - mipp::Reg<float>(i);  

            for (unsigned j = 0; j < DIM; j += mipp::N<float>()) {
                int tab_j[mipp::N<int>()];
                for (unsigned jj = 0; jj < mipp::N<float>(); jj++) {
                    tab_j[jj] = j + jj;
                }
                mipp::Reg<float> r_atan2f_in2 = mipp::Reg<float>(tab_j) - r_dim;

                // atan2f_approx_simd 
                mipp::Reg<float> r_abs_y = mipp::abs(r_atan2f_in1);
                mipp::Reg<float> r_abs_x = mipp::abs(r_atan2f_in2);
                mipp::Msk<mipp::N<float>()> invert_mask = r_abs_y > r_abs_x;
                mipp::Reg<float> r_z = mipp::blend(invert_mask, r_abs_x / r_abs_y, r_abs_y / r_abs_x);
                mipp::Reg<float> r_atan_approx = mipp::fmadd(r_z, r_pi_div_4, mipp::Reg<float>(0.273f) * z * (mipp::Reg<float>(1.f) - mipp::abs(r_z)));

                atan_approx = mipp::blend(mipp::Reg<float>(M_PI_2) - r_atan_approx, r_atan_approx,invert_mask);
                mipp::Msk<mipp::N<float>()> x_negative_mask = r_atan2f_in2 < mipp::Reg<float>(0.f);
                atan_approx = mipp::blend(r_pi - r_atan_approx, r_atan_approx,x_negative_mask);
                mipp::Msk<mipp::N<float>()> y_negative_mask = r_atan2f_in1 < mipp::Reg<float>(0.f);
                atan_approx = mipp::blend(-r_atan_approx, r_atan_approx,y_negative_mask);

                mipp::Reg<float> r_angle = atan_approx + mipp::Reg<float>(M_PI + base_angle);
                mipp::Reg<float> r_mod_reg = fmodf_approx_simd(r_angle, r_pi_div_4);

                //ratio 
                mipp::Reg<float> r_ratio = mipp::abs((r_mod_reg - r_pi_div_8) / r_pi_div_8);

                // 
                mipp::Reg<int> r_r = mipp::cvt<float, int>(mipp::fmadd(mipp::Reg<float>(color_a_r), ratio, mipp::Reg<float>(color_b_r) * (mipp::Reg<float>(1.f) - ratio)));
                mipp::Reg<int> r_g = mipp::cvt<float, int>(mipp::fmadd(mipp::Reg<float>(color_a_g), ratio, mipp::Reg<float>(color_b_g) * (mipp::Reg<float>(1.f) - ratio)));
                mipp::Reg<int> r_b = mipp::cvt<float, int>(mipp::fmadd(mipp::Reg<float>(color_a_b), ratio, mipp::Reg<float>(color_b_b) * (mipp::Reg<float>(1.f) - ratio)));
                mipp::Reg<int> r_a = mipp::cvt<float, int>(mipp::fmadd(mipp::Reg<float>(color_a_a), ratio, mipp::Reg<float>(color_b_a) * (mipp::Reg<float>(1.f) - ratio)));

                
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

EXTERN unsigned spin_compute_simd_v6(unsigned nb_iter) {
  return 0;
}

#endif /* ENABLE_VECTO */
