#include "global.h"
#include "img_data.h"
#include "cppdefs.h"
#include "mipp.h"

extern "C" int blur2_do_tile_default(int x, int y, int width, int height);
extern "C" void compute_borders(int x, int y, int width, int height, int bsize);

#ifdef ENABLE_VECTO

/**
 * Divide by 9 using a series of shifts and additions. In this 
 * way we can avoid the division operation and the corresponding
 * cost of conversion to float and back to integer.
 */
mipp::Reg<uint16_t> mipp_vdiv9(mipp::Reg<uint16_t> n) {
    mipp::Reg<uint16_t> q1, q2, r;
    q1 =  n - (n >> 3);
    q1 += (q1 >> 6);
    q2 = q1 >> 3;
    r = n - (q1 + q2);
    r = q2 + ((r + 7) >> 4);
    return r;
}


EXTERN int blur2_do_tile_urrot2_mipp_div9_f32(int x, int y, int width, int height)
{
  /* #########################################################################
   * #########################################################################
   * Variables notation: ^r(a[0-4])?_c_[0-3]_l_[0-2]_u[8,16,32,64]
   * r -> register
   * a[0-4] -> array of registers and its dimension
   * c_[0-2] -> column number
   * l_[0-2] -> line number
   * u[8,16,32,64] -> type of the registers
   *
   * Example ra4_c_2_l_0_u8 -> array of four registers u_0, right column, first line
   * #########################################################################
   * #########################################################################
   */

  /*
   * Contain the interleaved pixel colors of the right group column.
   * Four pixels are interleaved in a single register.
   */
  mipp::Reg<uint8_t> r_c_0_l_0_u8, r_c_0_l_1_u8, r_c_0_l_2_u8;
  mipp::Reg<uint8_t> r_c_1_l_0_u8, r_c_1_l_1_u8, r_c_1_l_2_u8;
  mipp::Reg<uint8_t> r_c_2_l_0_u8, r_c_2_l_1_u8, r_c_2_l_2_u8;
  mipp::Reg<uint8_t> r_sum_u8;

  /**
   * Two array, one for the lower and one for higher part, for each
   * line and for each column.
   */
  mipp::Reg<uint16_t> r_c_0_l_0_u16_l;                  // left column, first line, no need for higher part
  mipp::Reg<uint16_t> r_c_0_l_1_u16_h, r_c_0_l_1_u16_l; // left column, second line
  mipp::Reg<uint16_t> r_c_0_l_2_u16_l;                  // left column, third line, no need for higher part

  mipp::Reg<uint16_t> r_c_1_l_0_u16_h, r_c_1_l_0_u16_l; // central column, first line
  mipp::Reg<uint16_t> r_c_1_l_1_u16_h, r_c_1_l_1_u16_l; // central column, second line
  mipp::Reg<uint16_t> r_c_1_l_2_u16_h, r_c_1_l_2_u16_l; // central column, third line

  mipp::Reg<uint16_t> r_c_2_l_0_u16_h, r_c_2_l_0_u16_l; // right column, first line
  mipp::Reg<uint16_t> r_c_2_l_1_u16_h, r_c_2_l_1_u16_l; // right column, second line
  mipp::Reg<uint16_t> r_c_2_l_2_u16_h, r_c_2_l_2_u16_l; // right column, third line

  // for storing the sum of the pixels on the higher part of the central column
  // used in variable reduction
  mipp::Reg<uint16_t> r_c_1_l_1_u16_h_temp;

  // To store shiffted values for the reduction
  mipp::Reg<uint16_t> r_left_l, r_left_h;
  mipp::Reg<uint16_t> r_right_l, r_right_h;

  // for the promotion to 32 bits
  mipp::Reg<uint32_t> r_sum_l_l, r_sum_l_h, r_sum_h_l, r_sum_h_h;

  // for the division by 9
  mipp::Reg<float_t> r_sumf_l_l, r_sumf_l_h, r_sumf_h_l, r_sumf_h_h;

  /*
   * Our code should be portable now, so the number of pixels loaded
   * at each iteration depends on the ISA of the target architecture.
   * 
   * We can use the mipp::N<uint8_t> to get the number of pixels loaded
   * at each iteration and use it to increment the x-loop.
   */
  constexpr int inc = mipp::N<uint8_t>() / 4; // number of pixels loaded at each iteration

  // loop over y (start from +1, end at -1 => no border)
  for (int i = y + 1; i < y + height - 1; i++)
  {
    // #############################################################
    //                      PROLOGUE
    // #############################################################
    /*
     * In order to start the variable rotation, we need to precompute
     * the left and central columns. The computation of the central column
     * is done in the same way as the right column in the main loop.
     * Since the x-loop starts from the second pixel, the left column
     * is composed by only the first pixel of each line. So, one
     * possible strategy is to load to perform the exact same computation
     * as the central column, but starting from the first pixel. After that,
     * we can use vextq_u16 to shift the first pixel to the last position.
     *
     * In this way, we can use simd instructions to perform the computation
     * also for the first mipp::N<uint8_t> / 4 pixels of each line instead 
     * of using scalar instructions.
     */
    {
      r_c_0_l_0_u8 = (uint8_t *)&cur_img(i - 1, x);
      r_c_0_l_1_u8 = (uint8_t *)&cur_img(i + 0, x);
      r_c_0_l_2_u8 = (uint8_t *)&cur_img(i + 1, x);

      r_c_1_l_0_u8 = (uint8_t *)&cur_img(i - 1, x + 1);
      r_c_1_l_1_u8 = (uint8_t *)&cur_img(i + 0, x + 1);
      r_c_1_l_2_u8 = (uint8_t *)&cur_img(i + 1, x + 1);

      /*
       * please note that we need only the lower part of the mipp::N<uint8_t> / 4
       * pixels for the left column. The higher part is not used in the
       * computation but only to store the first pixel for the computation 
       * in the loop.
       */
      // first line
      r_c_0_l_0_u16_l = mipp::low(r_c_0_l_0_u8);

      r_c_1_l_0_u16_l = mipp::low(r_c_1_l_0_u8);
      r_c_1_l_0_u16_h = mipp::high(r_c_1_l_0_u8);

      // second line
      r_c_0_l_1_u16_l = mipp::low(r_c_0_l_1_u8);

      r_c_1_l_1_u16_l = mipp::low(r_c_1_l_1_u8);
      r_c_1_l_1_u16_h = mipp::high(r_c_1_l_1_u8);

      // third line
      r_c_0_l_2_u16_l = mipp::low(r_c_0_l_2_u8);

      r_c_1_l_2_u16_l = mipp::low(r_c_1_l_2_u8);
      r_c_1_l_2_u16_h = mipp::high(r_c_1_l_2_u8);

      // reduction
      r_c_0_l_1_u16_l = r_c_0_l_2_u16_l + r_c_0_l_1_u16_l + r_c_0_l_0_u16_l; // lower part

      r_c_1_l_1_u16_l = r_c_1_l_2_u16_l + r_c_1_l_1_u16_l + r_c_1_l_0_u16_l; // lower part
      r_c_1_l_1_u16_h = r_c_1_l_2_u16_h + r_c_1_l_1_u16_h + r_c_1_l_0_u16_h; // higher part

      // move the first bit to the last position using vextq_u16
      r_c_0_l_1_u16_h = mipp::combine<4>(r_c_0_l_1_u16_h, r_c_0_l_1_u16_l);
    }
    // #############################################################
    //                      END PROLOGUE
    // #############################################################

    for (int j = x + 1; j < x + width; j += inc)
    {

      // 3. Memory load
      /*
       * Mipp load is interleaved by default.
       * For all the three lines of the right column-group, we load from memory 
       * mipp::N<uint8_t> / 4 pixels, leaving interleaved the colors.
       * Now each register is composed by mipp::N<uint8_t> / 4 pixels.
       */
      r_c_2_l_0_u8 = (uint8_t *)&cur_img(i - 1, j + 4); // [[ r1 g1 b1 a1 ] [r2 g2 b2 a2] [r3 g3 b3 a3] [ r4 g4 b4 a4 ]]
      r_c_2_l_1_u8 = (uint8_t *)&cur_img(i + 0, j + 4);
      r_c_2_l_2_u8 = (uint8_t *)&cur_img(i + 1, j + 4);

      // 4.
      /*
       * Promote the 8-bit components into 16-bit components to perform the accumulation
       * First we extract the lower and higher part of the 8-bit components using the
       * mipp::low and mipp::high.
       *
       */
      // first line
      r_c_2_l_0_u16_l = mipp::low(r_c_2_l_0_u8);
      r_c_2_l_0_u16_h = mipp::high(r_c_2_l_0_u8);

      // second line
      r_c_2_l_1_u16_l = mipp::low(r_c_2_l_1_u8);
      r_c_2_l_1_u16_h = mipp::high(r_c_2_l_1_u8);

      // third line
      r_c_2_l_2_u16_l = mipp::low(r_c_2_l_2_u8);
      r_c_2_l_2_u16_h = mipp::high(r_c_2_l_2_u8);

      // 5. Compute the reduction of the second column
      /*
       * Accumulate the color component of the right column-group.
       */
      r_c_2_l_1_u16_l = r_c_2_l_0_u16_l + r_c_2_l_1_u16_l + r_c_2_l_2_u16_l;
      r_c_2_l_1_u16_h = r_c_2_l_0_u16_h + r_c_2_l_1_u16_h + r_c_2_l_2_u16_h;

      // 6. left-right pattern
      /*
       * Perform the left-right pattern to compute the sum of the pixel components.
       * Now, since colors are interleaved, we need to shift 4 colors when preparing the
       * left and right part of the sum.
       */
      r_left_l = mipp::combine<4>(r_c_0_l_1_u16_h, r_c_1_l_1_u16_l);
      r_right_l = mipp::combine<4>(r_c_1_l_1_u16_l, r_c_1_l_1_u16_h);

      r_left_h = mipp::combine<4>(r_c_1_l_1_u16_l, r_c_1_l_1_u16_h);
      r_right_h = mipp::combine<4>(r_c_1_l_1_u16_h, r_c_2_l_1_u16_l);

      // store the previous value before overwrite
      r_c_1_l_1_u16_h_temp = r_c_1_l_1_u16_h;

      // vertical sum
      r_c_1_l_1_u16_l = r_left_l + r_c_1_l_1_u16_l + r_right_l; // sum of the lower part
      r_c_1_l_1_u16_h = r_left_h + r_c_1_l_1_u16_h + r_right_h; // sum of the higher part

      // 7. promotion to uint
      r_sum_l_l = mipp::low(r_c_1_l_1_u16_l);
      r_sum_l_h = mipp::high(r_c_1_l_1_u16_l);
      r_sum_h_l = mipp::low(r_c_1_l_1_u16_h);
      r_sum_h_h = mipp::high(r_c_1_l_1_u16_h);

      // 7. convert to float
      r_sumf_l_l = mipp::cvt<uint32_t, float_t>(r_sum_l_l);
      r_sumf_l_h = mipp::cvt<uint32_t, float_t>(r_sum_l_h);
      r_sumf_h_l = mipp::cvt<uint32_t, float_t>(r_sum_h_l);
      r_sumf_h_h = mipp::cvt<uint32_t, float_t>(r_sum_h_h);

      // 8. divison by 9
      r_sumf_l_l = r_sumf_l_l / 9;
      r_sumf_l_h = r_sumf_l_h / 9;
      r_sumf_h_l = r_sumf_h_l / 9;
      r_sumf_h_h = r_sumf_h_h / 9;

      // 9. convert back to uint
      r_sum_l_l = mipp::cvt<float_t, uint32_t>(r_sumf_l_l);
      r_sum_l_h = mipp::cvt<float_t, uint32_t>(r_sumf_l_h);
      r_sum_h_l = mipp::cvt<float_t, uint32_t>(r_sumf_h_l);
      r_sum_h_h = mipp::cvt<float_t, uint32_t>(r_sumf_h_h);

      // 10. convert back to uint16
      r_c_1_l_1_u16_l = {r_sum_l_l, r_sum_l_h};
      r_c_1_l_1_u16_h = {r_sum_h_l, r_sum_h_h};

      // 11. convert back to uint8
      r_sum_u8 = {r_c_1_l_1_u16_l, r_c_1_l_1_u16_h};

      // 12. store
      // store back the data
      r_sum_u8.store((uint8_t *)&next_img(i, j));

      // 13. variable rotation
      // copy the lowest part of the middle column into the highest part of the left column ( we should pass the sum of the pixels)
      // col 0 <- col 1
      r_c_0_l_1_u16_h = r_c_1_l_1_u16_h_temp;
      // col 1 <- col 2
      r_c_1_l_1_u16_l = r_c_2_l_1_u16_l;
      r_c_1_l_1_u16_h = r_c_2_l_1_u16_h;
    }
  }

  // left-right borders size
  uint32_t bsize = 1;
  // compute the borders
  compute_borders(x, y, width, height, bsize);

  return 0;
}

EXTERN int blur2_do_tile_urrot2_mipp_div9_u16(int x, int y, int width, int height)
{
  /* #########################################################################
   * #########################################################################
   * Variables notation: ^r(a[0-4])?_c_[0-3]_l_[0-2]_u[8,16,32,64]
   * r -> register
   * a[0-4] -> array of registers and its dimension
   * c_[0-2] -> column number
   * l_[0-2] -> line number
   * u[8,16,32,64] -> type of the registers
   *
   * Example ra4_c_2_l_0_u8 -> array of four registers u_0, right column, first line
   * #########################################################################
   * #########################################################################
   */

  /*
   * Contain the interleaved pixel colors of the right group column.
   * Four pixels are interleaved in a single register.
   */
  mipp::Reg<uint8_t> r_c_0_l_0_u8, r_c_0_l_1_u8, r_c_0_l_2_u8;
  mipp::Reg<uint8_t> r_c_1_l_0_u8, r_c_1_l_1_u8, r_c_1_l_2_u8;
  mipp::Reg<uint8_t> r_c_2_l_0_u8, r_c_2_l_1_u8, r_c_2_l_2_u8;
  mipp::Reg<uint8_t> r_sum_u8;

  /**
   * Two array, one for the lower and one for higher part, for each
   * line and for each column.
   */
  mipp::Reg<uint16_t> r_c_0_l_0_u16_l;                  // left column, first line, no need for higher part
  mipp::Reg<uint16_t> r_c_0_l_1_u16_h, r_c_0_l_1_u16_l; // left column, second line
  mipp::Reg<uint16_t> r_c_0_l_2_u16_l;                  // left column, third line, no need for higher part

  mipp::Reg<uint16_t> r_c_1_l_0_u16_h, r_c_1_l_0_u16_l; // central column, first line
  mipp::Reg<uint16_t> r_c_1_l_1_u16_h, r_c_1_l_1_u16_l; // central column, second line
  mipp::Reg<uint16_t> r_c_1_l_2_u16_h, r_c_1_l_2_u16_l; // central column, third line

  mipp::Reg<uint16_t> r_c_2_l_0_u16_h, r_c_2_l_0_u16_l; // right column, first line
  mipp::Reg<uint16_t> r_c_2_l_1_u16_h, r_c_2_l_1_u16_l; // right column, second line
  mipp::Reg<uint16_t> r_c_2_l_2_u16_h, r_c_2_l_2_u16_l; // right column, third line

  // for storing the sum of the pixels on the higher part of the central column
  // used in variable reduction
  mipp::Reg<uint16_t> r_c_1_l_1_u16_h_temp;

  // To store shiffted values for the reduction
  mipp::Reg<uint16_t> r_left_l, r_left_h;
  mipp::Reg<uint16_t> r_right_l, r_right_h;

  /*
   * Our code should be portable now, so the number of pixels loaded
   * at each iteration depends on the ISA of the target architecture.
   * 
   * We can use the mipp::N<uint8_t> to get the number of pixels loaded
   * at each iteration and use it to increment the x-loop.
   */
  constexpr int inc = mipp::N<uint8_t>() / 4; // number of pixels loaded at each iteration

  // loop over y (start from +1, end at -1 => no border)
  for (int i = y + 1; i < y + height - 1; i++)
  {
    // #############################################################
    //                      PROLOGUE
    // #############################################################
    /*
     * In order to start the variable rotation, we need to precompute
     * the left and central columns. The computation of the central column
     * is done in the same way as the right column in the main loop.
     * Since the x-loop starts from the second pixel, the left column
     * is composed by only the first pixel of each line. So, one
     * possible strategy is to load to perform the exact same computation
     * as the central column, but starting from the first pixel. After that,
     * we can use vextq_u16 to shift the first pixel to the last position.
     *
     * In this way, we can use simd instructions to perform the computation
     * also for the first mipp::N<uint8_t> / 4 pixels of each line instead 
     * of using scalar instructions.
     */
    {
      r_c_0_l_0_u8 = (uint8_t *)&cur_img(i - 1, x);
      r_c_0_l_1_u8 = (uint8_t *)&cur_img(i + 0, x);
      r_c_0_l_2_u8 = (uint8_t *)&cur_img(i + 1, x);

      r_c_1_l_0_u8 = (uint8_t *)&cur_img(i - 1, x + 1);
      r_c_1_l_1_u8 = (uint8_t *)&cur_img(i + 0, x + 1);
      r_c_1_l_2_u8 = (uint8_t *)&cur_img(i + 1, x + 1);

      /*
       * please note that we need only the lower part of the mipp::N<uint8_t> / 4
       * pixels for the left column. The higher part is not used in the
       * computation but only to store the first pixel for the computation 
       * in the loop.
       */
      // first line
      r_c_0_l_0_u16_l = mipp::low(r_c_0_l_0_u8);

      r_c_1_l_0_u16_l = mipp::low(r_c_1_l_0_u8);
      r_c_1_l_0_u16_h = mipp::high(r_c_1_l_0_u8);

      // second line
      r_c_0_l_1_u16_l = mipp::low(r_c_0_l_1_u8);

      r_c_1_l_1_u16_l = mipp::low(r_c_1_l_1_u8);
      r_c_1_l_1_u16_h = mipp::high(r_c_1_l_1_u8);

      // third line
      r_c_0_l_2_u16_l = mipp::low(r_c_0_l_2_u8);

      r_c_1_l_2_u16_l = mipp::low(r_c_1_l_2_u8);
      r_c_1_l_2_u16_h = mipp::high(r_c_1_l_2_u8);

      // reduction
      r_c_0_l_1_u16_l = r_c_0_l_2_u16_l + r_c_0_l_1_u16_l + r_c_0_l_0_u16_l; // lower part

      r_c_1_l_1_u16_l = r_c_1_l_2_u16_l + r_c_1_l_1_u16_l + r_c_1_l_0_u16_l; // lower part
      r_c_1_l_1_u16_h = r_c_1_l_2_u16_h + r_c_1_l_1_u16_h + r_c_1_l_0_u16_h; // higher part

      // move the first bit to the last position using vextq_u16
      r_c_0_l_1_u16_h = mipp::combine<4>(r_c_0_l_1_u16_h, r_c_0_l_1_u16_l);
    }
    // #############################################################
    //                      END PROLOGUE
    // #############################################################

    for (int j = x + 1; j < x + width; j += inc)
    {

      // 3. Memory load
      /*
       * Mipp load is interleaved by default.
       * For all the three lines of the right column-group, we load from memory 
       * mipp::N<uint8_t> / 4 pixels, leaving interleaved the colors.
       * Now each register is composed by mipp::N<uint8_t> / 4 pixels.
       */
      r_c_2_l_0_u8 = (uint8_t *)&cur_img(i - 1, j + 4); // [[ r1 g1 b1 a1 ] [r2 g2 b2 a2] [r3 g3 b3 a3] [ r4 g4 b4 a4 ]]
      r_c_2_l_1_u8 = (uint8_t *)&cur_img(i + 0, j + 4);
      r_c_2_l_2_u8 = (uint8_t *)&cur_img(i + 1, j + 4);

      // 4.
      /*
       * Promote the 8-bit components into 16-bit components to perform the accumulation
       * First we extract the lower and higher part of the 8-bit components using the
       * mipp::low and mipp::high.
       *
       */
      // first line
      r_c_2_l_0_u16_l = mipp::low(r_c_2_l_0_u8);
      r_c_2_l_0_u16_h = mipp::high(r_c_2_l_0_u8);

      // second line
      r_c_2_l_1_u16_l = mipp::low(r_c_2_l_1_u8);
      r_c_2_l_1_u16_h = mipp::high(r_c_2_l_1_u8);

      // third line
      r_c_2_l_2_u16_l = mipp::low(r_c_2_l_2_u8);
      r_c_2_l_2_u16_h = mipp::high(r_c_2_l_2_u8);

      // 5. Compute the reduction of the second column
      /*
       * Accumulate the color component of the right column-group.
       */
      r_c_2_l_1_u16_l = r_c_2_l_0_u16_l + r_c_2_l_1_u16_l + r_c_2_l_2_u16_l;
      r_c_2_l_1_u16_h = r_c_2_l_0_u16_h + r_c_2_l_1_u16_h + r_c_2_l_2_u16_h;

      // 6. left-right pattern
      /*
       * Perform the left-right pattern to compute the sum of the pixel components.
       * Now, since colors are interleaved, we need to shift 4 colors when preparing the
       * left and right part of the sum.
       */
      r_left_l = mipp::combine<4>(r_c_0_l_1_u16_h, r_c_1_l_1_u16_l);
      r_right_l = mipp::combine<4>(r_c_1_l_1_u16_l, r_c_1_l_1_u16_h);

      r_left_h = mipp::combine<4>(r_c_1_l_1_u16_l, r_c_1_l_1_u16_h);
      r_right_h = mipp::combine<4>(r_c_1_l_1_u16_h, r_c_2_l_1_u16_l);

      // store the previous value before overwrite
      r_c_1_l_1_u16_h_temp = r_c_1_l_1_u16_h;

      // vertical sum
      r_c_1_l_1_u16_l = r_left_l + r_c_1_l_1_u16_l + r_right_l; // sum of the lower part
      r_c_1_l_1_u16_h = r_left_h + r_c_1_l_1_u16_h + r_right_h; // sum of the higher part

      // performing the division
      r_c_1_l_1_u16_l = mipp_vdiv9(r_c_1_l_1_u16_l);
      r_c_1_l_1_u16_h = mipp_vdiv9(r_c_1_l_1_u16_h);

      // 11. convert back to uint8
      r_sum_u8 = {r_c_1_l_1_u16_l, r_c_1_l_1_u16_h};

      // 12. store
      // store back the data
      r_sum_u8.store((uint8_t *)&next_img(i, j));

      // 13. variable rotation
      // copy the lowest part of the middle column into the highest part of the left column ( we should pass the sum of the pixels)
      // col 0 <- col 1
      r_c_0_l_1_u16_h = r_c_1_l_1_u16_h_temp;
      // col 1 <- col 2
      r_c_1_l_1_u16_l = r_c_2_l_1_u16_l;
      r_c_1_l_1_u16_h = r_c_2_l_1_u16_h;
    }
  }

  // left-right borders size
  uint32_t bsize = 1;
  // compute the borders
  compute_borders(x, y, width, height, bsize);

  return 0;
}

EXTERN int blur2_do_tile_urrot2_mipp_div8_u16(int x, int y, int width, int height)
{
  /* #########################################################################
   * #########################################################################
   * Variables notation: ^r(a[0-4])?_c_[0-3]_l_[0-2]_u[8,16,32,64]
   * r -> register
   * a[0-4] -> array of registers and its dimension
   * c_[0-2] -> column number
   * l_[0-2] -> line number
   * u[8,16,32,64] -> type of the registers
   *
   * Example ra4_c_2_l_0_u8 -> array of four registers u_0, right column, first line
   * #########################################################################
   * #########################################################################
   */

  /*
   * Contain the interleaved pixel colors of the right group column.
   * Four pixels are interleaved in a single register.
   */
  mipp::Reg<uint8_t> r_c_0_l_0_u8, r_c_0_l_1_u8, r_c_0_l_2_u8;
  mipp::Reg<uint8_t> r_c_1_l_0_u8, r_c_1_l_1_u8, r_c_1_l_2_u8;
  mipp::Reg<uint8_t> r_c_2_l_0_u8, r_c_2_l_1_u8, r_c_2_l_2_u8;
  mipp::Reg<uint8_t> r_sum_u8;

  /**
   * Two array, one for the lower and one for higher part, for each
   * line and for each column.
   */
  mipp::Reg<uint16_t> r_c_0_l_0_u16_l;                  // left column, first line, no need for higher part
  mipp::Reg<uint16_t> r_c_0_l_1_u16_h, r_c_0_l_1_u16_l; // left column, second line
  mipp::Reg<uint16_t> r_c_0_l_2_u16_l;                  // left column, third line, no need for higher part

  mipp::Reg<uint16_t> r_c_1_l_0_u16_h, r_c_1_l_0_u16_l; // central column, first line
  mipp::Reg<uint16_t> r_c_1_l_1_u16_h, r_c_1_l_1_u16_l; // central column, second line
  mipp::Reg<uint16_t> r_c_1_l_2_u16_h, r_c_1_l_2_u16_l; // central column, third line

  mipp::Reg<uint16_t> r_c_2_l_0_u16_h, r_c_2_l_0_u16_l; // right column, first line
  mipp::Reg<uint16_t> r_c_2_l_1_u16_h, r_c_2_l_1_u16_l; // right column, second line
  mipp::Reg<uint16_t> r_c_2_l_2_u16_h, r_c_2_l_2_u16_l; // right column, third line

  // for storing the sum of the pixels on the higher part of the central column
  // used in variable reduction
  mipp::Reg<uint16_t> r_c_1_l_1_u16_h_temp;

  // To store shiffted values for the reduction
  mipp::Reg<uint16_t> r_left_l, r_left_h;
  mipp::Reg<uint16_t> r_right_l, r_right_h;

  // store the central column in the first iteration
  mipp::Reg<uint16_t> r_c_1_l_1_u16_l_central, r_c_1_l_1_u16_h_central;
  mipp::Reg<uint16_t> r_c_2_l_1_u16_l_central, r_c_2_l_1_u16_h_central;


  /*
   * Our code should be portable now, so the number of pixels loaded
   * at each iteration depends on the ISA of the target architecture.
   * 
   * We can use the mipp::N<uint8_t> to get the number of pixels loaded
   * at each iteration and use it to increment the x-loop.
   */
  constexpr int inc = mipp::N<uint8_t>() / 4; // number of pixels loaded at each iteration

  // loop over y (start from +1, end at -1 => no border)
  for (int i = y + 1; i < y + height - 1; i++)
  {
    // #############################################################
    //                      PROLOGUE
    // #############################################################
    /*
     * In order to start the variable rotation, we need to precompute
     * the left and central columns. The computation of the central column
     * is done in the same way as the right column in the main loop.
     * Since the x-loop starts from the second pixel, the left column
     * is composed by only the first pixel of each line. So, one
     * possible strategy is to load to perform the exact same computation
     * as the central column, but starting from the first pixel. After that,
     * we can use vextq_u16 to shift the first pixel to the last position.
     *
     * In this way, we can use simd instructions to perform the computation
     * also for the first mipp::N<uint8_t> / 4 pixels of each line instead 
     * of using scalar instructions.
     */
    {
      r_c_0_l_0_u8 = (uint8_t *)&cur_img(i - 1, x);
      r_c_0_l_1_u8 = (uint8_t *)&cur_img(i + 0, x);
      r_c_0_l_2_u8 = (uint8_t *)&cur_img(i + 1, x);

      r_c_1_l_0_u8 = (uint8_t *)&cur_img(i - 1, x + 1);
      r_c_1_l_1_u8 = (uint8_t *)&cur_img(i + 0, x + 1);
      r_c_1_l_2_u8 = (uint8_t *)&cur_img(i + 1, x + 1);

      /*
       * please note that we need only the lower part of the mipp::N<uint8_t> / 4
       * pixels for the left column. The higher part is not used in the
       * computation but only to store the first pixel for the computation 
       * in the loop.
       */
      // first line
      r_c_0_l_0_u16_l = mipp::low(r_c_0_l_0_u8);

      r_c_1_l_0_u16_l = mipp::low(r_c_1_l_0_u8);
      r_c_1_l_0_u16_h = mipp::high(r_c_1_l_0_u8);

      // second line
      r_c_0_l_1_u16_l = mipp::low(r_c_0_l_1_u8);

      r_c_1_l_1_u16_l = mipp::low(r_c_1_l_1_u8);
      r_c_1_l_1_u16_h = mipp::high(r_c_1_l_1_u8);

      // NOTE: in the first we need to store the central pixels 
      // otherwise the variable rotation will not work.
      r_c_1_l_1_u16_l_central = r_c_1_l_1_u16_l;
      r_c_1_l_1_u16_h_central = r_c_1_l_1_u16_h;

      // third line
      r_c_0_l_2_u16_l = mipp::low(r_c_0_l_2_u8);

      r_c_1_l_2_u16_l = mipp::low(r_c_1_l_2_u8);
      r_c_1_l_2_u16_h = mipp::high(r_c_1_l_2_u8);

      // reduction
      r_c_0_l_1_u16_l = r_c_0_l_2_u16_l + r_c_0_l_1_u16_l + r_c_0_l_0_u16_l; // lower part

      r_c_1_l_1_u16_l = r_c_1_l_2_u16_l + r_c_1_l_1_u16_l + r_c_1_l_0_u16_l; // lower part
      r_c_1_l_1_u16_h = r_c_1_l_2_u16_h + r_c_1_l_1_u16_h + r_c_1_l_0_u16_h; // higher part

      // move the first bit to the last position using vextq_u16
      r_c_0_l_1_u16_h = mipp::combine<4>(r_c_0_l_1_u16_h, r_c_0_l_1_u16_l);
    }
    // #############################################################
    //                      END PROLOGUE
    // #############################################################

    for (int j = x + 1; j < x + width; j += inc)
    {

      // 3. Memory load
      /*
       * Mipp load is interleaved by default.
       * For all the three lines of the right column-group, we load from memory 
       * mipp::N<uint8_t> / 4 pixels, leaving interleaved the colors.
       * Now each register is composed by mipp::N<uint8_t> / 4 pixels.
       */
      r_c_2_l_0_u8 = (uint8_t *)&cur_img(i - 1, j + 4); // [[ r1 g1 b1 a1 ] [r2 g2 b2 a2] [r3 g3 b3 a3] [ r4 g4 b4 a4 ]]
      r_c_2_l_1_u8 = (uint8_t *)&cur_img(i + 0, j + 4);
      r_c_2_l_2_u8 = (uint8_t *)&cur_img(i + 1, j + 4);

      // 4.
      /*
       * Promote the 8-bit components into 16-bit components to perform the accumulation
       * First we extract the lower and higher part of the 8-bit components using the
       * mipp::low and mipp::high.
       *
       */
      // first line
      r_c_2_l_0_u16_l = mipp::low(r_c_2_l_0_u8);
      r_c_2_l_0_u16_h = mipp::high(r_c_2_l_0_u8);

      // second line
      r_c_2_l_1_u16_l = mipp::low(r_c_2_l_1_u8);
      r_c_2_l_1_u16_h = mipp::high(r_c_2_l_1_u8);

      // third line
      r_c_2_l_2_u16_l = mipp::low(r_c_2_l_2_u8);
      r_c_2_l_2_u16_h = mipp::high(r_c_2_l_2_u8);

      // save the central line before overwriting
      r_c_2_l_1_u16_l_central = r_c_2_l_1_u16_l;
      r_c_2_l_1_u16_h_central = r_c_2_l_1_u16_h;


      // 5. Compute the reduction of the second column
      /*
       * Accumulate the color component of the right column-group.
       */
      r_c_2_l_1_u16_l = r_c_2_l_0_u16_l + r_c_2_l_1_u16_l + r_c_2_l_2_u16_l;
      r_c_2_l_1_u16_h = r_c_2_l_0_u16_h + r_c_2_l_1_u16_h + r_c_2_l_2_u16_h;

      // 6. left-right pattern
      /*
       * Perform the left-right pattern to compute the sum of the pixel components.
       * Now, since colors are interleaved, we need to shift 4 colors when preparing the
       * left and right part of the sum.
       */
      r_left_l = mipp::combine<4>(r_c_0_l_1_u16_h, r_c_1_l_1_u16_l);
      r_right_l = mipp::combine<4>(r_c_1_l_1_u16_l, r_c_1_l_1_u16_h);

      r_left_h = mipp::combine<4>(r_c_1_l_1_u16_l, r_c_1_l_1_u16_h);
      r_right_h = mipp::combine<4>(r_c_1_l_1_u16_h, r_c_2_l_1_u16_l);

      // store the previous value before overwrite
      r_c_1_l_1_u16_h_temp = r_c_1_l_1_u16_h;

      // vertical sum
      r_c_1_l_1_u16_l = r_left_l + r_c_1_l_1_u16_l + r_right_l; // sum of the lower part
      r_c_1_l_1_u16_h = r_left_h + r_c_1_l_1_u16_h + r_right_h; // sum of the higher part

      // remove the central line
      r_c_1_l_1_u16_l = r_c_1_l_1_u16_l - r_c_1_l_1_u16_l_central;
      r_c_1_l_1_u16_h = r_c_1_l_1_u16_h - r_c_1_l_1_u16_h_central;

      // division by 8 using just a vectorized shift operation
      r_c_1_l_1_u16_l = r_c_1_l_1_u16_l >> 3;
      r_c_1_l_1_u16_h = r_c_1_l_1_u16_h >> 3;

      // 11. convert back to uint8
      r_sum_u8 = {r_c_1_l_1_u16_l, r_c_1_l_1_u16_h};

      // 12. store
      // store back the data
      r_sum_u8.store((uint8_t *)&next_img(i, j));

      // 13. variable rotation
      // copy the lowest part of the middle column into the highest part of the left column ( we should pass the sum of the pixels)
      // col 0 <- col 1
      r_c_0_l_1_u16_h = r_c_1_l_1_u16_h_temp; 

      // pass the central column
      r_c_1_l_1_u16_l_central = r_c_2_l_1_u16_l_central;
      r_c_1_l_1_u16_h_central = r_c_2_l_1_u16_h_central;

      // col 1 <- col 2
      r_c_1_l_1_u16_l = r_c_2_l_1_u16_l;
      r_c_1_l_1_u16_h = r_c_2_l_1_u16_h;
    }
  }

  // left-right borders size
  uint32_t bsize = 1;
  // compute the borders
  compute_borders(x, y, width, height, bsize);

  return 0;
}

#endif