
#include "easypap.h"

static unsigned MASK = 1;

// The stripes kernel aims at highlighting the behavior of a GPU kernel in the
// presence of code divergence

void stripes_config (char *param)
{
  if (param != NULL) {
    unsigned n = atoi (param);
    if (n >= 0 && n <= 12)
      MASK = 1 << n;
    else
      exit_with_error ("Shift value should be in range 0..12");
  }
}

static unsigned scale_component (unsigned c, float ratio)
{
  unsigned coul;

  coul = c * ratio;
  if (coul > 255)
    coul = 255;

  return coul;
}

static unsigned scale_color (unsigned c, float ratio)
{
  uint8_t r, g, b, a;

  r = ezv_c2r (c);
  g = ezv_c2g (c);
  b = ezv_c2b (c);
  a = ezv_c2a (c);

  r = scale_component (r, ratio);
  g = scale_component (g, ratio);
  b = scale_component (b, ratio);

  return ezv_rgba (r, g, b, a);
}

static unsigned brighten (unsigned c)
{
  for (int i = 0; i < 20; i++)
    c = scale_color (c, 1.5f);

  return c;
}

static unsigned darken (unsigned c)
{
  for (int i = 0; i < 20; i++)
    c = scale_color (c, 0.5f);

  return c;
}

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline(s):
// ./run -l data/img/1024.png -k stripes -v seq -c 4
//
unsigned stripes_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++)
        if (j & MASK)
          cur_img (i, j) = brighten (cur_img (i, j));
        else
          cur_img (i, j) = darken (cur_img (i, j));
  }

  return 0;
}

///////////////////////////// OpenCL version (ocl)
// Suggested cmdline(s):
// ./run -l data/img/1024.png -k stripes -g -c 2 -tw 128 -th 1
//
// See kernel/ocl/stripes.cl file.


