#include <omp.h>
#include <stdint.h>
#include <mipp.h>
#include <cmath>  


#include <math.h>
#include <stdlib.h>
#include <nrc2.h>

#include "motion/macros.h"
#include "motion/sigma_delta/sigma_delta_compute.h"

sigma_delta_data_t* sigma_delta_alloc_data(const int i0, const int i1, const int j0, const int j1, const uint8_t vmin,
                                           const uint8_t vmax) {
    sigma_delta_data_t* sd_data = (sigma_delta_data_t*)malloc(sizeof(sigma_delta_data_t));
    sd_data->i0 = i0;
    sd_data->i1 = i1;
    sd_data->j0 = j0;
    sd_data->j1 = j1;
    sd_data->vmin = vmin;
    sd_data->vmax = vmax;
    sd_data->M = ui8matrix(sd_data->i0, sd_data->i1, sd_data->j0, sd_data->j1);
    sd_data->O = ui8matrix(sd_data->i0, sd_data->i1, sd_data->j0, sd_data->j1);
    sd_data->V = ui8matrix(sd_data->i0, sd_data->i1, sd_data->j0, sd_data->j1);
    return sd_data;
}

void sigma_delta_init_data(sigma_delta_data_t* sd_data, const uint8_t** img_in, const int i0, const int i1,
                           const int j0, const int j1) {
    for (int i = i0; i <= i1; i++) {
        for (int j = j0; j <= j1; j++) {
            sd_data->M[i][j] = img_in != NULL ? img_in[i][j] : sd_data->vmax;
            sd_data->V[i][j] = sd_data->vmin;
        }
    }
}

void sigma_delta_free_data(sigma_delta_data_t* sd_data) {
    free_ui8matrix(sd_data->M, sd_data->i0, sd_data->i1, sd_data->j0, sd_data->j1);
    free_ui8matrix(sd_data->O, sd_data->i0, sd_data->i1, sd_data->j0, sd_data->j1);
    free_ui8matrix(sd_data->V, sd_data->i0, sd_data->i1, sd_data->j0, sd_data->j1);
    free(sd_data);
}

void sigma_delta_compute(sigma_delta_data_t *sd_data, const uint8_t** img_in, uint8_t** img_out, const int i0,
                         const int i1, const int j0, const int j1, const uint8_t N) {
    
#pragma omp parallel
{
    
    #pragma omp for collapse(2)
    for (int i = i0; i <= i1; i++) {
        for (int j = j0; j <= j1; j++) {
            uint8_t new_m = sd_data->M[i][j];
            if (sd_data->M[i][j] < img_in[i][j])
                new_m += 1;
            else if (sd_data->M[i][j] > img_in[i][j])
                new_m -= 1;
            sd_data->M[i][j] = new_m;
        }
    }

    #pragma omp for collapse(2)
    for (int i = i0; i <= i1; i++) {
        for (int j = j0; j <= j1; j++) {
            sd_data->O[i][j] = abs(sd_data->M[i][j] - img_in[i][j]);
        }
    }

    #pragma omp for collapse(2)
    for (int i = i0; i <= i1; i++) {
        for (int j = j0; j <= j1; j++) {
            uint8_t new_v = sd_data->V[i][j];
            if (sd_data->V[i][j] < N * sd_data->O[i][j])
                new_v += 1;
            else if (sd_data->V[i][j] > N * sd_data->O[i][j])
                new_v -= 1;
            sd_data->V[i][j] = MAX(MIN(new_v, sd_data->vmax), sd_data->vmin);
        }
    }

    #pragma omp for collapse(2)
    for (int i = i0; i <= i1; i++) {
        for (int j = j0; j <= j1; j++) {
            img_out[i][j] = sd_data->O[i][j] < sd_data->V[i][j] ? 0 : 255;
        }
    }

}

}

//version mipp
void sigma_delta_compute_mipp(sigma_delta_data_t *sd_data, const uint8_t** img_in, uint8_t** img_out, const int i0,
                         const int i1, const int j0, const int j1, const uint8_t N) {
    const int width = j1 - j0 + 1;
    const int simd_width = mipp::N<uint8_t>(); //Get the SIMD vector width supported by the current platform

    for (int i = i0; i <= i1; i++) {
        for (int j = j0; j <= j1; j += simd_width) {
            mipp::Reg<uint8_t> M_r(&sd_data->M[i][j]);
            mipp::Reg<uint8_t> img_in_r(&img_in[i][j]);
            mipp::Reg<uint8_t> inc_r = mipp::Reg<uint8_t>(1);

            mipp::Reg<uint8_t> mask_lt_r = M_r < img_in_r;
            mipp::Reg<uint8_t> mask_gt_r = M_r > img_in_r;

            M_r = mipp::blend(mask_lt_r, M_r + inc_r, M_r);
            M_r = mipp::blend(mask_gt_r, M_r - inc_r, M_r);

            M_r.store(&sd_data->M[i][j]);
        }
    }


    for (int i = i0; i <= i1; i++) {
        for (int j = j0; j <= j1; j += simd_width) {
            mipp::Reg<uint8_t> M_r(&sd_data->M[i][j]);
            mipp::Reg<uint8_t> img_in_r(&img_in[i][j]);

            mipp::Reg<uint8_t> O_r = mipp::abs(M_r - img_in_r);
            O_r.store(&sd_data->O[i][j]);
        }
    }

    for (int i = i0; i <= i1; i++) {
        for (int j = j0; j <= j1; j += simd_width) {
            mipp::Reg<uint8_t> V_r(&sd_data->V[i][j]);
            mipp::Reg<uint8_t> O_r(&sd_data->O[i][j]);
            mipp::Reg<uint8_t> inc_r = mipp::Reg<uint8_t>(1);
            mipp::Reg<uint8_t> N_r = mipp::Reg<uint8_t>(N);

            mipp::Reg<uint8_t> target_r = N_r * O_r;

            mipp::Reg<uint8_t> mask_lt_r = V_r < target_r;
            mipp::Reg<uint8_t> mask_gt_r = V_r > target_r;

            V_r = mipp::blend(mask_lt_r, V_r + inc_r, V_r);
            V_r = mipp::blend(mask_gt_r, V_r - inc_r, V_r);

            V_r = mipp::min(mipp::max(V_r, mipp::Reg<uint8_t>(sd_data->vmin)), mipp::Reg<uint8_t>(sd_data->vmax));
            V_r.store(&sd_data->V[i][j]);
        }
    }

    for (int i = i0; i <= i1; i++) {
        for (int j = j0; j <= j1; j += simd_width) {
            mipp::Reg<uint8_t> O_r(&sd_data->O[i][j]);
            mipp::Reg<uint8_t> V_r(&sd_data->V[i][j]);

            mipp::Reg<uint8_t> out_r = mipp::blend(O_r < V_r, mipp::Reg<uint8_t>(0), mipp::Reg<uint8_t>(255));
            out_r.store(&img_out[i][j]);
        }
    }
}
