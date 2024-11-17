#include <math.h>
#include <stdlib.h>
#include <nrc2.h>
#include <mipp.h>

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
    
#pragma omp parallel // we generate the threads only once
{
    #pragma omp for schedule(static,3)
    for (int i = i0; i <= i1; i++) {
        for (int j = j0; j <= j1; j+=mipp::N<uint8_t>()) {
            mipp::Reg<uint8_t> M_r(&sd_data->M[i][j]); // uint8_t new_m = sd_data->M[i][j]; 
            mipp::Reg<uint8_t> img_in_r(&img_in[i][j]); // img_in[i][j]

            mipp::Msk<mipp::N<uint8_t>()> mask_lt_r = M_r < img_in_r;
            mipp::Msk<mipp::N<uint8_t>()> mask_gt_r = M_r > img_in_r;

            M_r = mipp::blend(M_r + 1, M_r, mask_lt_r);
            M_r = mipp::blend(M_r - 1, M_r, mask_gt_r);
            M_r.store(&sd_data->M[i][j]);
        }
    }

    #pragma omp for schedule(static,3)
    for (int i = i0; i <= i1; i++) {
        for (int j = j0; j <= j1; j+=mipp::N<int8_t>()) {
            mipp::Reg<uint8_t> M_r(&sd_data->M[i][j]);
            mipp::Reg<uint8_t> img_in_r(&img_in[i][j]);

            mipp::Msk<mipp::N<uint8_t>()> mask_lt_r = M_r > img_in_r;
            mipp::Reg<uint8_t> res = mipp::blend(M_r - img_in_r, img_in_r - M_r, mask_lt_r);
            res.store(&sd_data->O[i][j]);
        }
    }

    #pragma omp for schedule(static,3)
    for (int i = i0; i <= i1; i++) {
        for (int j = j0; j <= j1; j+=mipp::N<int8_t>()) {
            mipp::Reg<uint8_t> V_r(&sd_data->V[i][j]); //uint8_t new_v = sd_data->V[i][j];
            mipp::Reg<uint8_t> O_r(&sd_data->O[i][j]);
            
            mipp::Msk<mipp::N<uint8_t>()> mask_lt_r = V_r < O_r * N;
            mipp::Msk<mipp::N<uint8_t>()> mask_gt_r = V_r > O_r * N;
            V_r = mipp::blend(V_r + 1, V_r, mask_lt_r);
            V_r = mipp::blend(V_r - 1, V_r, mask_gt_r);
            V_r = mipp::max(mipp::min(V_r, mipp::Reg<uint8_t>(sd_data->vmax) ), mipp::Reg<uint8_t>(sd_data->vmin));
            V_r.store(&sd_data->V[i][j]); 
        }
    }

    #pragma omp for schedule(static,3)
    for (int i = i0; i <= i1; i++) {
        for(int j = j0; j <= j1; j+=mipp::N<uint8_t>()){
            mipp::Reg<uint8_t> O_r(&sd_data->O[i][j]);
            mipp::Reg<uint8_t> V_r(&sd_data->V[i][j]);
            
            mipp::Msk<mipp::N<uint8_t>()> mask_gt = O_r < V_r;
            mipp::Reg<uint8_t> res = mipp::blend( mipp::Reg<uint8_t>((uint8_t)0), mipp::Reg<uint8_t>((uint8_t)255), mask_gt);
            res.store(&img_out[i][j]);
        }
    }

}

}
