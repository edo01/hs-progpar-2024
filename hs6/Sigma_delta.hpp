#pragma once

#include <streampu.hpp>
#include "motion/sigma_delta.h"
#include "motion/tools.h"

class Sigma_delta : public spu::module::Stateful {
public:
    /**
     * Constructor to initialize the Sigma_delta module
     * @param sd_data Pointer to the sigma_delta_data_t structure used for Sigma-Delta algorithm
     * @param i0 Starting y-coordinate of the image
     * @param i1 Ending y-coordinate of the image
     * @param j0 Starting x-coordinate of the image
     * @param j1 Ending x-coordinate of the image
     * @param p_sd_n Value of the N parameter in the Sigma-Delta algorithm 
     */
    Sigma_delta(sigma_delta_data_t* sd_data, int i0, int i1, int j0,
                    int j1, int p_sd_n);
private:
    int i0, i1, j0, j1, p_sd_n;
    sigma_delta_data_t* sd_data;
};