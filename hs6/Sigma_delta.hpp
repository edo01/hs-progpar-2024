#pragma once

#include <streampu.hpp>
#include "motion/sigma_delta.h"
#include "motion/tools.h"

class Sigma_delta : public spu::module::Stateful {
private:
    int i0, i1, j0, j1, p_sd_n;
    sigma_delta_data_t* sd_data;
public:
    Sigma_delta(sigma_delta_data_t* sd_data, int i0, int i1, int j0, int j1, int p_sd_n);
};