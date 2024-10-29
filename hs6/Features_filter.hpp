#ifndef FEATURES_FILTER_HPP
#define FEATURES_FILTER_HPP

#include <streampu.hpp>
#include "motion/tools.h" 
#include "motion/features.h"    

class Features_filter : public spu::module::Stateful {
public:
    /**
     * Constructor to initialize the Features_filter module
     * @param i0 Starting y-coordinate of the image
     * @param i1 Ending y-coordinate of the image
     * @param j0 Starting x-coordinate of the image
     * @param j1 Ending x-coordinate of the image
     * @param n_RoIs Number of connected components
     * @param s_min Minimum surface area for RoIs
     * @param s_max Maximum surface area for RoIs
     * @param p_cca_roi_max2 Maximum number of RoIs
     */
    Features_filter(const int i0, const int i1, const int j0, const int j1,
                const int n_RoIs_tmp, const int s_min, const int s_max,
                const int p_cca_roi_max2);

private:
    const int i0, i1, j0, j1; 
    const size_t n_RoIs_tmp; 
    const int s_min, s_max;  
    const int p_cca_roi_max2;

};

#endif 
