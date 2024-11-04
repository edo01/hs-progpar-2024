#ifndef FEATURES_FILTER_HPP
#define FEATURES_FILTER_HPP

#include <streampu.hpp>
#include "motion/tools.h" 
#include "motion/features.h"    

/*
AUTHORS: MENGQIAN XU (21306077), EDOARDO CARRA' (21400562)
BOARD ID: Q
*/
class Features_filter : public spu::module::Stateful {
public:
    /**
     * Constructor to initialize the Features_filter module
     * @param i0 Starting y-coordinate of the image
     * @param i1 Ending y-coordinate of the image
     * @param j0 Starting x-coordinate of the image
     * @param j1 Ending x-coordinate of the image
     * @param p_cca_roi_max1 maximum number of inpu RoIs 
     * @param s_min Minimum surface area for RoIs
     * @param s_max Maximum surface area for RoIs
     * @param p_cca_roi_max2 Maximum number of output RoIs
     */
    Features_filter( int i0,  int i1, int j0, int j1,
                int p_cca_roi_max1, int s_min, int s_max,
                int p_cca_roi_max2);

private:
    int i0, i1, j0, j1; 
    int p_cca_roi_max1; 
    int s_min, s_max;  
    int p_cca_roi_max2;

};

#endif 
