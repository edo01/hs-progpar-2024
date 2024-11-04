#ifndef FEATURES_CCA_HPP
#define FEATURES_CCA_HPP

#include <streampu.hpp>
#include "motion/tools.h" 
#include "motion/features.h"        

/*
AUTHORS: MENGQIAN XU (21306077), EDOARDO CARRA' (21400562)
BOARD ID: Q
*/

class Features_CCA : public spu::module::Stateful {
public:
    /**
     * Constructor to initialize the Features_CCA module
     * @param i0 Starting y-coordinate of the image
     * @param i1 Ending y-coordinate of the image
     * @param j0 Starting x-coordinate of the image
     * @param j1 Ending x-coordinate of the image
     * @param p_cca_roi_max1 Maximum number of regions of interest
     */
    Features_CCA(int i0, int i1, int j0, int j1, int p_cca_roi_max1);

private:
    int i0, i1, j0, j1, p_cca_roi_max1;
};

#endif 
