#ifndef FEATURES_CCA_HPP
#define FEATURES_CCA_HPP

#include "spu/module/stateful.hpp"   
#include "motion/features.h"        

class Features_CCA : public spu::module::Stateful {
public:
    /**
     * Constructor to initialize the Features_CCA module
     * @param i0 Starting y-coordinate of the image
     * @param i1 Ending y-coordinate of the image
     * @param j0 Starting x-coordinate of the image
     * @param j1 Ending x-coordinate of the image
     * @param max_rois Maximum number of regions of interest 
     */
    Features_CCA(int i0, int i1, int j0, int j1, int max_rois);

private:
    int i0, i1, j0, j1; 
    int max_rois;  
};

#endif 
