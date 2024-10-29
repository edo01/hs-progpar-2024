#ifndef CCL_HPP
#define CCL_HPP

#include <streampu.hpp>
#include "motion/tools.h" 
#include "motion/CCL.h"              

class CCL : public spu::module::Stateful {
public:
    /**
     * Constructor to initialize the CCL module
     * @param ccl_data Pointer to the CCL_data_t structure used for CCL operations
     * @param i0 Starting y-coordinate of the image
     * @param i1 Ending y-coordinate of the image
     * @param j0 Starting x-coordinate of the image
     * @param j1 Ending x-coordinate of the image
     */
    CCL(CCL_data_t* ccl_data, int i0, int i1, int j0, int j1);

private:
    CCL_data_t* ccl_data;  
    int i0, i1, j0, j1;  
};

#endif 
