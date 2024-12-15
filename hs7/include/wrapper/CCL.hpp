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
     */
    CCL(CCL_data_t* ccl_data, int def_p_cca_roi_max);
    virtual CCL* clone() const override;
    void deep_copy(const CCL& c);

private:
    CCL_data_t* ccl_data;   
    int def_p_cca_roi_max;

};

#endif 
