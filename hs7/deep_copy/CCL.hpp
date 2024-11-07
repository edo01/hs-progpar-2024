#ifndef CCL_HPP
#define CCL_HPP

#include <streampu.hpp>
#include "motion/tools.h" 
#include "motion/CCL.h"              

class CCL : public spu::module::Stateful {
    private:
        CCL_data_t* ccl_data;   
        int def_p_cca_roi_max;
    public:
        /**
         * Constructor to initialize the CCL module
         * @param ccl_data Pointer to the CCL_data_t structure used for CCL operations
         */
        //CCL(CCL_data_t* ccl_data, int def_p_cca_roi_max);

        CCL* CCL::clone() const {
            auto c = new CCL(*this);  
            c->deep_copy(*this);      
            return c;
        }

        void CCL::deep_copy(const CCL& c) {
            Stateful::deep_copy(c);  
            this->ccl_data = ccl_alloc_data(c.ccl_data->i0, c.ccl_data->i1,
                                            c.ccl_data->j0, c.ccl_data->j1);
            ccl_init_data(this->ccl_data);
        }
};

#endif 
