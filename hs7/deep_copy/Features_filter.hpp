#ifndef FEATURES_FILTER_HPP
#define FEATURES_FILTER_HPP

#include <streampu.hpp>
#include "motion/tools.h" 
#include "motion/features.h"    

class Features_filter : public spu::module::Stateful {
    private:
        int i0, i1, j0, j1; 
        int p_cca_roi_max1; 
        int s_min, s_max;  
        int p_cca_roi_max2;
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
        /*Features_filter( int i0,  int i1, int j0, int j1,
                    int p_cca_roi_max1, int s_min, int s_max,
                    int p_cca_roi_max2);
        */
    
        
        Features_filter* Features_filter::clone() const {
            auto f = new Features_filter(*this);  
            f->deep_copy(*this);                  
            return f;
        }

        void Features_filter::deep_copy(const Features_filter& f) {
            Stateful::deep_copy(f);  
            this->filter_data = filter_alloc_data(f.filter_data->i0, f.filter_data->i1, f.filter_data->j0, f.filter_data->j1,
                                                f.filter_data->p_cca_roi_max1, f.filter_data->p_flt_s_min, f.filter_data->p_flt_s_max, f.filter_data->p_cca_roi_max2);
                                                
            filter_init_data(this->filter_data);
        }
};

#endif 
