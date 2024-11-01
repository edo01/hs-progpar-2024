//tracking_perform(tracking_data, RoIs1, n_RoIs1, cur_fra, p_trk_ext_d, p_trk_obj_min,
//                         p_trk_roi_path != NULL || visu, p_trk_ext_o, p_knn_s);
//const RoI_t *RoIs, size_t n_RoIs, size_t frame, size_t r_extrapol, size_t fra_obj_min, 
// uint8_t save_RoIs_id, uint8_t extrapol_order_max, float min_extrapol_ratio_S)
#ifndef TRACKING_HPP
#define TRACKING_HPP

#include <streampu.hpp>
#include "motion/tools.h" 
#include "motion/tracking.h"   

class Tracking : public spu::module::Stateful {
public:
    Tracking(tracking_data_t* tracking_data, int p_cca_roi_max2, size_t frame, size_t r_extrapol, 
            size_t fra_obj_min, uint8_t save_RoIs_id, uint8_t extrapol_order_max, 
            float min_extrapol_ratio_S); 

private:
    tracking_data_t* tracking_data;
    size_t p_cca_roi_max2;
    size_t frame;
    size_t r_extrapol;
    size_t fra_obj_min;
    uint8_t save_RoIs_id;
    uint8_t extrapol_order_max;
    float min_extrapol_ratio_S;
};

#endif 
