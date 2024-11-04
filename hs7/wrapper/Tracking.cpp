#include "motion/wrapper/Tracking.hpp"

using namespace spu;

Tracking:: Tracking(tracking_data_t* tracking_data, int p_cca_roi_max2, size_t r_extrapol, 
            size_t fra_obj_min, uint8_t save_RoIs_id, uint8_t extrapol_order_max, 
            float min_extrapol_ratio_S)
    : spu::module::Stateful(), tracking_data(tracking_data), p_cca_roi_max2(p_cca_roi_max2), r_extrapol(r_extrapol), 
        fra_obj_min(fra_obj_min), save_RoIs_id(save_RoIs_id), extrapol_order_max(extrapol_order_max), 
        min_extrapol_ratio_S(min_extrapol_ratio_S)
    
{
    const std::string name = "Tracking";
    this->set_name(name);
    this->set_short_name(name);

    auto &t = this->create_task("perform");
    
    // output socket
    size_t si_n_RoIs = this->template create_socket_in<uint32_t>(t, "in_n_RoIs", 1);
    size_t si_RoIs = this->template create_socket_in<uint8_t>(t, "in_RoIs", p_cca_roi_max2* sizeof(RoI_t));
    size_t si_frame = this->template create_socket_in<uint32_t>(t, "in_frame", 1);

    create_codelet(t, 
        [si_n_RoIs, si_RoIs, si_frame]
        (Module &m, spu::runtime::Task &tsk, size_t frame) -> int 
        {
            Tracking tracking = static_cast<Tracking&>(m);
            
            // Get the input and output data pointers from the task
            const RoI_t* RoIs_in = (RoI_t*)tsk[si_RoIs].get_dataptr<const uint8_t>();
            const uint32_t* n_RoIs_in = tsk[si_n_RoIs].get_dataptr<const uint32_t>();
            const uint32_t* frame_in = tsk[si_frame].get_dataptr<const uint32_t>();

            tracking_perform(tracking.tracking_data, RoIs_in, 
                        *n_RoIs_in, *frame_in, tracking.r_extrapol, 
                        tracking.fra_obj_min, tracking.save_RoIs_id, tracking.extrapol_order_max,
                        tracking.min_extrapol_ratio_S);

            return runtime::status_t::SUCCESS;
        }
    );
}
