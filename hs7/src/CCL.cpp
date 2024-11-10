#include "motion/wrapper/CCL.hpp"

using namespace spu;

CCL::CCL(CCL_data_t* ccl_data, int def_p_cca_roi_max)
    : spu::module::Stateful(), ccl_data(ccl_data), def_p_cca_roi_max(def_p_cca_roi_max)
{
    const std::string name = "CCL";
    this->set_name(name);
    this->set_short_name(name);

    auto &t = this->create_task("apply");

    // input socket
    size_t si_img = this->template create_2d_sck_in<uint8_t>(t, "in_img", (ccl_data->i1 - ccl_data->i0) + 1, (ccl_data->j1 - ccl_data->j0) + 1); 
    // output socket
    size_t so_labels = this->template create_2d_sck_out<uint32_t>(t, "out_labels", (ccl_data->i1 - ccl_data->i0) + 1, (ccl_data->j1 - ccl_data->j0) + 1); 
    size_t so_n_RoIs = this->template create_sck_out<uint32_t>(t, "out_n_RoIs", 1);

    create_codelet(t, 
        [si_img, so_labels, so_n_RoIs] 
        (Module &m, spu::runtime::Task &tsk, size_t frame) -> int 
        {
            CCL ccl = static_cast<CCL&>(m);        

            // Get the input and output data pointers from the task
            const uint8_t** img_in = tsk[si_img].get_2d_dataptr<const uint8_t>();
            uint32_t** labels_out = tsk[so_labels].get_2d_dataptr<uint32_t>();

            uint32_t* n_RoIs_out = tsk[so_n_RoIs].get_dataptr<uint32_t>();
            
            // Apply the opening and closing morphological operations
            *n_RoIs_out = CCL_LSL_apply(ccl.ccl_data, img_in, labels_out, 0);
            
            assert(*n_RoIs_out <= (uint32_t)ccl.def_p_cca_roi_max);

            return runtime::status_t::SUCCESS;
        }
    );
}

CCL* CCL::clone() const {
    auto c = new CCL(*this);  
    c->deep_copy(*this);      
    return c;
}

void CCL::deep_copy(const CCL& c) {
    Stateful::deep_copy(c);  
    this->ccl_data = CCL_LSL_alloc_data(c.ccl_data->i0, c.ccl_data->i1,
                                    c.ccl_data->j0, c.ccl_data->j1);
    CCL_LSL_init_data(this->ccl_data);
}
