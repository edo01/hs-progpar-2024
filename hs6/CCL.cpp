#include "motion/wrapper/CCL.hpp"

using namespace spu;

CCL::CCL(CCL_data_t* ccl_data)
    : spu::module::Stateful(), ccl_data(ccl_data) 
{
    const std::string name = "CCL";
    this->set_name(name);
    this->set_short_name(name);

    auto &t = this->create_task("apply");

    // input socket
    size_t si_data_IB = this->template create_2d_sck_in<uint8_t>(t, "in_IB", (ccl_data->i1 - ccl_data->i0) + 1, (ccl_data->j1 - ccl_data->j0) + 1); 
    // output socket
    size_t so_data_L1 = this->template create_2d_sck_out<uint32_t>(t, "out_L1", (ccl_data->i1 - ccl_data->i0) + 1, (ccl_data->j1 - ccl_data->j0) + 1); 
    size_t so_data_n_RoIs = this->template create_sck_out<uint32_t>(t, "out_n_RoIs", 1);

    create_codelet(t, 
        [si_data_IB, so_data_L1, so_data_n_RoIs] 
        (Module &m, spu::runtime::Task &tsk, size_t frame) -> int 
        {
            CCL ccl = static_cast<CCL&>(m);        

            // Get the input and output data pointers from the task
            const uint8_t** IB_in = tsk[si_data_IB].get_2d_dataptr<const uint8_t>();
            uint32_t** L1_out = tsk[so_data_L1].get_2d_dataptr<uint32_t>();

            uint32_t* n_RoIs_out = tsk[so_data_n_RoIs].get_dataptr<uint32_t>();
            
            // Apply the opening and closing morphological operations
            *n_RoIs_out = CCL_LSL_apply(ccl.ccl_data, IB_in, L1_out, 0);
            
            return runtime::status_t::SUCCESS;
        }
    );
}
