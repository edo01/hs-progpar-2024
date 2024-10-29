#include "motion/wrapper/CCL.hpp"

using namespace spu;

CCL::CCL(CCL_data_t* ccl_data, int i0, int i1, int j0, int j1)
    : spu::module::Stateful(), ccl_data(ccl_data), i0(i0), i1(i1), j0(j0), j1(j1) 
{
    const std::string name = "CCL";
    this->set_name(name);
    this->set_short_name(name);

    auto &t = this->create_task("CCL_compute");

    // input socket
    size_t si_data_IB = create_2d_sck_in<uint8_t>(t, "in_IB", (i1 - i0), (j1 - j0)); 
    // output socket
    size_t so_data_L1 = create_2d_sck_out<uint32_t>(t, "out_L1", (i1 - i0), (j1 - j0)); 
    size_t so_data_n_RoIs_tmp0 = create_sck_out<int>(t, "out_n_RoIs_tmp0", 1);

    create_codelet(t, 
        [si_data_IB, so_data_L1, so_data_n_RoIs_tmp0] 
        (Module &m, spu::runtime::Task &tsk, size_t frame) -> int 
        {
            CCL ccl = static_cast<CCL&>(m);        

            // Get the input and output data pointers from the task
            const uint8_t** IB_in = tsk[si_data_IB].get_2d_dataptr<const uint8_t>();
            uint32_t** L1_out = tsk[so_data_L1].get_2d_dataptr<uint32_t>();
            int* n_RoIs_tmp0_out = tsk[so_data_n_RoIs_tmp0].get_dataptr<int>();
            
            // Apply the opening and closing morphological operations
            *n_RoIs_tmp0_out = CCL_LSL_apply(ccl.ccl_data, IB_in, L1_out, 0);
            
            return runtime::status_t::SUCCESS;
        }
    );
}
