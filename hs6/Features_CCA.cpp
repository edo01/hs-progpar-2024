#include "motion/wrapper/Features_CCA.hpp"

using namespace spu;

Features_CCA::Features_CCA(int i0, int i1, int j0, int j1, size_t n_RoIs) 
    : spu::module::Stateful(), i0(i0), i1(i1), j0(j0), j1(j1), n_RoIs(n_RoIs) 
{
    const std::string name = "Features_CCA";
    this->set_name(name);
    this->set_short_name(name);

    auto &t = this->create_task("extract");

    // input socket
    size_t si_data_L1 = this->template create_2d_sck_in<uint32_t>(t, "in_L1", (i1 - i0) + 1, (j1 - j0) + 1); 
    // output socket
    size_t so_RoIs = this->template create_socket_out<uint8_t>(t, "out_RoIs", n_RoIs * sizeof(RoI_t));

    create_codelet(t, 
        [si_data_L1, so_RoIs] 
        (Module &m, spu::runtime::Task &tsk, size_t frame) -> int 
        {
            Features_CCA features_CCA = static_cast<Features_CCA&>(m);

            // Get the input and output data pointers from the task
            const uint32_t** L1_in = tsk[si_data_L1].get_2d_dataptr<const uint32_t>();

            RoI_t* RoIs_out = (RoI_t*)tsk[so_RoIs].get_dataptr<uint8_t>();
            
            features_extract(L1_in, features_CCA.i0, features_CCA.i1, features_CCA.j0, features_CCA.j1, RoIs_out, features_CCA.n_RoIs);

            return runtime::status_t::SUCCESS;
        }
    );
}
