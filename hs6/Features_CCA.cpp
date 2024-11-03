#include "motion/wrapper/Features_CCA.hpp"

/*
AUTHORS: MENGQIAN XU (21306077), EDOARDO CARRA' (21400562)
BOARD ID: Q
*/
using namespace spu;

Features_CCA::Features_CCA(int i0, int i1, int j0, int j1, int p_cca_roi_max1)
    : spu::module::Stateful(), i0(i0), i1(i1), j0(j0), j1(j1), p_cca_roi_max1(p_cca_roi_max1) 
{
    const std::string name = "Features_CCA";
    this->set_name(name);
    this->set_short_name(name);

    auto &t = this->create_task("extract");

    // input socket
    size_t si_labels = this->template create_2d_sck_in<uint32_t>(t, "in_labels", (i1 - i0) + 1, (j1 - j0) + 1); 
    size_t si_n_RoIs = this->template create_socket_in<uint32_t>(t, "in_n_RoIs", 1);
    // output socket
    size_t so_RoIs = this->template create_socket_out<uint8_t>(t, "out_RoIs", p_cca_roi_max1 * sizeof(RoI_t));

    create_codelet(t, 
        [si_labels, si_n_RoIs, so_RoIs] 
        (Module &m, spu::runtime::Task &tsk, size_t frame) -> int 
        {
            Features_CCA features_CCA = static_cast<Features_CCA&>(m);

            // Get the input and output data pointers from the task
            const uint32_t** labels_in = tsk[si_labels].get_2d_dataptr<const uint32_t>();
            const uint32_t* n_RoIs_in = tsk[si_n_RoIs].get_dataptr<const uint32_t>();
            RoI_t* RoIs_out = (RoI_t*)tsk[so_RoIs].get_dataptr<uint8_t>();
            
            features_extract(labels_in, features_CCA.i0, features_CCA.i1, features_CCA.j0, features_CCA.j1, RoIs_out, *n_RoIs_in);

            return runtime::status_t::SUCCESS;
        }
    );
}
