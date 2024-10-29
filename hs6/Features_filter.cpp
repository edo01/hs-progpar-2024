#include "motion/wrapper/Features_filter.hpp"

using namespace spu;
// n_RoIs0 = features_filter_surface((const uint32_t**)L10, L20, i0, i1, j0, j1, RoIs_tmp0, 
//                                                 n_RoIs_tmp0, p_flt_s_min, p_flt_s_max);

Features_filter::Features_filter(int i0, int i1, int j0, int j1,
                    int n_RoIs_tmp,  int s_min,  int s_max, int p_cca_roi_max2)
    : spu::module::Stateful(), i0(i0), i1(i1), j0(j0), j1(j1), n_RoIs_tmp(n_RoIs_tmp), s_min(s_min),
     s_max(s_max), p_cca_roi_max2(p_cca_roi_max2)
{
    const std::string name = "Features_filter";
    this->set_name(name);
    this->set_short_name(name);

    auto &t = this->create_task("filter");
    // input socket
    size_t si_data_L1 = this->template create_2d_sck_in<uint32_t>(t, "in_L1", (i1 - i0) + 1, (j1 - j0) + 1); 
    
    // it is accessed in write mode in the method and so we must treat it as an output socket
    size_t so_RoIs_tmp = this->template create_socket_out<uint8_t>(t, "in_RoIs_tmp", n_RoIs_tmp * sizeof(RoI_t));
    
    size_t so_data_L2 = 0;
    // output socket
    //size_t so_data_L2 = this->template create_2d_sck_out<uint32_t>(t, "out_L2", (i1 - i0) + 1, (j1 - j0) + 1); 
    // we don't know the size of the output RoIs, so we allocate the maximum size
    size_t so_RoIs = this->template create_socket_out<uint8_t>(t, "out_RoIs", p_cca_roi_max2 * sizeof(RoI_t));
    uint32_t so_n_RoIs = this->template create_socket_out<uint32_t>(t, "out_n_RoIs", 1);

    create_codelet(t, 
        [si_data_L1, so_RoIs_tmp, so_data_L2, so_RoIs, so_n_RoIs] 
        (Module &m, spu::runtime::Task &tsk, size_t frame) -> int 
        {
            Features_filter features_filter = static_cast<Features_filter&>(m);
            
            // Get the input and output data pointers from the task
            const uint32_t** L1_in = tsk[si_data_L1].get_2d_dataptr<const uint32_t>();
            // not const
            RoI_t* RoIs_tmp_in = (RoI_t*)tsk[so_RoIs_tmp].get_dataptr<uint8_t>();

            //uint32_t** L2_out =  tsk[so_data_L2].get_2d_dataptr<uint32_t>();

            RoI_t* RoIs_out = (RoI_t*)tsk[so_RoIs].get_dataptr<uint8_t>();
            uint32_t* n_RoIs_out = tsk[so_n_RoIs].get_dataptr<uint32_t>();

            *n_RoIs_out = features_filter_surface(L1_in, NULL, 
                                    features_filter.i0, features_filter.i1, 
                                    features_filter.j0, features_filter.j1,
                                    RoIs_tmp_in, features_filter.n_RoIs_tmp,
                                    features_filter.s_min, features_filter.s_max);
            
            assert(*n_RoIs_out <= (uint32_t)features_filter.p_cca_roi_max2);

            features_shrink_basic(RoIs_tmp_in, features_filter.n_RoIs_tmp, RoIs_out);

            return runtime::status_t::SUCCESS;
        }
    );
}
