#include "motion/wrapper/Features_filter.hpp"

using namespace spu;

Features_filter::Features_filter(int i0, int i1, int j0, int j1,
                                 int p_cca_roi_max1, int s_min, int s_max, int p_cca_roi_max2)
    : spu::module::Stateful(), i0(i0), i1(i1), j0(j0), j1(j1), 
      p_cca_roi_max1(p_cca_roi_max1), s_min(s_min), s_max(s_max), p_cca_roi_max2(p_cca_roi_max2)
{
    const std::string name = "Features_filter";
    this->set_name(name);
    this->set_short_name(name);

    auto &t = this->create_task("filterf"); 

    // input socket 
    size_t si_data_L1_in = this->template create_2d_sck_in<uint32_t>(t, "in_labels", (i1 - i0) + 1, (j1 - j0) + 1); 
    size_t so_n_RoIs_in = this->template create_socket_in<uint32_t>(t, "in_n_RoIs", 1);
    
    // forward socket for RoIs
    size_t sf_RoIs = this->template create_socket_fwd<RoI_t>(t, "RoIs", p_cca_roi_max2);

    // output for number of RoIs
    uint32_t so_n_RoIs_out = this->template create_socket_out<uint32_t>(t, "out_n_RoIs", 1);

    create_codelet(t, 
        [si_data_L1_in, so_n_RoIs_in, sf_RoIs, so_n_RoIs_out] 
        (Module &m, spu::runtime::Task &tsk, size_t frame) -> int 
        {
            Features_filter& features_filter = static_cast<Features_filter&>(m);
         
            const uint32_t** L1_in = tsk[si_data_L1_in].get_2d_dataptr<const uint32_t>();
            const uint32_t* n_RoIs_in = tsk[so_n_RoIs_in].get_dataptr<const uint32_t>();
            RoI_t* RoIs_data = tsk[sf_RoIs].get_dataptr<RoI_t>();
            uint32_t* n_RoIs_out = tsk[so_n_RoIs_out].get_dataptr<uint32_t>();

      
            *n_RoIs_out = features_filter_surfacef(L1_in, NULL, 
                                                  features_filter.i0, features_filter.i1, 
                                                  features_filter.j0, features_filter.j1,
                                                  RoIs_data, *n_RoIs_in,
                                                  features_filter.s_min, features_filter.s_max);
            
            assert(*n_RoIs_out <= static_cast<uint32_t>(features_filter.p_cca_roi_max2));

            features_shrink_basic(RoIs_data, *n_RoIs_in, RoIs_data);

            return runtime::status_t::SUCCESS;
        }
    );
}
