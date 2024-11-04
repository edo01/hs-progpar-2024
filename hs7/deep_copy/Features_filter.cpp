#include "motion/wrapper/Features_filter.hpp"

using namespace spu;
// n_RoIs0 = features_filter_surface((const uint32_t**)L10, L20, i0, i1, j0, j1, RoIs_tmp0, 
//                                                 n_RoIs_tmp0, p_flt_s_min, p_flt_s_max);

Features_filter::Features_filter(int i0, int i1, int j0, int j1,
                    int p_cca_roi_max1,  int s_min,  int s_max, int p_cca_roi_max2)
    : spu::module::Stateful(), i0(i0), i1(i1), j0(j0), j1(j1), p_cca_roi_max1(p_cca_roi_max1), s_min(s_min),
     s_max(s_max), p_cca_roi_max2(p_cca_roi_max2)
{
    const std::string name = "Features_filter";
    this->set_name(name);
    this->set_short_name(name);

    auto &t = this->create_task("filter");
    // input socket
    size_t si_labels_in = this->template create_2d_sck_in<uint32_t>(t, "in_labels", (i1 - i0) + 1, (j1 - j0) + 1); 
    
    size_t so_n_RoIs_in = this->template create_socket_in<uint32_t>(t, "in_n_RoIs", 1);
    size_t so_RoIs_in = this->template create_socket_in<uint8_t>(t, "in_RoIs", p_cca_roi_max1 * sizeof(RoI_t));
    // output socket
    size_t so_labels_out = this->template create_2d_sck_out<uint32_t>(t, "out_labels", (i1 - i0) + 1, (j1 - j0) + 1); 
    // we don't know the size of the output RoIs, so we allocate the maximum size
    size_t so_RoIs_out = this->template create_socket_out<uint8_t>(t, "out_RoIs", p_cca_roi_max2 * sizeof(RoI_t));
    uint32_t so_n_RoIs_out = this->template create_socket_out<uint32_t>(t, "out_n_RoIs", 1);

    create_codelet(t, 
        [si_labels_in, so_n_RoIs_in, so_RoIs_in, so_labels_out, so_RoIs_out, so_n_RoIs_out] 
        (Module &m, spu::runtime::Task &tsk, size_t frame) -> int 
        {
            Features_filter features_filter = static_cast<Features_filter&>(m);
            
            // Get the input and output data pointers from the task
            const uint32_t** labels_in = tsk[si_labels_in].get_2d_dataptr<const uint32_t>();
            const uint32_t* n_RoIs_in = tsk[so_n_RoIs_in].get_dataptr<const uint32_t>();
            // not const
            const RoI_t* RoIs_in = (RoI_t*)tsk[so_RoIs_in].get_dataptr<const uint8_t>();
            
            uint32_t** labels_out =  tsk[so_labels_out].get_2d_dataptr<uint32_t>();

            RoI_t* RoIs_out = (RoI_t*)tsk[so_RoIs_out].get_dataptr<uint8_t>();
            uint32_t* n_RoIs_out = tsk[so_n_RoIs_out].get_dataptr<uint32_t>();

            *n_RoIs_out = features_filter_surface(labels_in, labels_out, 
                                    features_filter.i0, features_filter.i1, 
                                    features_filter.j0, features_filter.j1,
                                    (RoI_t*)RoIs_in, *n_RoIs_in,
                                    features_filter.s_min, features_filter.s_max);
            
            assert(*n_RoIs_out <= (uint32_t)features_filter.p_cca_roi_max2);

            features_shrink_basic((RoI_t*)RoIs_in, *n_RoIs_in, RoIs_out);

            return runtime::status_t::SUCCESS;
        }
    );
}

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