#include "motion/wrapper/KNN.hpp"

using namespace spu;

//

KNN::KNN(kNN_data_t* knn_data, int p_cca_roi_max2, int knn_k, uint32_t knn_d, float knn_s)
    : spu::module::Stateful(), p_cca_roi_max2(p_cca_roi_max2), knn_data(knn_data), 
        knn_k(knn_k), knn_d(knn_d), knn_s(knn_s)   
{
    const std::string name = "KNN";
    this->set_name(name);
    this->set_short_name(name);

    auto &match     = this->create_task("match");
    auto &matchf    = this->create_task("matchf");
    
    // -------------------- //
    // -- NORMAL VERSION -- //
    // -------------------- //

    // input data
    size_t si_n_RoIs0   = this->template create_socket_in<uint32_t>(match, "in_n_RoIs0", 1);
    size_t si_RoIs0     = this->template create_socket_in<uint8_t>(match, "in_RoIs0", p_cca_roi_max2* sizeof(RoI_t));
    size_t si_n_RoIs1   = this->template create_socket_in<uint32_t>(match, "in_n_RoIs1", 1);
    size_t si_RoIs1     = this->template create_socket_in<uint8_t>(match, "in_RoIs1", p_cca_roi_max2* sizeof(RoI_t));
    
    //output data
    size_t so_RoIs0      = this->template create_socket_out<uint8_t>(match, "out_RoIs0", p_cca_roi_max2* sizeof(RoI_t));
    size_t so_RoIs1      = this->template create_socket_out<uint8_t>(match, "out_RoIs1", p_cca_roi_max2* sizeof(RoI_t));
    
    size_t so_distances = this->template create_2d_socket_out<float>(match, "out_distances", knn_data->_max_size, knn_data->_max_size);
    size_t so_nearest = this->template create_2d_socket_out<uint32_t>(match, "out_nearest", knn_data->_max_size, knn_data->_max_size);

    // return value
    uint32_t so_n_assoc = this->template create_socket_out<uint32_t>(match, "out_n_assoc", 1);


    // ---------------------- //
    // -- FORWARD VERSION -- //
    // --------------------- //

    size_t si_n_RoIs0_f   = this->template create_socket_in<uint32_t>(matchf, "in_n_RoIs0", 1);
    size_t si_n_RoIs1_f   = this->template create_socket_in<uint32_t>(matchf, "in_n_RoIs1", 1);
    
    // forward data
    size_t sf_RoIs0_f      = this->template create_socket_fwd<uint8_t>(matchf, "fwd_RoIs0", p_cca_roi_max2* sizeof(RoI_t));
    size_t sf_RoIs1_f      = this->template create_socket_fwd<uint8_t>(matchf, "fwd_RoIs1", p_cca_roi_max2* sizeof(RoI_t));

    // return value
    uint32_t so_n_assoc_f = this->template create_socket_out<uint32_t>(matchf, "out_n_assoc", 1);

    create_codelet(matchf, 
        [si_n_RoIs0_f, si_n_RoIs1_f, so_n_assoc_f, sf_RoIs0_f, sf_RoIs1_f]
        (Module &m, spu::runtime::Task &tsk, size_t frame) -> int 
        {
            KNN knn = static_cast<KNN&>(m);
            
            // Get the input and output data pointers from the task
            const uint32_t* n_RoIs0_in = tsk[si_n_RoIs0_f].get_dataptr<const uint32_t>();
            const uint32_t* n_RoIs1_in = tsk[si_n_RoIs1_f].get_dataptr<const uint32_t>();
            
            uint32_t* n_assoc_out = tsk[so_n_assoc_f].get_dataptr<uint32_t>();

            RoI_t* RoIs0_fwd = (RoI_t*)tsk[sf_RoIs0_f].get_dataptr<uint8_t>();
            RoI_t* RoIs1_fwd = (RoI_t*)tsk[sf_RoIs1_f].get_dataptr<uint8_t>();

            *n_assoc_out = kNN_match(knn.knn_data, RoIs0_fwd, *n_RoIs0_in, RoIs1_fwd,
                                        *n_RoIs1_in, knn.knn_k, knn.knn_d, knn.knn_s); 

            return runtime::status_t::SUCCESS;
        }
    );
    
    create_codelet(match, 
        [si_n_RoIs0, si_RoIs0, si_n_RoIs1, si_RoIs1, so_n_assoc, so_RoIs0, so_RoIs1, so_distances, so_nearest]
        (Module &m, spu::runtime::Task &tsk, size_t frame) -> int 
        {
            KNN knn = static_cast<KNN&>(m);
            
            // Get the input and output data pointers from the task
            const uint32_t* n_RoIs0_in = tsk[si_n_RoIs0].get_dataptr<const uint32_t>();
            const RoI_t* RoIs0_in = (RoI_t*)tsk[si_RoIs0].get_dataptr<const uint8_t>();
            const uint32_t* n_RoIs1_in = tsk[si_n_RoIs1].get_dataptr<const uint32_t>();
            const RoI_t* RoIs1_in = (RoI_t*)tsk[si_RoIs1].get_dataptr<const uint8_t>();
            
            uint32_t* n_assoc_out = tsk[so_n_assoc].get_dataptr<uint32_t>();
            RoI_t* RoIs_out0 = (RoI_t*)tsk[so_RoIs0].get_dataptr<uint8_t>();
            RoI_t* RoIs_out1 = (RoI_t*)tsk[so_RoIs1].get_dataptr<uint8_t>();
            float** distances_out = tsk[so_distances].get_2d_dataptr<float>();
            uint32_t** nearest_out = tsk[so_nearest].get_2d_dataptr<uint32_t>();


            *n_assoc_out = kNN_match(knn.knn_data, (RoI_t*)RoIs0_in, *n_RoIs0_in, (RoI_t*)RoIs1_in,
                                        *n_RoIs1_in, knn.knn_k, knn.knn_d, knn.knn_s); 

            // copy the result to the output
            memcpy(RoIs_out0, RoIs0_in, *n_RoIs0_in * sizeof(RoI_t));
            memcpy(RoIs_out1, RoIs1_in, *n_RoIs1_in * sizeof(RoI_t));

            memcpy(distances_out[0], knn.knn_data->distances[0], knn.knn_data->_max_size * knn.knn_data->_max_size * sizeof(float));
            memcpy(nearest_out[0], knn.knn_data->nearest[0], knn.knn_data->_max_size * knn.knn_data->_max_size * sizeof(uint32_t));

            return runtime::status_t::SUCCESS;
        }
    );
}

KNN* KNN::clone() const {
    auto k = new KNN(*this);  
    k->deep_copy(*this);     
    return k;
}

void KNN::deep_copy(const KNN& k) {
    Stateful::deep_copy(k); 
    this->knn_data = kNN_alloc_data(k.p_cca_roi_max2);
    kNN_init_data(this->knn_data);
}