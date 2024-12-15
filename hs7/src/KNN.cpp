#include "motion/wrapper/KNN.hpp"

using namespace spu;


KNN::KNN(kNN_data_t* knn_data, int p_cca_roi_max2, int knn_k, uint32_t knn_d, float knn_s, int p_log_path)
    : spu::module::Stateful(), knn_data(knn_data), p_cca_roi_max2(p_cca_roi_max2),
        knn_k(knn_k), knn_d(knn_d), knn_s(knn_s), p_log_path(p_log_path)    
{
    const std::string name = "KNN";
    this->set_name(name);
    this->set_short_name(name);

    auto &matchf    = this->create_task("matchf");
    
    // ---------------------- //
    // -- FORWARD VERSION -- //
    // --------------------- //

    size_t si_n_RoIs0_f   = this->template create_socket_in<uint32_t>(matchf, "in_n_RoIs0", 1);
    size_t si_n_RoIs1_f   = this->template create_socket_in<uint32_t>(matchf, "in_n_RoIs1", 1);
    
    // forward data
    size_t sf_RoIs0_f      = this->template create_socket_fwd<uint8_t>(matchf, "fwd_RoIs0", p_cca_roi_max2* sizeof(RoI_t));
    size_t sf_RoIs1_f      = this->template create_socket_fwd<uint8_t>(matchf, "fwd_RoIs1", p_cca_roi_max2* sizeof(RoI_t));
    
    size_t so_distances_f = 0;
    size_t so_nearest_f = 0;
#ifdef MOTION_ENABLE_DEBUG
    size_t so_conflicts_f = 0;
#endif

    // output data
    if(p_log_path){
        so_distances_f = this->template create_2d_socket_out<float>(matchf, "out_distances", knn_data->_max_size, knn_data->_max_size);
        so_nearest_f = this->template create_2d_socket_out<uint32_t>(matchf, "out_nearest", knn_data->_max_size, knn_data->_max_size);
#ifdef MOTION_ENABLE_DEBUG
        so_conflicts_f = this->template create_socket_out<uint32_t>(matchf, "out_conflicts", knn_data->_max_size);
#endif
    }
    // return value
    uint32_t so_n_assoc_f = this->template create_socket_out<uint32_t>(matchf, "out_n_assoc", 1);

    create_codelet(matchf, 
#ifdef MOTION_ENABLE_DEBUG
        [si_n_RoIs0_f, si_n_RoIs1_f, so_n_assoc_f, sf_RoIs0_f, sf_RoIs1_f, so_distances_f, so_nearest_f, so_conflicts_f]
#else
        [si_n_RoIs0_f, si_n_RoIs1_f, so_n_assoc_f, sf_RoIs0_f, sf_RoIs1_f, so_distances_f, so_nearest_f]
#endif
        (Module &m, spu::runtime::Task &tsk, size_t frame) -> int 
        {
            KNN knn = static_cast<KNN&>(m);
            
            // Get the input and output data pointers from the task
            const uint32_t* n_RoIs0_in = tsk[si_n_RoIs0_f].get_dataptr<const uint32_t>();
            const uint32_t* n_RoIs1_in = tsk[si_n_RoIs1_f].get_dataptr<const uint32_t>();

            float** distances_out = nullptr;
            uint32_t** nearest_out = nullptr;
#ifdef MOTION_ENABLE_DEBUG
            uint32_t* conflicts_out = nullptr;
#endif

            if(knn.p_log_path){
                distances_out = tsk[so_distances_f].get_2d_dataptr<float>();
                nearest_out = tsk[so_nearest_f].get_2d_dataptr<uint32_t>();
#ifdef MOTION_ENABLE_DEBUG
                conflicts_out = tsk[so_conflicts].get_dataptr<uint32_t>();
#endif 
            }

            
            uint32_t* n_assoc_out = tsk[so_n_assoc_f].get_dataptr<uint32_t>();

            RoI_t* RoIs0_fwd = (RoI_t*)tsk[sf_RoIs0_f].get_dataptr<uint8_t>();
            RoI_t* RoIs1_fwd = (RoI_t*)tsk[sf_RoIs1_f].get_dataptr<uint8_t>();

            *n_assoc_out = kNN_match(knn.knn_data, RoIs0_fwd, *n_RoIs0_in, RoIs1_fwd,
                                        *n_RoIs1_in, knn.knn_k, knn.knn_d, knn.knn_s); 

            if(knn.p_log_path){
                memcpy(distances_out[0], knn.knn_data->distances[0], knn.knn_data->_max_size * knn.knn_data->_max_size * sizeof(float));
                memcpy(nearest_out[0], knn.knn_data->nearest[0], knn.knn_data->_max_size * knn.knn_data->_max_size * sizeof(uint32_t));
#ifdef MOTION_ENABLE_DEBUG
                memcpy(conflicts_out, knn.knn_data->conflicts, knn.knn_data->_max_size * sizeof(uint32_t));
#endif
            }

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