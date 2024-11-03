#include "motion/wrapper/KNN.hpp"

using namespace spu;

KNN::KNN(kNN_data_t* knn_data, int p_cca_roi_max2, int knn_k, uint32_t knn_d, float knn_s)
    : spu::module::Stateful(), p_cca_roi_max2(p_cca_roi_max2), knn_data(knn_data), 
      knn_k(knn_k), knn_d(knn_d), knn_s(knn_s)
{
    const std::string name = "KNN";
    this->set_name(name);
    this->set_short_name(name);

    auto &t = this->create_task("matchf"); 
    
    size_t si_n_RoIs0 = this->template create_socket_in<uint32_t>(t, "in_n_RoIs0", 1);
    size_t si_RoIs0 = this->template create_socket_in<uint8_t>(t, "in_RoIs0", p_cca_roi_max2 * sizeof(RoI_t));
    size_t si_n_RoIs1 = this->template create_socket_in<uint32_t>(t, "in_n_RoIs1", 1);

    // forward socket
    size_t sf_RoIs1 = this->template create_socket_fwd<RoI_t>(t, "RoIs1", p_cca_roi_max2);
    
    uint32_t so_n_assoc = this->template create_socket_out<uint32_t>(t, "out_n_assoc", 1);

    create_codelet(t, 
        [si_n_RoIs0, si_RoIs0, si_n_RoIs1, sf_RoIs1, so_n_assoc]
        (Module &m, spu::runtime::Task &tsk, size_t frame) -> int 
        {
            KNN& knn = static_cast<KNN&>(m);
            
            const uint32_t* n_RoIs0_in = tsk[si_n_RoIs0].get_dataptr<const uint32_t>();
            const RoI_t* RoIs0_in = tsk[si_RoIs0].get_dataptr<const RoI_t>();
            const uint32_t* n_RoIs1_in = tsk[si_n_RoIs1].get_dataptr<const uint32_t>();

            RoI_t* RoIs1_data = tsk[sf_RoIs1].get_dataptr<RoI_t>();
            uint32_t* n_assoc_out = tsk[so_n_assoc].get_dataptr<uint32_t>();

            *n_assoc_out = kNN_match(knn.knn_data, RoIs0_in, *n_RoIs0_in, RoIs1_data,
                                     *n_RoIs1_in, knn.knn_k, knn.knn_d, knn.knn_s); 

            return runtime::status_t::SUCCESS;
        }
    );
}
