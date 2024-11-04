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

    auto &t = this->create_task("match");
    
    // input data
    size_t si_n_RoIs0 = this->template create_socket_in<uint32_t>(t, "in_n_RoIs0", 1);
    size_t si_RoIs0 = this->template create_socket_in<uint8_t>(t, "in_RoIs0", p_cca_roi_max2* sizeof(RoI_t));
    size_t si_n_RoIs1 = this->template create_socket_in<uint32_t>(t, "in_n_RoIs1", 1);
    size_t si_RoIs1 = this->template create_socket_in<uint8_t>(t, "in_RoIs1", p_cca_roi_max2* sizeof(RoI_t));
    
    //output data
    size_t so_n_RoIs = this->template create_socket_out<uint32_t>(t, "out_n_RoIs", 1);
    size_t so_RoIs = this->template create_socket_out<uint8_t>(t, "out_RoIs", p_cca_roi_max2* sizeof(RoI_t));

    // return value
    uint32_t so_n_assoc = this->template create_socket_out<uint32_t>(t, "out_n_assoc", 1);

    create_codelet(t, 
        [si_n_RoIs0, si_RoIs0, si_n_RoIs1, si_RoIs1, so_n_assoc, so_n_RoIs, so_RoIs]
        (Module &m, spu::runtime::Task &tsk, size_t frame) -> int 
        {
            KNN knn = static_cast<KNN&>(m);
            
            // Get the input and output data pointers from the task
            const uint32_t* n_RoIs0_in = tsk[si_n_RoIs0].get_dataptr<const uint32_t>();
            const RoI_t* RoIs0_in = (RoI_t*)tsk[si_RoIs0].get_dataptr<const uint8_t>();
            const uint32_t* n_RoIs1_in = tsk[si_n_RoIs1].get_dataptr<const uint32_t>();
            const RoI_t* RoIs1_in = (RoI_t*)tsk[si_RoIs1].get_dataptr<const uint8_t>();
            
            uint32_t* n_assoc_out = tsk[so_n_assoc].get_dataptr<uint32_t>();
            uint32_t* n_RoIs_out = tsk[so_n_RoIs].get_dataptr<uint32_t>();
            RoI_t* RoIs_out = (RoI_t*)tsk[so_RoIs].get_dataptr<uint8_t>();

            *n_assoc_out = kNN_match(knn.knn_data, (RoI_t*)RoIs0_in, *n_RoIs0_in, (RoI_t*)RoIs1_in,
                                        *n_RoIs1_in, knn.knn_k, knn.knn_d, knn.knn_s); 

            // copy the result to the output
            *n_RoIs_out = *n_RoIs1_in;
            memcpy(RoIs_out, RoIs1_in, *n_RoIs1_in * sizeof(RoI_t));

            return runtime::status_t::SUCCESS;
        }
    );
}
