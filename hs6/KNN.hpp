#ifndef KNN_HPP
#define KNN_HPP

/*
AUTHORS: MENGQIAN XU (21306077), EDOARDO CARRA' (21400562)
BOARD ID: Q
*/

#include <streampu.hpp>
#include "motion/tools.h" 
#include "motion/kNN.h"   

class KNN : public spu::module::Stateful {
public:
    /**
     * Constructor to initialize the KNN module
     * @param knn_k Number of neighbors to consider
     * @param knn_d Maximum distance for matching
     * @param knn_s Minimum surface ratio for matching
     */
    KNN(kNN_data_t* knn_data, int p_cca_roi_max2,
            int knn_k, uint32_t knn_d, float knn_s);

private:
    kNN_data_t* knn_data;
    int p_cca_roi_max2;
    int knn_k;
    uint32_t knn_d;
    float knn_s;
};

#endif 
