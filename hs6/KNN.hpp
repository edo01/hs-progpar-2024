#ifndef KNN_HPP
#define KNN_HPP

#include "spu/module/stateful.hpp"   
#include "motion/kNN.h"              

class KNN : public spu::module::Stateful {
public:
    /**
     * Constructor to initialize the KNN module
     * @param knn_k Number of neighbors to consider
     * @param knn_d Maximum distance for matching
     * @param knn_s Minimum surface ratio for matching
     */
    KNN(int knn_k, int knn_d, float knn_s);

private:
    int knn_k, knn_d;
    float knn_s;
};

#endif 
