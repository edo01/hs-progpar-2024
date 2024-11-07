#ifndef FEATURES_CCA_HPP
#define FEATURES_CCA_HPP

#include <streampu.hpp>
#include "motion/tools.h" 
#include "motion/features.h"        

class Features_CCA : public spu::module::Stateful {
    private:
        int i0, i1, j0, j1, p_cca_roi_max1;

    public:
        /**
         * Constructor to initialize the Features_CCA module
         * @param i0 Starting y-coordinate of the image
         * @param i1 Ending y-coordinate of the image
         * @param j0 Starting x-coordinate of the image
         * @param j1 Ending x-coordinate of the image
         * @param p_cca_roi_max1 Maximum number of regions of interest
         */
        //Features_CCA(int i0, int i1, int j0, int j1, int p_cca_roi_max1);

        CCA* CCA::clone() const {
            auto c = new CCA(*this);
            c->deep_copy(*this);
            return c;
        }

        void CCA::deep_copy(const CCA& c) {
            Stateful::deep_copy(c);
            this->cca_data = cca_alloc_data(c.cca_data->i0, c.cca_data->i1,
                                            c.cca_data->j0, c.cca_data->j1,
                                            f.cca_data->p_cca_roi_max1);
            cca_init_data(this->cca_data);
        }
};

#endif 
