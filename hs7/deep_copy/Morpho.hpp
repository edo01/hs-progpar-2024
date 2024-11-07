#ifndef MORPHO_HPP
#define MORPHO_HPP

#include <streampu.hpp>
#include "motion/tools.h"
#include "motion/morpho.h"           

/**
 * Constructor to initialize the Morpho module
 * @param morpho_data Pointer to the morpho_data_t structure used for morphological operations
 * @param i0 Starting y-coordinate of the image
 * @param i1 Ending y-coordinate of the image
 * @param j0 Starting x-coordinate of the image
 * @param j1 Ending x-coordinate of the image
 */
class Morpho : public spu::module::Stateful {
    public:
        Morpho(morpho_data_t* morpho_data, int i0, int i1, int j0, int j1);
    private:
        morpho_data_t* morpho_data;  
        int i0, i1, j0, j1;  
        
    Morpho* Morpho::clone() const {
        auto m = new Morpho(*this);
        m->deep_copy(*this); // we override this method just after
        return m;
    }
    // in the deep_copy method, 'this' is the newly allocated object while 'm' is the former object
    void Morpho::deep_copy(const Morpho& m) {
        // call the 'deep_copy' method of the Module class
        Stateful::deep_copy(m);
        // allocate new morpho inner data
        this->morpho_data = morpho_alloc_data(m.morpho_data->i0, m.morpho_data->i1,
                                            m.morpho_data->j0, m.morpho_data->j1);
        // initialize the previously allocated data
        morpho_init_data(this->morpho_data);
    }
};

#endif 