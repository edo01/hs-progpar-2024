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
    virtual Morpho* clone() const override;
    void deep_copy(const Morpho& m);
    
private:
    morpho_data_t* morpho_data;  
    int i0, i1, j0, j1;  
};

#endif 