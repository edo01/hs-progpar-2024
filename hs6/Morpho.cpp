#include "motion/wrapper/Morpho.hpp"

Morpho::Morpho(morpho_data_t* morpho_data, int i0, int i1, int j0, int j1) 
    : spu::module::Stateful(), morpho_data(morpho_data), i0(i0), i1(i1), j0(j0), j1(j1) 
{
    const std::string name = "Morpho";
    this->set_name(name);


    auto &t = this->create_task("Morpho_compute");

    size_t si_data_IB = create_2d_sck_in<uint8_t>(t, "in_IB", (i1 - i0), (j1 - j0)); 
    size_t so_data_IB = create_2d_sck_out<uint8_t>(t, "out_IB", (i1 - i0), (j1 - j0)); 

    create_codelet(t, 
        [this, si_data_IB, so_data_IB] (Module &m, spu::runtime::Task &tsk) -> int {
            // Get the input and output data pointers from the task
            uint8_t** IB_in = tsk[si_data_IB].get_2d_dataptr<uint8_t>();
            uint8_t** IB_out = tsk[so_data_IB].get_2d_dataptr<uint8_t>();

            // Apply the opening and closing morphological operations
            morpho_compute_opening3(morpho_data, (const uint8_t**)IB_in, IB_out, i0, i1, j0, j1);
            morpho_compute_closing3(morpho_data, (const uint8_t**)IB_out, IB_out, i0, i1, j0, j1);
            
            return 0;  
        }
    );
}
