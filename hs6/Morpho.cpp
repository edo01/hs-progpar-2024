#include "motion/wrapper/Morpho.hpp"

using namespace spu;

Morpho::Morpho(morpho_data_t* morpho_data, int i0, int i1, int j0, int j1) 
    : spu::module::Stateful(), morpho_data(morpho_data), i0(i0), i1(i1), j0(j0), j1(j1) 
{
    const std::string name = "Morpho";
    this->set_name(name);
    this->set_short_name(name);

    auto &t = this->create_task("compute");

    // input socket
    size_t si_img = this->template create_2d_sck_in<uint8_t>(t, "in_img", (i1 - i0 + 1), (j1 - j0 + 1)); 
    // output socket
    size_t so_img = this->template create_2d_sck_out<uint8_t>(t, "out_img", (i1 - i0 + 1), (j1 - j0 + 1)); 

    create_codelet(t, 
        [si_img, so_img] 
        (Module &m, spu::runtime::Task &tsk, size_t frame) -> int 
        {
            Morpho morpho = static_cast<Morpho&>(m);        

            // Get the input and output data pointers from the task
            const uint8_t** img_in = tsk[si_img].get_2d_dataptr<const uint8_t>();
            uint8_t** img_out = tsk[so_img].get_2d_dataptr<uint8_t>();

            // Apply the opening and closing morphological operations
            morpho_compute_opening3(morpho.morpho_data, img_in, img_out, morpho.i0, morpho.i1, morpho.j0, morpho.j1);
            morpho_compute_closing3(morpho.morpho_data, (const uint8_t**) img_out, img_out, morpho.i0, morpho.i1, morpho.j0, morpho.j1);
            
            return runtime::status_t::SUCCESS;
        }
    );
}
