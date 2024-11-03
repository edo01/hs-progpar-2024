#include "motion/wrapper/Morpho.hpp"

using namespace spu;

Morpho::Morpho(morpho_data_t* morpho_data, int i0, int i1, int j0, int j1) 
    : spu::module::Stateful(), morpho_data(morpho_data), i0(i0), i1(i1), j0(j0), j1(j1) 
{
    const std::string name = "Morpho";
    this->set_name(name);
    this->set_short_name(name);

    auto &t = this->create_task("computef"); 

    // forward socket for img
    size_t sf_img = this->template create_2d_sck_fwd<uint8_t>(t, "img", (i1 - i0 + 1), (j1 - j0 + 1));

    create_codelet(t, 
        [sf_img] 
        (Module &m, spu::runtime::Task &tsk, size_t frame) -> int 
        {
            Morpho& morpho = static_cast<Morpho&>(m);        

            //Merge the original img_in and img_out into one img_data and operate directly in the task.
            uint8_t** img_data = tsk[sf_img].get_2d_dataptr<uint8_t>();

            morpho_compute_opening3f(morpho.morpho_data, (const uint8_t**)img_data, img_data, morpho.i0, morpho.i1, morpho.j0, morpho.j1);
            morpho_compute_closing3f(morpho.morpho_data, (const uint8_t**)img_data, img_data, morpho.i0, morpho.i1, morpho.j0, morpho.j1);
            
            return runtime::status_t::SUCCESS;
        }
    );
}
