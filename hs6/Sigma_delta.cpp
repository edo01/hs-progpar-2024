#include "motion/wrapper/Sigma_delta.hpp"

using namespace spu;
/**
 *  INPUTS:
    int i0, i1, j0, j1; // image dimension (i0 = y_min, i1 = y_max, j0 = x_min, j1 = x_max)
    uint8_t **IG = ui8matrix(i0, i1, j0, j1); // grayscale input image at t
    uint8_t **IB = ui8matrix(i0, i1, j0, j1); // binary image (after Sigma-Delta) at t
    int p_sd_n; Value of the N parameter in the Sigma-Delta algorithm
 */
// sigma_delta_compute(sd_data1, (const uint8_t**)IG1, IB1, i0, i1, j0, j1, p_sd_n);
Sigma_delta::Sigma_delta(sigma_delta_data_t* sd_data, int i0, int i1, int j0, int j1, int p_sd_n): 
    module::Stateful(), sd_data(sd_data), i0(i0), i1(i1), j0(j0), j1(j1), p_sd_n(p_sd_n)
{
    const std::string name = "Sigma_delta";
    this->set_name(name);
    this->set_short_name(name);

    auto &t = this->create_task("compute");

    // Input socket
    size_t si_img = this->template create_2d_sck_in<uint8_t>(t, "in_img", (i1-i0)+1, (j1-j0)+1);
    // Output socket
    size_t so_img = this->template create_2d_sck_out<uint8_t>(t, "out_img", (i1-i0)+1, (j1-j0)+1);

    create_codelet(t, 
        [si_img, so_img] 
        (Module &m, runtime::Task &tsk, size_t frame) -> int 
        {
            Sigma_delta sigma = static_cast<Sigma_delta&>(m);
            
            const uint8_t** in_img = tsk[si_img].get_2d_dataptr<const uint8_t>();
            uint8_t** out_img = tsk[so_img].get_2d_dataptr<uint8_t>();
            
            sigma_delta_compute(sigma.sd_data, in_img, out_img, sigma.i0, sigma.i1, sigma.j0, sigma.j1, sigma.p_sd_n);
            return runtime::status_t::SUCCESS;
        }
    );


        
}