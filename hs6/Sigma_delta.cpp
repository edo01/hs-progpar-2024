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

    auto &t = this->create_task("Sigma_delta_compute");

    size_t si_data_IG = create_2d_sck_in<uint8_t>(t, "in_IG", (i1-i0), (j1-j0));

    size_t so_data_IB = create_2d_sck_out<uint8_t>(t, "out_IB", (i1-i0), (j1-j0));

    create_codelet(t, 
        [si_data_IG, so_data_IB] (Module &m, runtime::Task &tsk, size_t frame) -> int {
            Sigma_delta s = static_cast<Sigma_delta&>(m);
            
            const uint8_t** IG = tsk[si_data_IG].get_2d_dataptr<const uint8_t>();
            uint8_t** IB = tsk[so_data_IB].get_2d_dataptr<uint8_t>();

            sigma_delta_compute(s.sd_data, IG, IB, s.i0, s.i1, s.j0, s.j1, s.p_sd_n);
            return runtime::status_t::SUCCESS;
        }
    );
}