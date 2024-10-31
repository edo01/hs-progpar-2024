#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include "Sigma_delta.hpp"
#include "Morpho.hpp"
#include "CCL.hpp"
#include "Features_CCA.hpp"
#include "Features_filter.hpp"
#include "KNN.hpp"
#include "Tracking.hpp"
#include "Delayer.hpp"
#include "Finalizer.hpp"
#include "tools/Stats.hpp"  

/*
 * Implementation steps
 * 1) Check the command line parameters at the beginning of the program to determine whether to enable statistics display.
 * 2) Call the get_modules method to get the list of modules from the Sequence object and enable statistics for each module.
 * 3) Execute the sequence
 * 4) If the --stats parameter is enabled, call tools::Stats::show at the end to print the statistics results.
 */


int main(int argc, char* argv[]) {
    // Check if the stats parameter is passed
    bool enable_stats = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--stats") {
            enable_stats = true;
            break;
        }
    }

    // 1) create the module objects
    Source_user_binary<uint8_t> src(4, "vid.mp4", false); 

    /* Initialize the Delayer module through uint8_t init_val to 
     * ensure that the delayed data of the first frame is valid.
    */
    uint8_t init_val = 0; // buff val for stream #0

    Delayer<uint8_t> delay(4, init_val);  //Delay module, used to save t-1 frame
    Sigma_delta sigma_delta_module(...);                   
    Morpho morpho_module(...);                             
    CCL ccl_module(...);                                   
    Features_CCA features_cca_module(...);                 
    Features_filter features_filter_module(...);           
    KNN knn_module(...);                                   
    Tracking tracking_module(...);                         
    Finalizer<uint8_t> fin(4);  


    // 2) bind the tasks
    fin["finalize::in"] = delay["produce::out"];
    delay["memorize::in"] = src["generate::out_data"];
    delay["produce::out"] = sigma_delta_module["input::image"];  // Bind Delayer output as Sigma_delta input
    sigma_delta_module["output::motion"] = morpho_module["input::binary"];  // Bind Sigma_delta output to Morpho input
    morpho_module["output::processed"] = ccl_module["input::binary"];     // Bind Morpho output to CCL input
    ccl_module["output::labels"] = features_cca_module["input::labels"];   // Bind CCL output to Features_CCA input
    features_cca_module["output::rois"] = features_filter_module["input::rois"]; // Bind Features_CCA output to Features_filter input
    features_filter_module["output::filtered"] = knn_module["input::current_rois"]; // Bind Features_filter output to KNN input
    knn_module["output::matches"] = tracking_module["input::rois"];        // Bind KNN output to Tracking input
    tracking_module["output::tracks"] = fin["finalize::in"];               //Finally output to Finalizer

    // 3) create the sequence
    std::vector<runtime::Task*> firsts = { &delay("produce"), &src("generate") };
    runtime::Sequence seq(firsts);

    // 4) enable the statistics collection for each task of the sequence
    if (enable_stats) {
        for (auto& mdl : seq.get_modules<module::Module>(false)) {
            for (auto& tsk : mdl->tasks) {
                tsk->set_stats(true); // enable the statistics
            }
        }
    }

    // 5) Execute sequence (run 100000 times)
    unsigned int exe_counter = 0;
    seq.exec([&exe_counter]() { return ++exe_counter >= 100000; });

    // 6)Display task statistics (if --stats is enabled)
    if (enable_stats) {
        const bool ordered = true;
        const bool display_throughput = false;
        tools::Stats::show(seq.get_modules_per_types(), ordered, display_throughput);
    }

    return 0;
}
