#include "Sigma_delta.hpp"
#include "Morpho.hpp"
#include "CCL.hpp"
#include "Features_CCA.hpp"
#include "Features_filter.hpp"
#include "KNN.hpp"
#include "Tracking.hpp"
#include "Delayer.hpp"

/*
 * To achieve sequential data flow, we need to bind the output socket of each module to the 
 * input socket of the next module so that data automatically flows from one task to the next. 
 * This eliminates the need to manually call .exec().
 * 
 * Delayer works by storing the output of the previous frame and providing this data 
 * when needed in the next frame.
 * 
*/


int main() {
    #include "Sigma_delta.hpp"
#include "Morpho.hpp"
#include "CCL.hpp"
#include "Features_CCA.hpp"
#include "Features_filter.hpp"
#include "KNN.hpp"
#include "Tracking.hpp"
#include "Delayer.hpp"
#include "Source_user_binary.hpp"
#include "Finalizer.hpp"

int main() {
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

    // 2) Set a custom name for debugging
    src.set_custom_name("Video Source");
    delay.set_custom_name("Frame Delayer");
    sigma_delta_module.set_custom_name("Motion Detection");
    morpho_module.set_custom_name("Morphology Processing");
    ccl_module.set_custom_name("Connected Component Labeling");
    features_cca_module.set_custom_name("Feature Extraction");
    features_filter_module.set_custom_name("Feature Filtering");
    knn_module.set_custom_name("k-NN Matching");
    tracking_module.set_custom_name("Tracking");
    fin.set_custom_name("Finalizer");

    // 3) bind the tasks
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

    // 4) Export task graph for debugging
    export_dot("task_graph.dot");  

    // 5) Enable logging and visualization
    enable_logs(true);          
    enable_visualization(true); 

    // 6)  create the sequence
    std::vector<runtime::Task*> firsts = { &delay("produce"), &src("generate") };
    runtime::Sequence seq(firsts);

    // 7)  execute the sequence (no stop)
    seq.exec([]() {
        std::cout << "Executing task in sequence..." << std::endl;
        return false; 
    });


    return 0;
}
