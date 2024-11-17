/**
##############################################################
##############################################################
##############################################################

AUTHORS: MENGQIAN XU (21306077), EDOARDO CARRA' (21400562)
BOARD ID: Q

##############################################################
##############################################################
##############################################################
*/

#include <stdio.h>
#include <assert.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <nrc2.h>
#include <math.h>
#include <streampu.hpp>

#include "vec.h"

#include "motion/args.h"
#include "motion/tools.h"
#include "motion/macros.h"

#include "motion/CCL.h"
#include "motion/features.h"
#include "motion/kNN.h"
#include "motion/tracking.h"
#include "motion/video.h"
#include "motion/image.h"
#include "motion/video.h"
#include "motion/sigma_delta.h"
#include "motion/morpho.h"
#include "motion/visu.h"

#include "motion/wrapper/Video_reader.hpp"
#include "motion/wrapper/Logger_frame.hpp"
#include "motion/wrapper/Logger_RoIs.hpp"
#include "motion/wrapper/Logger_kNN.hpp"
#include "motion/wrapper/Logger_tracks.hpp"
#include "motion/wrapper/Visu.hpp"

// our defined modules
#include "motion/wrapper/Sigma_delta.hpp"
#include "motion/wrapper/Morpho.hpp"
#include "motion/wrapper/CCL.hpp"
#include "motion/wrapper/Features_CCA.hpp"
#include "motion/wrapper/Features_filter.hpp"
#include "motion/wrapper/KNN.hpp"
#include "motion/wrapper/Tracking.hpp"


int main(int argc, char** argv) {

    // ---------------------------------- //
    // -- DEFAULT VALUES OF PARAMETERS -- //
    // ---------------------------------- //

    char* def_p_vid_in_path = NULL;
    int def_p_vid_in_start = 0;
    int def_p_vid_in_stop = 0;
    int def_p_vid_in_skip = 0;
    int def_p_vid_in_loop = 1;
    int def_p_vid_in_threads = 0;
    char def_p_vid_in_dec_hw[16] = "NONE";
    int def_p_sd_n = 2;
    char* def_p_ccl_fra_path = NULL;
    int def_p_flt_s_min = 50;
    int def_p_flt_s_max = 100000;
    int def_p_knn_k = 3;
    int def_p_knn_d = 10;
    float def_p_knn_s = 0.125f;
    int def_p_trk_ext_d = 5;
    int def_p_trk_ext_o = 3;
    int def_p_trk_obj_min = 2;
    char* def_p_trk_roi_path = NULL;
    char* def_p_log_path = NULL;
    int def_p_cca_roi_max1 = 65536; // Maximum number of RoIs
    int def_p_cca_roi_max2 = 8192; // Maximum number of RoIs after filtering
    char* def_p_vid_out_path = NULL;

    // ------------------------ //
    // -- CMD LINE ARGS HELP -- //
    // ------------------------ //

    if (args_find(argc, argv, "--help,-h")) {
        fprintf(stderr,
                "  --vid-in-path     Path to video file or to an images sequence                            [%s]\n",
                def_p_vid_in_path ? def_p_vid_in_path : "NULL");
        fprintf(stderr,
                "  --vid-in-start    Start frame id (included) in the video                                 [%d]\n",
                def_p_vid_in_start);
        fprintf(stderr,
                "  --vid-in-stop     Stop frame id (included) in the video (if set to 0, read entire video) [%d]\n",
                def_p_vid_in_stop);
        fprintf(stderr,
                "  --vid-in-skip     Number of frames to skip                                               [%d]\n",
                def_p_vid_in_skip);
        fprintf(stderr,
                "  --vid-in-buff     Bufferize all the video in global memory before executing the chain        \n");
        fprintf(stderr,
                "  --vid-in-loop     Number of times the video is read in loop                              [%d]\n",
                def_p_vid_in_loop);
        fprintf(stderr,
                "  --vid-in-threads  Select the number of threads to use to decode video input (in ffmpeg)  [%d]\n",
                def_p_vid_in_threads);
        fprintf(stderr,
                "  --vid-in-dec-hw   Select video decoder hardware acceleration ('NONE', 'NVDEC', 'VIDTB')  [%s]\n",
                def_p_vid_in_dec_hw);
        fprintf(stderr,
                "  --sd-n            Value of the N parameter in the Sigma-Delta algorithm                  [%d]\n",
                def_p_sd_n);
        fprintf(stderr,
                "  --ccl-fra-path    Path of the files for CC debug frames                                  [%s]\n",
                def_p_ccl_fra_path ? def_p_ccl_fra_path : "NULL");
#ifdef MOTION_OPENCV_LINK
        fprintf(stderr,
                "  --ccl-fra-id      Show the RoI/CC ids on the ouptut CC frames                                \n");
#endif
        fprintf(stderr,
                "  --cca-roi-max1    Maximum number of RoIs after CCA                                       [%d]\n",
                def_p_cca_roi_max1);
        fprintf(stderr,
                "  --cca-roi-max2    Maximum number of RoIs after surface filtering                         [%d]\n",
                def_p_cca_roi_max2);
        fprintf(stderr,
                "  --flt-s-min       Minimum surface of the CCs in pixels                                   [%d]\n",
                def_p_flt_s_min);
        fprintf(stderr,
                "  --flt-s-max       Maxumum surface of the CCs in pixels                                   [%d]\n",
                def_p_flt_s_max);
        fprintf(stderr,
                "  --knn-k           Maximum number of neighbors considered in k-NN algorithm               [%d]\n",
                def_p_knn_k);
        fprintf(stderr,
                "  --knn-d           Maximum distance in pixels between two images (in k-NN)                [%d]\n",
                def_p_knn_d);
        fprintf(stderr,
                "  --knn-s           Minimum surface ratio to match two CCs in k-NN                         [%f]\n",
                def_p_knn_s);
        fprintf(stderr,
                "  --trk-ext-d       Search radius in pixels for CC extrapolation (piece-wise tracking)     [%d]\n",
                def_p_trk_ext_d);
        fprintf(stderr,
                "  --trk-ext-o       Maximum number of frames to extrapolate (linear) for lost objects      [%d]\n",
                def_p_trk_ext_o);
        fprintf(stderr,
                "  --trk-obj-min     Minimum number of frames required to track an object                   [%d]\n",
                def_p_trk_obj_min);
        fprintf(stderr,
                "  --trk-roi-path    Path to the file containing the RoI ids for each track                 [%s]\n",
                def_p_trk_roi_path ? def_p_trk_roi_path : "NULL");
        fprintf(stderr,
                "  --log-path        Path of the output statistics, only required for debugging purpose     [%s]\n",
                def_p_log_path ? def_p_log_path : "NULL");
        fprintf(stderr,
                "  --vid-out-path    Path to video file or to an images sequence to write the output        [%s]\n",
                def_p_vid_out_path ? def_p_vid_out_path : "NULL");
        fprintf(stderr,
                "  --vid-out-play    Show the output video in a SDL window                                      \n");
#ifdef MOTION_OPENCV_LINK
        fprintf(stderr,
                "  --vid-out-id      Draw the track ids on the ouptut video                                     \n");
#endif
        fprintf(stderr,
                "  --stats           Show the average latency of each task                                      \n");
		fprintf(stderr,
                "  --help, -h        This help                                                                  \n");
        exit(1);
    }

    // ------------------------- //
    // -- PARSE CMD LINE ARGS -- //
    // ------------------------- //

    const char* p_vid_in_path = args_find_char(argc, argv, "--vid-in-path", def_p_vid_in_path);
    const int p_vid_in_start = args_find_int_min(argc, argv, "--vid-in-start", def_p_vid_in_start, 0);
    const int p_vid_in_stop = args_find_int_min(argc, argv, "--vid-in-stop", def_p_vid_in_stop, 0);
    const int p_vid_in_skip = args_find_int_min(argc, argv, "--vid-in-skip", def_p_vid_in_skip, 0);
    const int p_vid_in_buff = args_find(argc, argv, "--vid-in-buff");
    const int p_vid_in_loop = args_find_int_min(argc, argv, "--vid-in-loop", def_p_vid_in_loop, 1);
    const int p_vid_in_threads = args_find_int_min(argc, argv, "--vid-in-threads", def_p_vid_in_threads, 0);
    const char* p_vid_in_dec_hw = args_find_char(argc, argv, "--vid-in-dec-hw", def_p_vid_in_dec_hw);
    const int p_sd_n = args_find_int_min(argc, argv, "--sd-n", def_p_sd_n, 0);
    const char* p_ccl_fra_path = args_find_char(argc, argv, "--ccl-fra-path", def_p_ccl_fra_path);
#ifdef MOTION_OPENCV_LINK
    const int p_ccl_fra_id = args_find(argc, argv, "--ccl-fra-id,--show-id");
#else
    const int p_ccl_fra_id = 0;
#endif
    const int p_cca_roi_max1 = args_find_int_min(argc, argv, "--cca-roi-max1", def_p_cca_roi_max1, 0);
    const int p_cca_roi_max2 = args_find_int_min(argc, argv, "--cca-roi-max2", def_p_cca_roi_max2, 0);
    const int p_flt_s_min = args_find_int_min(argc, argv, "--flt-s-min", def_p_flt_s_min, 0);
    const int p_flt_s_max = args_find_int_min(argc, argv, "--flt-s-max", def_p_flt_s_max, 0);
    const int p_knn_k = args_find_int_min(argc, argv, "--knn-k", def_p_knn_k, 0);
    const int p_knn_d = args_find_int_min(argc, argv, "--knn-d", def_p_knn_d, 0);
    const float p_knn_s = args_find_float_min_max(argc, argv, "--knn-s", def_p_knn_s, 0.f, 1.f);
    const int p_trk_ext_d = args_find_int_min(argc, argv, "--trk-ext-d", def_p_trk_ext_d, 0);
    const int p_trk_ext_o = args_find_int_min_max(argc, argv, "--trk-ext-o", def_p_trk_ext_o, 0, 255);
    const int p_trk_obj_min = args_find_int_min(argc, argv, "--trk-obj-min", def_p_trk_obj_min, 2);
    const char* p_trk_roi_path = args_find_char(argc, argv, "--trk-roi-path", def_p_trk_roi_path);
    const char* p_log_path = args_find_char(argc, argv, "--log-path", def_p_log_path);
    const char* p_vid_out_path = args_find_char(argc, argv, "--vid-out-path", def_p_vid_out_path);
    const int p_vid_out_play = args_find(argc, argv, "--vid-out-play");
#ifdef MOTION_OPENCV_LINK
    const int p_vid_out_id = args_find(argc, argv, "--vid-out-id");
#else
    const int p_vid_out_id = 0;
#endif
    const int p_stats = args_find(argc, argv, "--stats");

    // --------------------- //
    // -- HEADING DISPLAY -- //
    // --------------------- //

    printf("#  --------------- \n");
    printf("# |  MOTION2 SPU  |\n");
    printf("#  --------------- \n");
    printf("#\n");
    printf("# Parameters:\n");
    printf("# -----------\n");
    printf("#  * vid-in-path    = %s\n", p_vid_in_path);
    printf("#  * vid-in-start   = %d\n", p_vid_in_start);
    printf("#  * vid-in-stop    = %d\n", p_vid_in_stop);
    printf("#  * vid-in-skip    = %d\n", p_vid_in_skip);
    printf("#  * vid-in-buff    = %d\n", p_vid_in_buff);
    printf("#  * vid-in-loop    = %d\n", p_vid_in_loop);
    printf("#  * vid-in-threads = %d\n", p_vid_in_threads);
    printf("#  * vid-in-dec-hw  = %s\n", p_vid_in_dec_hw);
    printf("#  * sd-n           = %d\n", p_sd_n);
    printf("#  * ccl-fra-path   = %s\n", p_ccl_fra_path);
#ifdef MOTION_OPENCV_LINK
    printf("#  * ccl-fra-id     = %d\n", p_ccl_fra_id);
#endif
    printf("#  * cca-roi-max1   = %d\n", p_cca_roi_max1);
    printf("#  * cca-roi-max2   = %d\n", p_cca_roi_max2);
    printf("#  * flt-s-min      = %d\n", p_flt_s_min);
    printf("#  * flt-s-max      = %d\n", p_flt_s_max);
    printf("#  * knn-k          = %d\n", p_knn_k);
    printf("#  * knn-d          = %d\n", p_knn_d);
    printf("#  * knn-s          = %1.3f\n", p_knn_s);
    printf("#  * trk-ext-d      = %d\n", p_trk_ext_d);
    printf("#  * trk-ext-o      = %d\n", p_trk_ext_o);
    printf("#  * trk-obj-min    = %d\n", p_trk_obj_min);
    printf("#  * trk-roi-path   = %s\n", p_trk_roi_path);
    printf("#  * log-path       = %s\n", p_log_path);
    printf("#  * vid-out-path   = %s\n", p_vid_out_path);
    printf("#  * vid-out-play   = %d\n", p_vid_out_play);
#ifdef MOTION_OPENCV_LINK
    printf("#  * vid-out-id     = %d\n", p_vid_out_id);
#endif
    printf("#  * stats          = %d\n", p_stats);
    printf("#\n");

    // -------------------------- //
    // -- CMD LINE ARGS CHECKS -- //
    // -------------------------- //

    if (!p_vid_in_path) {
        fprintf(stderr, "(EE) '--vid-in-path' is missing\n");
        exit(1);
    }
    if (p_vid_in_stop && p_vid_in_stop < p_vid_in_start) {
        fprintf(stderr, "(EE) '--vid-in-stop' has to be higher than '--vid-in-start'\n");
        exit(1);
    }
#ifdef MOTION_OPENCV_LINK
    if (p_ccl_fra_id && !p_ccl_fra_path)
        fprintf(stderr, "(WW) '--ccl-fra-id' has to be combined with the '--ccl-fra-path' parameter\n");
#endif
    if (p_vid_out_path && p_vid_out_play)
        fprintf(stderr, "(WW) '--vid-out-path' will be ignore because '--vid-out-play' is set\n");
#ifdef MOTION_OPENCV_LINK
    if (p_vid_out_id && !p_vid_out_path && !p_vid_out_play)
        fprintf(stderr,
                "(WW) '--vid-out-id' will be ignore because neither '--vid-out-play' nor 'p_vid_out_path' are set\n");
#endif

    // --------------------------------------- //
    // -- VIDEO ALLOCATION & INITIALISATION -- //
    // --------------------------------------- //

    TIME_POINT(start_alloc_init);
    Video_reader video(p_vid_in_path, p_vid_in_start, p_vid_in_stop, p_vid_in_skip,
                       p_vid_in_buff, p_vid_in_threads, VCDC_FFMPEG_IO, video_hwaccel_str_to_enum(p_vid_in_dec_hw));
    int i0 = video.get_i0(), i1 = video.get_i1(), j0 = video.get_j0(), j1 = video.get_j1();
    video.set_loop_size(p_vid_in_loop);

    std::unique_ptr<Logger_frame> log_fra;
    if (p_ccl_fra_path)
        log_fra.reset(new Logger_frame(p_ccl_fra_path, p_vid_in_start, p_ccl_fra_id, i0, i1, j0, j1, p_cca_roi_max2));

    // --------------------- //
    // -- DATA ALLOCATION -- //
    // --------------------- //

    sigma_delta_data_t* sd_data1 = sigma_delta_alloc_data(i0, i1, j0, j1, 1, 254);
    morpho_data_t* morpho_data1 = morpho_alloc_data(i0, i1, j0, j1);
    CCL_data_t* ccl_data1 = CCL_LSL_alloc_data(i0, i1, j0, j1);
    kNN_data_t* knn_data = kNN_alloc_data(p_cca_roi_max2);
    tracking_data_t* tracking_data = tracking_alloc_data(MAX(p_trk_obj_min, p_trk_ext_o) + 1, p_cca_roi_max2);

    Logger_RoIs log_RoIs(p_log_path ? p_log_path : "", p_vid_in_start, p_vid_in_skip, p_cca_roi_max2, tracking_data);
    Logger_kNN log_kNN(p_log_path ? p_log_path : "", p_vid_in_start, p_cca_roi_max2);
    Logger_tracks log_trk(p_log_path ? p_log_path : "", p_vid_in_start, tracking_data);

    std::unique_ptr<Visu> visu;
    if (p_vid_out_play || p_vid_out_path) {
        const uint8_t n_threads = 1;
        visu.reset(new Visu(p_vid_out_path, p_vid_in_start, n_threads, i0, i1, j0, j1, PIXFMT_GRAY8, PIXFMT_RGB24,
                            VCDC_FFMPEG_IO, p_vid_out_id, p_vid_out_play, p_trk_obj_min, p_cca_roi_max2, p_vid_in_skip,
                            tracking_data));
    }
    
    // ------------------------- //
    // -- DATA INITIALISATION -- //
    // ------------------------- //
    morpho_init_data(morpho_data1);
    CCL_LSL_init_data(ccl_data1);
    kNN_init_data(knn_data);
    tracking_init_data(tracking_data);

	// ---------------------- //
	// -- Initialization t -- //
	// ---------------------- //
	Sigma_delta sigma_delta_mod1(sd_data1, i0, i1, j0, j1, p_sd_n);
    Morpho morpho_mod1(morpho_data1, i0, i1, j0, j1);
    CCL ccl_mod1(ccl_data1, def_p_cca_roi_max1);
    Features_CCA features_mod1(i0, i1, j0, j1, p_cca_roi_max1);
    Features_filter features_filter_mod1(i0, i1, j0, j1, p_cca_roi_max1,
                    p_flt_s_min, p_flt_s_max, p_cca_roi_max2);

	// ----------------------------- //
	// -- Associations (t - 1, t) -- //
	// ----------------------------- //
	KNN knn_mod(knn_data, p_cca_roi_max2, p_knn_k, p_knn_d, p_knn_s, p_log_path != NULL);
    Tracking tracking_mod(tracking_data, p_cca_roi_max2, p_trk_ext_d, p_trk_obj_min,
    						p_trk_roi_path != NULL || visu, p_trk_ext_o, p_knn_s);

	// delayer initialization
	spu::module::Delayer<uint32_t> delayer_n_RoIs(1, 0);
    spu::module::Delayer<uint8_t> delayer_RoIs(p_cca_roi_max2 * sizeof(RoI_t), 0);

	TIME_POINT(stop_alloc_init);

	// ----------------------- //
	// -- GRAPH DEFINITION -- //
	// ---------------------- //

    printf("# Allocations and initialisations took %6.3f sec\n", TIME_ELAPSED2_SEC(start_alloc_init, stop_alloc_init));

    size_t n_moving_objs = 0, n_processed_frames = 0;

    // run video in order to initialize correctly the first frame
    video("generate").exec();

    //delayer
    //delayer.set_data(video["generate::out_img_gray8"].get_dataptr<uint8_t>());
    //delayer["memorize::in"] = video["generate::out_img_gray8"];

	// initialize sigma delta with the first frame
	sigma_delta_mod1.sigma_delta_init((const uint8_t**)video["generate::out_img_gray8"].get_2d_dataptr<uint8_t>());
    
    // --------------------- //
    // -- Processing at t -- //
    // --------------------- //
    {
        // step 1: motion detection (per pixel) with Sigma-Delta algorithm
        sigma_delta_mod1["compute::in_img"] = video["generate::out_img_gray8"];
        // step 2: mathematical morphology
        morpho_mod1["compute::in_img"] = sigma_delta_mod1["compute::out_img"];
        // step 3: connected components labeling (CCL)
        ccl_mod1["apply::in_img"] = morpho_mod1["compute::out_img"];
        // step 4: connected components analysis (CCA): from image of labels to "regions of interest" (RoIs)
        features_mod1["extract::in_labels"] = ccl_mod1["apply::out_labels"];
        features_mod1["extract::in_n_RoIs"] = ccl_mod1["apply::out_n_RoIs"];
        // step 5: surface filtering (rm too small and too big RoIs)
        features_filter_mod1["filterf::in_n_RoIs"] = ccl_mod1["apply::out_n_RoIs"];
        features_filter_mod1["filterf::in_RoIs"] = features_mod1["extract::out_RoIs"];
        features_filter_mod1["filterf::fwd_labels"] = ccl_mod1["apply::out_labels"];

    }

    delayer_n_RoIs["memorize::in"] = features_filter_mod1["filterf::out_n_RoIs"];
    delayer_RoIs["memorize::in"] = features_filter_mod1["filterf::out_RoIs"];

    // ----------------------------- //
    // -- Associations (t - 1, t) -- //
    // ----------------------------- //
    {
        // step 6: k-NN matching (RoIs associations)
        knn_mod["matchf::in_n_RoIs0"] = delayer_n_RoIs["produce::out"];
        knn_mod["matchf::in_n_RoIs1"] = features_filter_mod1["filterf::out_n_RoIs"];
        knn_mod["matchf::fwd_RoIs0"] = delayer_RoIs["produce::out"];
        knn_mod["matchf::fwd_RoIs1"] = features_filter_mod1["filterf::out_RoIs"];

        // step 7: temporal tracking
        tracking_mod["perform::in_n_RoIs"] = features_filter_mod1["filterf::out_n_RoIs"];
        tracking_mod["perform::in_RoIs"] = knn_mod["matchf::fwd_RoIs1"];

        tracking_mod["perform::in_frame"]= video["generate::out_frame"];    
    }
	
    // --------------------- //
	// -- LOGGING & VISU -- //
	// --------------------- //
    {
        // save frames (CCs)
        if (p_ccl_fra_path) {
            (*log_fra)["write::in_labels"] = features_filter_mod1["filterf::out_labels"];
            (*log_fra)["write::in_RoIs"] = tracking_mod["perform::out_RoIs"];
            (*log_fra)["write::in_n_RoIs"] = tracking_mod["perform::out_n_RoIs"];
        }

        // save stats
        if (p_log_path) {
            log_RoIs["write::in_RoIs0"] = delayer_RoIs["produce::out"];
            log_RoIs["write::in_n_RoIs0"] = delayer_n_RoIs["produce::out"];
            log_RoIs["write::in_RoIs1"] = features_filter_mod1["filterf::out_RoIs"];
            log_RoIs["write::in_n_RoIs1"] = features_filter_mod1["filterf::out_n_RoIs"];

            log_RoIs["write::in_frame"] = video["generate::out_frame"];

            // always enable associations logging
            //if (cur_fra > (uint32_t)p_vid_in_start) {
            log_kNN["write::in_nearest"] = knn_mod["matchf::out_nearest"]; //.bind(knn_data->nearest[0]);
            log_kNN["write::in_distances"] = knn_mod["matchf::out_distances"]; //.bind(knn_data->distances[0]);
#ifdef MOTION_ENABLE_DEBUG
            log_kNN["write::in_conflicts"] = knn_mod["matchf::out_conflicts"]; //.bind(knn_data->conflicts[0]);
#endif
            log_kNN["write::in_RoIs0"] = delayer_RoIs["produce::out"];
            log_kNN["write::in_n_RoIs0"] = delayer_n_RoIs["produce::out"];
            log_kNN["write::in_RoIs1"] = features_filter_mod1["filterf::out_RoIs"];
            log_kNN["write::in_n_RoIs1"] = features_filter_mod1["filterf::out_n_RoIs"];

            log_kNN["write::in_frame"] = video["generate::out_frame"];

            log_trk["write::in_frame"] = video["generate::out_frame"];
            //}
        }

        // display the result to the screen or write it into a video file
        if (visu) {
            (*visu)["display::in_frame"] = video["generate::out_frame"];
            (*visu)["display::in_img"] = video["generate::out_img_gray8"];
            (*visu)["display::in_RoIs"] = knn_mod["matchf::fwd_RoIs1"];
            (*visu)["display::in_n_RoIs"] = features_filter_mod1["filterf::out_n_RoIs"];
            (*visu)("display").exec();
        }
    }

	// --------------------- //
    // ----- PIPELINE ------ //
    // --------------------- //
    std::vector<std::tuple<
        std::vector<spu::runtime::Task*>, 
        std::vector<spu::runtime::Task*>,
        std::vector<spu::runtime::Task*>>> pip_stages = {
			std::make_tuple<std::vector<spu::runtime::Task*>, 
				std::vector<spu::runtime::Task*>,
				std::vector<spu::runtime::Task*>>(
					{&video("generate"), },
					{&sigma_delta_mod1("compute")},
					{}      
				),
			std::make_tuple<std::vector<spu::runtime::Task*>, 
				std::vector<spu::runtime::Task*>,
				std::vector<spu::runtime::Task*>>(
					{&morpho_mod1("compute"),},
					{&features_filter_mod1("filterf")},
					{}
				),
			std::make_tuple<std::vector<spu::runtime::Task*>, 
				std::vector<spu::runtime::Task*>,
				std::vector<spu::runtime::Task*>>(
					{&delayer_n_RoIs("produce"), &delayer_RoIs("produce"), 
                        &delayer_n_RoIs("memorize"), &delayer_RoIs("memorize"), 
                        &knn_mod("matchf"), 
                        &tracking_mod("perform")},
					{},
					{}
				)
    };

    if(p_ccl_fra_path){
        std::get<2>(pip_stages[1]).push_back(&(*log_fra)("write"));
        std::get<0>(pip_stages[2]).push_back(&(*log_fra)("write"));
    }
    if(p_log_path){
        std::get<2>(pip_stages[0]).push_back(&log_RoIs("write")); // Why if I remove this still work?
        std::get<2>(pip_stages[1]).push_back(&log_RoIs("write"));
		std::get<0>(pip_stages[2]).push_back(&log_RoIs("write"));
        // log_kNN
        std::get<2>(pip_stages[0]).push_back(&log_kNN("write"));
        std::get<2>(pip_stages[1]).push_back(&log_kNN("write"));
		std::get<0>(pip_stages[2]).push_back(&log_kNN("write"));

        // log_trk
        std::get<2>(pip_stages[0]).push_back(&log_trk("write"));
        std::get<2>(pip_stages[1]).push_back(&log_trk("write"));
        std::get<0>(pip_stages[2]).push_back(&log_trk("write"));
    }
    if(visu){
        std::get<2>(pip_stages[0]).push_back(&(*visu)("display"));
        std::get<2>(pip_stages[1]).push_back(&(*visu)("display"));
        std::get<0>(pip_stages[2]).push_back(&(*visu)("display"));
    }

    std::vector<spu::runtime::Task*> seq_first_tasks = {&delayer_n_RoIs("produce"), &delayer_RoIs("produce"), &video("generate")};

    spu::runtime::Pipeline pipeline(seq_first_tasks, pip_stages, 
        {   1,      4,     1 }, 
        {   1,             1 },
        {       false,  false},
        {true,  true,   true },
        {"PU0|PU1,PU2,PU3,PU4|PU5"});

    
    // Enabling statistics
    for (auto& mdl : pipeline.get_modules<spu::module::Module>(false))
        for (auto& tsk : mdl->tasks)
                tsk->set_stats(p_stats);


    TIME_POINT(start_compute);
    pipeline.exec({
        [] (const std::vector<const int*>& statuses) { return false; }, 
        [] (const std::vector<const int*>& statuses) { return false; }, 
        [&video, tracking_data, &n_processed_frames] (const std::vector<const int*>& statuses) { 
            n_processed_frames++;
            unsigned long n_moving_objs = tracking_count_objects(tracking_data->tracks);
            fprintf(stderr, " -- Processed frames = %ld", n_processed_frames);
            fprintf(stderr, " -- Tracks = %3lu\r", n_moving_objs);
            fflush(stderr);
            return false; // video.is_done();??
        }  
    });
    TIME_POINT(stop_compute);
    
    n_moving_objs = tracking_count_objects(tracking_data->tracks);

    fprintf(stderr, " -- Time = %6.3f sec", TIME_ELAPSED2_SEC(start_compute, stop_compute));
    fprintf(stderr, " -- FPS = %4d", (int)(n_processed_frames / (TIME_ELAPSED2_SEC(start_compute, stop_compute))));
    fprintf(stderr, " -- Tracks = %3lu\r", (unsigned long)n_moving_objs);
    fflush(stderr);
    
    //TIME_POINT(stop_compute);
    fprintf(stderr, "\n");
    
    if (p_trk_roi_path) {
        FILE* f = fopen(p_trk_roi_path, "w");
        if (f == NULL) {
            fprintf(stderr, "(EE) error while opening '%s'\n", p_trk_roi_path);
            exit(1);
        }
        tracking_tracks_RoIs_id_write(f, tracking_data->tracks);
        fclose(f);
    }
    tracking_tracks_write(stdout, tracking_data->tracks);

    printf("# Tracks statistics:\n");
    printf("# -> Processed frames = %4u\n", (unsigned)n_processed_frames);
    printf("# -> Detected tracks  = %4lu\n", (unsigned long)n_moving_objs);
    printf("# -> Took %6.3f seconds (avg %d FPS)\n", TIME_ELAPSED2_SEC(start_compute, stop_compute),
           (int)(n_processed_frames / (TIME_ELAPSED2_SEC(start_compute, stop_compute))));

    // some frames have been buffered for the visualization, display or write these frames here
    if (visu)
        visu->flush();

    // print stats
    if(p_stats) {
        const bool ordered = true;                  
        const bool display_throughput = false;      
        auto stages = pipeline.get_stages();        
        for (size_t s = 0; s < stages.size(); s++) {
            const int n_threads = stages[s]->get_n_threads();  
            std::cout << "#" << std::endl;
            std::cout << "# Pipeline stage " << (s + 1) << " (" << n_threads << " thread(s)): " << std::endl;
            spu::tools::Stats::show(stages[s]->get_tasks_per_types(), ordered, display_throughput);
        }
    }


    // ---------- //
    // -- FREE -- //
    // ---------- //
    morpho_free_data(morpho_data1);
    CCL_LSL_free_data(ccl_data1);
    kNN_free_data(knn_data);
    tracking_free_data(tracking_data);

    printf("#\n");
    printf("# End of the program, exiting.\n");

    return EXIT_SUCCESS;
}