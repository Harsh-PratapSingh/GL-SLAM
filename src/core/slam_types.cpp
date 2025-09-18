#include "core/slam_types.h"

namespace slam_types {

    const float match_thr = 0.7f;
    const float map_match_thr = 0.7f;
    const int map_match_window = 20;
    const int Full_ba_window_size = 7;  //7
    const int Full_ba_include_past_optimized_frame_size = 3;  //3
    const float mag_filter = 0.05f;
    const float rot_filter = 0.3f;
    int max_idx   = 4540;           // max 4540

    int run_window = -1;

    Map map;
    std::mutex map_mutex;  
    std::mutex local_ba_mutex;
    std::mutex tracking_mutex;

    std::string img_dir_path = "/home/tomato/Downloads/data_odometry_gray/dataset/sequences/00/image_0/";
    std::string calibPath = "/home/tomato/Downloads/data_odometry_gray/dataset/sequences/00/calib.txt";
    std::string posesPath = "/home/tomato/Downloads/data_odometry_gray/dataset/sequences/00/00.txt";

    SuperPointTRT sp;
    LightGlueTRT lg;
    Keypt2SubpxTRT ks;

    std::condition_variable cv_local_ba; 
    std::condition_variable run_tracking;
    bool tracking_frame = true;   
    bool local_ba_start = false; 
    int local_ba_window = 0;
    bool local_ba_done = false; 

    double reprog_error_threshold_for_old_points = 1.0f;
    int obs_count_threshold_for_old_points = 3;

    std::vector<int> mpid_to_correct;
    std::vector<int> kpid_to_correct;

}