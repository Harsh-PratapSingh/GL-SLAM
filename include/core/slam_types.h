#pragma once
#include <opencv2/opencv.hpp>
#include <core/lightglue.h>
#include <core/superpoint.h>
#include <core/keypt2subpx.h>
#include <condition_variable>

struct CovisibleKeyframe {
    int keyframe_id;
    int shared_map_points;
};

struct Observation {
    int keyframe_id;
    cv::Point2d point2D;
    int kp_index; 
    // const Frame& kf = map.keyframes[obs.keyframe_id];
    // const float* desc = &kf.descriptors[obs.kp_index*256]; // For when to acces descriptors later
};

struct MapPoint {
    int id;
    cv::Point3d position;            
    std::vector<Observation> obs;  
    bool is_bad = false; 
};

struct Frame {
    int id;
    cv::Mat img;              
    cv::Mat R;                
    cv::Mat t;                

    SuperPointTRT::Result sp_res;
    std::vector<int64_t> keypoints;
    std::vector<float> descriptors;
    std::vector<int> map_point_ids;
    std::vector<int> kp_to_mpid;    
    bool is_keyframe = false;

    std::vector<CovisibleKeyframe> CovisibleKeyframes;

    // Helper to add without duplicates (weight immutable)
    bool add_covisible_keyframe(int kf_id, int shared_count, int threshold = 150) {
        if (shared_count <= threshold) return false;
        for (const auto& ckf : CovisibleKeyframes) {
            if (ckf.keyframe_id == kf_id) {
                return false;  // Duplicate, skip (no weight update)
            }
        }
        CovisibleKeyframes.push_back({kf_id, shared_count});
        return true;
    }
};

struct Map {
    std::unordered_map<int, MapPoint> map_points;
    std::unordered_map<int, Frame> keyframes;
    int next_point_id = 0;
    int next_keyframe_id = 0;
};


struct Match2D2D {
    int idx0;         
    int idx1;         
    cv::Point2d p0;   
    cv::Point2d p1;  
};

struct ObsPairs {
    int mpid;
    int idx1;
    cv::Point2d p1;
};

struct SyntheticMatch {
    int idx_curr_frame;
    int mpid;
};

namespace slam_types {
    
    extern const float match_thr;
    extern const float map_match_thr;
    extern const int map_match_window;
    extern const int Full_ba_window_size;
    extern const int Full_ba_include_past_optimized_frame_size;
    extern const float mag_filter;
    extern const float rot_filter;
    extern int max_idx;   
    extern int run_window;   
    extern int covisible_edge_threshold;

    extern bool run_pose_ba;
    extern bool cull_map_points;

    extern Map map;
    extern std::mutex map_mutex; 
    extern std::mutex local_ba_mutex;
    extern std::mutex tracking_mutex;

    extern std::string img_dir_path;
    extern std::string calibPath;
    extern std::string posesPath;

    extern SuperPointTRT sp;
    extern LightGlueTRT lg;
    extern Keypt2SubpxTRT ks;

    extern std::condition_variable cv_local_ba;     
    extern bool local_ba_start; 
    extern int local_ba_window;
    extern bool local_ba_done; 

    extern double reprog_error_threshold_for_old_points;
    extern int obs_count_threshold_for_old_points;

    extern std::vector<int> mpid_to_correct;
    extern std::vector<int> kpid_to_correct;

    
}

