#pragma once
#include "slam_types.h"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include <string>
#include "core/superpoint.h"
#include "core/lightglue.h"
#include "core/keypt2subpx.h"

constexpr float SCORE_THRESHOLD = 0.5f;
constexpr int IMAGE_WIDTH = 1241;
constexpr int IMAGE_HEIGHT = 376;
constexpr int MODEL_IMAGE_SIZE = 1024;

namespace slam_core {

    cv::Mat load_camera_matrix(const std::string& calibPath);

    std::vector<cv::Mat> load_poses(const std::string& path);

    void superpoint_lightglue_init(SuperPointTRT& sp, LightGlueTRT& lg);
                        
    std::vector<Match2D2D> lightglue_score_filter(LightGlueTRT::Result& result,Keypt2SubpxTRT::Result& f_result, const float& score);
    
    std::tuple<cv::Mat, cv::Mat, cv::Mat> pose_estimator(std::vector<Match2D2D>& matches, cv::Mat& K);

    std::vector<Match2D2D> pose_estimator_mask_filter(std::vector<Match2D2D>& matches, cv::Mat mask);

    cv::Mat adjust_translation_magnitude(std::vector<cv::Mat>& gtPoses, cv::Mat& t, int frame);

    std::tuple<std::vector<cv::Point3d>, std::vector<Match2D2D>> triangulate_and_filter_3d_points(
        cv::Mat& R1, cv::Mat& t1, cv::Mat& R2, cv::Mat& t2, cv::Mat& K, std::vector<Match2D2D> matches,
        const float& distance_threshold, const float& reprojection_threshold);

    void update_map_and_keyframe_data(Map& map, cv::Mat& img, cv::Mat& R, cv::Mat t,
        SuperPointTRT::Result& Result, std::vector<cv::Point3d>& points3d,
        std::vector<Match2D2D>& filteredPairs, SuperPointTRT::Result& f_res,
        cv::Mat& f_img, std::vector<ObsPairs>& obsPairs, bool if_first_frame, bool if_R_t_inversed);
       
    std::unordered_map<int, SyntheticMatch> get_matches_from_previous_frames(
        LightGlueTRT& lg,
        Map& map,
        int prev_frame_id,
        int i,
        cv::Mat& K,
        SuperPointTRT::Result& sp_res2,
        float score
    );

    std::tuple<cv::Mat, cv::Mat, cv::Mat, SuperPointTRT::Result,
        std::vector<Match2D2D>, std::vector<ObsPairs>, bool> 
        run_pnp(Map& map, SuperPointTRT& sp, LightGlueTRT& lg, Keypt2SubpxTRT& k2s,
            std::string& img_dir_path, cv::Mat& cameraMatrix, float match_thr,
            float map_match_thr, int idx, int window, bool get_inliner, std::vector<cv::Mat>& gtPoses);

    bool full_ba(std::mutex& map_mutex, Map& map, cv::Mat& cameraMatrix, int window);

}