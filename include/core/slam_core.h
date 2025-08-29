#pragma once
#include "slam_types.h"
#include "core/tensorrt_inference.h"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include <string>
#include "superpoint.h"
#include "lightglue.h"

constexpr float SCORE_THRESHOLD = 0.5f;
constexpr int IMAGE_WIDTH = 1241;
constexpr int IMAGE_HEIGHT = 376;
constexpr int MODEL_IMAGE_SIZE = 1024;

namespace slam_core {
    // std::vector<cv::Point3f> triangulatePoints(const cv::Mat& K, const cv::Mat& R1, const cv::Mat& T1,
    //                                           const cv::Mat& R2, const cv::Mat& T2,
    //                                           std::vector<cv::Point2f>& points1,
    //                                           std::vector<cv::Point2f>& points2,
    //                                           const cv::Mat& mask,
    //                                           const std::vector<int>& exclude_indices);

    cv::Mat load_camera_matrix(const std::string& calibPath);

    std::vector<cv::Mat> load_poses(const std::string& path);


    // void estimate_pose(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2,
    //                   const cv::Mat& K, cv::Mat& R, cv::Mat& T);

    // void bundleAdjustment(std::vector<cv::Mat>& Rs_est, std::vector<cv::Mat>& Ts_est,
    //                      std::vector<Point3D>& points3D, const cv::Mat& K);

    // double compute_rotation_error(const cv::Mat& R_est, const cv::Mat& R_gt);

    // double compute_translation_error(const cv::Mat& T_est, const cv::Mat& T_gt);

    void process_keypoints(TensorRTInference& infer, const cv::Mat& img1, const cv::Mat& img2,
                          std::vector<cv::Point2f>& img1_points_combined, std::vector<cv::Point2f>& img2_points_combined
                          );

    // void processSlidingWindowBA(const int i, const int window_size, const cv::Mat K, std::vector<cv::Mat> &Rs_est, 
    //                             std::vector<cv::Mat> &Ts_est, std::vector<Point3D> &global_points3D);


    void superpoint_lightglue_init(SuperPointTRT& sp, LightGlueTRT& lg);
                        
    std::vector<Match2D2D> lightglue_score_filter(LightGlueTRT::Result& result, const float& score);
    
    std::tuple<cv::Mat, cv::Mat, cv::Mat> pose_estimator(std::vector<Match2D2D>& matches, cv::Mat& K);

    std::vector<Match2D2D> pose_estimator_mask_filter(std::vector<Match2D2D>& matches, cv::Mat mask);

    cv::Mat adjust_translation_magnitude(std::vector<cv::Mat>& gtPoses, cv::Mat& t, int frame);

    std::tuple<std::vector<cv::Point3d>, std::vector<Match2D2D>> triangulate_and_filter_3d_points(
        cv::Mat& R1, cv::Mat& t1, cv::Mat& R2, cv::Mat& t2, cv::Mat& K, std::vector<Match2D2D> matches,
        const float& distance_threshold, const float& reprojection_threshold);

    void update_map_and_keyframe_data(Map& map, cv::Mat& img, cv::Mat& R, cv::Mat t,
        SuperPointTRT::Result& Result, std::vector<cv::Point3d>& points3d,
        std::vector<Match2D2D>& filteredPairs, SuperPointTRT::Result& f_res,
        cv::Mat& f_img, std::vector<int>& map_point_id, std::vector<int>& kp_index, bool if_first_frame, bool if_R_t_inversed);
       
    std::unordered_map<int, SyntheticMatch> get_matches_from_previous_frames(
        LightGlueTRT& lg,
        Map& map,
        int prev_frame_id,
        int i,
        cv::Mat& K,
        SuperPointTRT::Result& sp_res2);
}