#pragma once
#include "slam_types.h"
#include "core/tensorrt_inference.h"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include <string>

constexpr float SCORE_THRESHOLD = 0.70f;
constexpr int IMAGE_WIDTH = 1241;
constexpr int IMAGE_HEIGHT = 376;
constexpr int MODEL_IMAGE_SIZE = 1024;

namespace slam_core {
    std::vector<cv::Point3f> triangulatePoints(const cv::Mat& K, const cv::Mat& R1, const cv::Mat& T1,
                                              const cv::Mat& R2, const cv::Mat& T2,
                                              std::vector<cv::Point2f>& points1,
                                              std::vector<cv::Point2f>& points2,
                                              const cv::Mat& mask,
                                              const std::vector<int>& exclude_indices);

    cv::Mat load_calibration(const std::string& path);

    std::vector<cv::Mat> load_poses(const std::string& path, int num_poses);

    void estimate_pose(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2,
                      const cv::Mat& K, cv::Mat& R, cv::Mat& T);

    void bundleAdjustment(std::vector<cv::Mat>& Rs_est, std::vector<cv::Mat>& Ts_est,
                         std::vector<Point3D>& points3D, const cv::Mat& K);

    double compute_rotation_error(const cv::Mat& R_est, const cv::Mat& R_gt);

    double compute_translation_error(const cv::Mat& T_est, const cv::Mat& T_gt);

    void process_keypoints(TensorRTInference& infer, const cv::Mat& img1, const cv::Mat& img2,
                          std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2,
                          int i);
}