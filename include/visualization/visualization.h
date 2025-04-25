#pragma once
#include "core/slam_types.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace slam_visualization {
    void initialize_pangolin();

    void draw_frustum(float scale, float r, float g, float b);

    void visualize_poses(const std::vector<cv::Mat>& Rs_est, const std::vector<cv::Mat>& Ts_est,
                        const std::vector<cv::Mat>& Rs_gt, const std::vector<cv::Mat>& Ts_gt,
                        const std::vector<Point3D>& points3D, bool interactive = false);

    void visualize_optical_flow(const cv::Mat& img_current,
                               const std::vector<cv::Point2f>& points_prev,
                               const std::vector<cv::Point2f>& points_current,
                               const cv::Mat& mask, int frame_idx,
                               const std::vector<cv::Point2f>& projected_points = {});
}