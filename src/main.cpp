#include "core/slam_core.h"
// #include "visualization/visualization.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>


int main() { 

    std::string dir_path = "/home/tomato/Downloads/data_odometry_gray/dataset/sequences/00/";
    TensorRTInference infer("../third_party/Superpoint_Lightglue/superpoint_lightglue_pipeline.trt.onnx",
                           "../third_party/Superpoint_Lightglue/superpoint_lightglue_pipeline.trt");

    std::ostringstream oss1, oss2;
    oss1 << dir_path << "image_0/" << std::setw(6) << std::setfill('0') << 0 << ".png";
    oss2 << dir_path << "image_0/" << std::setw(6) << std::setfill('0') << 1 << ".png";

    cv::Mat img1 = cv::imread(oss1.str(), cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(oss2.str(), cv::IMREAD_GRAYSCALE);

    std::vector<cv::Point2f> img1_points_combined; img1_points_combined.reserve(4000);
    std::vector<cv::Point2f> img2_points_combined; img2_points_combined.reserve(4000);

    slam_core::process_keypoints(infer, img1, img2, img1_points_combined, img2_points_combined);


    // Quick visualization on original image1 to confirm placement
    cv::Mat img1_vis; cv::cvtColor(img1, img1_vis, cv::COLOR_GRAY2BGR);
    for (const auto& p : img1_points_combined) {
        cv::circle(img1_vis, cv::Point(cvRound(p.x), cvRound(p.y)), 1, cv::Scalar(0, 255, 0), 1);
    }
    cv::imshow("img1_combined_keypoints", img1_vis);
    cv::waitKey(0);
    cv::imwrite("img1_combined_keypoints.png", img1_vis);

    // If needed, also visualize for image2
    cv::Mat img2_vis; cv::cvtColor(img2, img2_vis, cv::COLOR_GRAY2BGR);
    for (const auto& p : img2_points_combined) cv::circle(img2_vis, cv::Point(cvRound(p.x), cvRound(p.y)), 1, cv::Scalar(0, 0, 255), 1);
    cv::imshow("img2_combined_keypoints", img2_vis); cv::waitKey(0); cv::imwrite("img2_combined_keypoints.png", img2_vis);


    cv::Mat side_by_side(img1.rows, img1.cols + img2.cols, CV_8UC3);
    cv::Mat left_roi = side_by_side(cv::Rect(0, 0, img1.cols, img1.rows));
    cv::Mat right_roi = side_by_side(cv::Rect(img1.cols, 0, img2.cols, img2.rows));
    cv::cvtColor(img1, left_roi, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, right_roi, cv::COLOR_GRAY2BGR);

    // Draw lines between matched points
    for (size_t i = 0; i < img1_points_combined.size() && i < img2_points_combined.size(); ++i) {
        const cv::Point2f& pt1 = img1_points_combined[i];
        const cv::Point2f& pt2 = img2_points_combined[i];
        // Draw circles on keypoints
        cv::circle(side_by_side, pt1, 2, cv::Scalar(0, 255, 0), 2);
        cv::Point2f pt2_shifted(pt2.x + img1.cols, pt2.y);
        cv::circle(side_by_side, pt2_shifted, 3, cv::Scalar(0, 0, 255), 2);

        // Draw connecting line
        cv::line(side_by_side, pt1, pt2_shifted, cv::Scalar(255, 0, 0), 1);
    }

    // Show and save the result
    cv::imshow("Matches Between Images Side-by-Side", side_by_side);
    cv::waitKey(0);
    cv::imwrite("img_matches_side_by_side.png", side_by_side);

    return 0;
}