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

    return 0;
}