#pragma once
#include <opencv2/opencv.hpp>

struct Observation {
    int camera_idx;       // Index of the camera/image
    cv::Point2f point2D;  // Observed 2D point in the image
};

struct Point3D {
    cv::Point3f position;            // 3D position in world coordinates
    std::vector<Observation> observations; // List of observations in different images
};