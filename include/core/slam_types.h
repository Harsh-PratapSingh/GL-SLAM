#pragma once
#include <opencv2/opencv.hpp>

// struct Observation {
//     int camera_idx;       // Index of the camera/image
//     cv::Point2f point2D;  // Observed 2D point in the image
// };

// struct Point3D {
//     cv::Point3f position;            // 3D position in world coordinates
//     std::vector<Observation> observations; // List of observations in different images
// };

// A 2D observation of a 3D point in a specific keyframe
struct Observation {
    int keyframe_id;
    cv::Point2f point2D;
};

// Persistent 3D Map Point
struct MapPoint {
    int id;
    cv::Point3f position;            // World coordinates
    std::vector<Observation> obs;    // Observations in various keyframes
    bool is_bad = false;
};

// Each Image Frame (will become keyframe or be discarded)
struct Frame {
    int id;
    cv::Mat img;              // Grayscale or RGB
    cv::Mat R;                // 3x3 rotation (world <- camera)
    cv::Mat t;                // 3x1 translation (world <- camera)
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<int> map_point_ids; // index into global MapPoints
    bool is_keyframe = false;
};

// Tracks both temporary and permanent entities
struct Map {
    std::unordered_map<int, MapPoint> map_points; // Permanent landmarks
    std::unordered_map<int, Frame> keyframes;     // Permanent keyframes
    int next_point_id = 0;
    int next_keyframe_id = 0;
};