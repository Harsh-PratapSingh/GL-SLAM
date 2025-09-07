#pragma once
#include <opencv2/opencv.hpp>
#include <core/superpoint.h>

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
    cv::Point2d point2D;
    int kp_index; //NEW
    // const Frame& kf = map.keyframes[obs.keyframe_id];
    // const float* desc = &kf.descriptors[obs.kp_index*256]; // For when to acces descriptors later
};

struct MapPoint {
    int id;
    cv::Point3d position;            // World coordinates
    std::vector<Observation> obs;    // Observations in various keyframes
    bool is_bad = false;
    // No descriptor here as requested
};

struct Frame {
    int id;
    cv::Mat img;              // Grayscale or RGB
    cv::Mat R;                // 3x3 rotation (world <- camera)
    cv::Mat t;                // 3x1 translation (world <- camera)

    SuperPointTRT::Result sp_res;
    std::vector<int64_t> keypoints;
    // cv::Mat descriptors;      // CV_32F matrix: rows = num keypoints, cols = 256
    std::vector<float> descriptors;
    std::vector<int> map_point_ids;
    std::vector<int> kp_to_mpid;    // NEW: size N, filled after mapping
    bool is_keyframe = false;
};

struct Map {
    std::unordered_map<int, MapPoint> map_points;
    std::unordered_map<int, Frame> keyframes;
    int next_point_id = 0;
    int next_keyframe_id = 0;
};

// NEW: Compact record carrying original SP indices and 2D locations
struct Match2D2D {
    int idx0;         // SuperPoint index in frame0
    int idx1;         // SuperPoint index in frame1
    cv::Point2d p0;   // 2D point in frame0
    cv::Point2d p1;   // 2D point in frame1
};

struct ObsPairs {
    int mpid;
    int idx1;
    cv::Point2d p1;
};

//Match struct for when we match current frame with synthetic frame
struct SyntheticMatch {
    int idx_curr_frame;
    int mpid;
};


//A bad visual slam is quite easy to implement. I give you a rough roadmap:

// Feature extraction for input image

// Match features to previous image(s)

// Compute essential matrix

// Decompose essential matrix into pose

// Triangulate initial point cloud

// Check if initialisation is good (enough inliers)

// Once initialisation is done:

// Use constant velocity model to predict next pose

// Use predicted pose to project existing map points into the camera and check if they match with 2D features nearby

// Use successful 2D/3D correspondences to compute an optimised pose (motion only bundle adjustment)

// Use optimised frame to triangulate new points
