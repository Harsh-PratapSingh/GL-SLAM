#include "core/lightglue.h"
#include "core/superpoint.h"
#include "core/slam_core.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cmath>


static std::vector<cv::Mat> loadKittiPoses4x4(const std::string& posesPath) {
    std::ifstream f(posesPath);
    if (!f.is_open()) throw std::runtime_error("Failed to open poses file");
    std::vector<cv::Mat> poses;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                iss >> T.at<double>(r, c);
            }
        }
        poses.push_back(T);
    }
    return poses;
}

static cv::Mat invertSE3(const cv::Mat& T) {
    cv::Mat R = T(cv::Rect(0,0,3,3)).clone();
    cv::Mat t = T(cv::Rect(3,0,1,3)).clone();
    cv::Mat Rt = R.t();
    cv::Mat Tinv = cv::Mat::eye(4,4,CV_64F);
    Rt.copyTo(Tinv(cv::Rect(0,0,3,3)));
    cv::Mat t_inv = -Rt * t;
    t_inv.copyTo(Tinv(cv::Rect(3,0,1,3)));
    return Tinv;
}

static double angleBetweenVectorsDeg(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat af, bf;
    a.convertTo(af, CV_64F);
    b.convertTo(bf, CV_64F);
    double na = cv::norm(af), nb = cv::norm(bf);
    if (na < 1e-9 || nb < 1e-9) return 0.0;
    double cosang = af.dot(bf) / (na * nb);
    cosang = std::max(-1.0, std::min(1.0, cosang));
    return std::acos(cosang) * 180.0 / CV_PI;
}

static double rotationAngleErrorDeg(const cv::Mat& R_est, const cv::Mat& R_gt) {
    cv::Mat R_err = R_gt.t() * R_est;
    double tr = std::max(-1.0, std::min(1.0, (R_err.at<double>(0,0) + R_err.at<double>(1,1) + R_err.at<double>(2,2) - 1.0) * 0.5));
    return std::acos(tr) * 180.0 / CV_PI;
}

int main() {
    SuperPointTRT sp;
    LightGlueTRT lg;
    slam_core::superpoint_lightglue_init(sp, lg);

    // Load camera matrix from calib.txt
    std::string calibPath = "/home/tomato/Downloads/data_odometry_gray/dataset/sequences/00/calib.txt";
    cv::Mat cameraMatrix = slam_core::loadCameraMatrix(calibPath);
    std::cout << "Camera Matrix:\n" << cameraMatrix << std::endl;

    std::string posesPath = "/home/tomato/Downloads/data_odometry_gray/dataset/sequences/00/00.txt"; // typical location
    // If your 00.txt sits next to calib.txt, adjust to that path.
    auto gtPoses = loadKittiPoses4x4(posesPath);

    cv::Mat img0 = cv::imread("temp3.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img1 = cv::imread("temp4.png", cv::IMREAD_GRAYSCALE);
    if (img0.empty() || img1.empty()) {
        throw std::runtime_error("Failed to load image0.png or image1.png");
    }
    SuperPointTRT::Result spRes0, spRes1;
    spRes0 = sp.runInference(img0, img0.rows, img0.cols);

    spRes1 = sp.runInference(img1, img1.rows, img1.cols);
    std::cout << "Image0: valid keypoints = " << spRes0.numValid << std::endl;
    std::cout << "Image1: valid keypoints = " << spRes1.numValid << std::endl;

    LightGlueTRT::Result lgRes;
    lgRes = lg.run_Direct_Inference(spRes0, spRes1);
    int nMatches = 0;
    for (int i = 0; i < spRes0.numValid; ++i) {
        if (lgRes.matches0[i] >= 0 && lgRes.mscores0[i] > 0.7) ++nMatches;
    }
    std::cout << "LightGlue matches: " << nMatches << " (of " << spRes0.numValid << " keypoints in image0)" << std::endl;

    // Extract matched keypoints
    std::vector<cv::Point2f> points0, points1;
    for (int i = 0; i < spRes0.numValid; ++i) {
        int match_idx = lgRes.matches0[i];
        if (match_idx >= 0 && lgRes.mscores0[i] > 0.7) {
            // Assuming keypoints are interleaved x,y as int64_t; cast to float
            float x0 = static_cast<float>(spRes0.keypoints[2 * i]);
            float y0 = static_cast<float>(spRes0.keypoints[2 * i + 1]);
            float x1 = static_cast<float>(spRes1.keypoints[2 * match_idx]);
            float y1 = static_cast<float>(spRes1.keypoints[2 * match_idx + 1]);
            points0.push_back(cv::Point2f(x0, y0));
            points1.push_back(cv::Point2f(x1, y1));
        }
    }
    
    if (points0.size() < 8) {  // Minimum for essential matrix
        std::cerr << "Not enough matches for pose estimation." << std::endl;
        return 1;
    }

    // cv::Mat T_w0 = gtPoses[0];
    cv::Mat T_w1 = gtPoses[1];
    // cv::Mat T_10_gt = T_w1 ;
    cv::Mat R_gt = T_w1(cv::Rect(0,0,3,3)).clone();
    cv::Mat t_gt = T_w1(cv::Rect(3,0,1,3)).clone();

    std::cout << "Pose : " << T_w1 << std::endl;
    double t_gt_mag = cv::norm(t_gt);
    
    
    // Compute essential matrix
    cv::Mat essentialMat, mask;
    essentialMat = cv::findEssentialMat(points0, points1, cameraMatrix, cv::USAC_MAGSAC, 0.999, 1.0, mask);
    
    // Recover relative pose
    cv::Mat R, t;
    int inliers = cv::recoverPose(essentialMat, points0, points1, cameraMatrix, R, t, mask);
    // R = R.t();
    // t = -R * t;

    // Scale translation to GT magnitude
    t *= (t_gt_mag / cv::norm(t));
    
    std::cout << "Recovered " << inliers << " inliers for pose estimation." << std::endl;
    std::cout << "Relative Rotation (R):\n" << R << std::endl;
    std::cout << "Relative Translation (t):\n" << t << std::endl;

    // Compare with GT
    double rot_err_deg = rotationAngleErrorDeg(R, R_gt);
    double t_dir_err_deg = angleBetweenVectorsDeg(t, t_gt);
    double t_mag_err = std::abs(cv::norm(t) - t_gt_mag);

    std::cout << "GT |t|: " << t_gt_mag << " m\n";
    std::cout << "Est |t| (scaled): " << cv::norm(t) << " m\n";
    std::cout << "Rotation error: " << rot_err_deg << " deg\n";
    std::cout << "Translation direction error: " << t_dir_err_deg << " deg\n";
    std::cout << "Translation magnitude error: " << t_mag_err << " m\n";

    // --- Visualization of Inlier Matches ---

    
    // Extract inliers immediately (using mask)
    std::vector<cv::Point2f> inlierPoints0, inlierPoints1;
    for (size_t i = 0; i < points0.size(); ++i) {
        if (mask.at<unsigned char>(i) > 0) {
            inlierPoints0.push_back(points0[i]);
            inlierPoints1.push_back(points1[i]);
        }
    }
    std::cout << "Extracted " << inlierPoints1.size() << " inlier points on second image." << std::endl;

    // Projection matrices: P0 = K [I|0], P1 = K [R|t]
    cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat Rt(3, 4, CV_64F), P0(3, 4, CV_64F), P1(3, 4, CV_64F);
    I.copyTo(P0.colRange(0, 3)); // [I|0]
    P0.col(3) = cv::Mat::zeros(3, 1, CV_64F);
    R.copyTo(Rt.colRange(0, 3));
    t.copyTo(Rt.col(3));
    P0 = cameraMatrix * P0;
    P1 = cameraMatrix * Rt;

    // Triangulate
    cv::Mat X4;
    cv::triangulatePoints(P0, P1, inlierPoints0, inlierPoints1, X4);
    X4.convertTo(X4,CV_64FC1);

    // Convert to Nx3 3D points (dehomogenize)
    std::vector<cv::Point3d> points3d;
    std::vector<cv::Point2f> P_0 = inlierPoints0;
    std::vector<cv::Point2f> P_1 = inlierPoints1;
    inlierPoints0.clear();
    inlierPoints1.clear();
    points3d.reserve(X4.cols);
    for (int i = 0; i < X4.cols; ++i) {
        double w = X4.at<double>(3, i);
        double X = X4.at<double>(0, i) / w;
        double Y = X4.at<double>(1, i) / w;
        double Z = X4.at<double>(2, i) / w;
        if (std::abs(w) < 1e-9 || Z <= 0 || Z > 100 ) continue;
        points3d.emplace_back(X, Y, Z);
        inlierPoints0.push_back(P_0[i]);
        inlierPoints1.push_back(P_1[i]);
    }

    std::cout << "Triangulated " << points3d.size() << " 3D points." << std::endl;

    //bookkeeping
    Map map;
    Frame frame0, frame1;
    frame0.id = map.next_keyframe_id++;
    frame1.id = map.next_keyframe_id++;
    frame0.img = img0;
    frame1.img = img1;
    frame0.R = cv::Mat::eye(3,3,CV_64F);
    frame0.t = cv::Mat::zeros(3,1,CV_64F);
    frame1.R = R.clone();
    frame1.t = t.clone();
    frame0.keypoints = spRes0.keypoints;
    frame1.keypoints = spRes1.keypoints;
    frame0.descriptors = spRes0.descriptors;
    frame0.descriptors = spRes0.descriptors;
    frame0.is_keyframe = true;
    frame1.is_keyframe = true;

    // --- Associate triangulated points with MapPoints and Observations ---
    for (size_t i = 0; i < points3d.size(); ++i) {
        MapPoint mp;
        mp.id = map.next_point_id++;
        mp.position = cv::Point3f(points3d[i].x, points3d[i].y, points3d[i].z);

        // Observations: two per map point, one from each frame
        Observation obs0, obs1;
        obs0.keyframe_id = frame0.id;
        obs0.point2D = inlierPoints0[i];

        // Copy descriptor for frame0 (use your method for extracting correct descriptor idx)
        obs0.descriptor.resize(256);
        for (int d = 0; d < 256; ++d)
            obs0.descriptor[d] = frame0.descriptors[i*256 +d];

        obs1.keyframe_id = frame1.id;
        obs1.point2D = inlierPoints1[i];
        obs1.descriptor.resize(256);
        for (int d = 0; d < 256; ++d)
            obs1.descriptor[d] = frame1.descriptors[i*256 +d];;

        mp.obs.push_back(obs0);
        mp.obs.push_back(obs1);

        map.map_points[mp.id] = mp;
        map.keyframes[frame0.id].map_point_ids.push_back(mp.id);
        map.keyframes[frame1.id].map_point_ids.push_back(mp.id);
    }
    std::cout << "Map contains " << map.map_points.size() << " MapPoints and "
            << map.keyframes.size() << " KeyFrames." << std::endl;
    
    // Visualize inliers on the second image
    cv::Mat img1_color;
    cv::cvtColor(img1, img1_color, cv::COLOR_GRAY2BGR);
    for (const auto& pt : inlierPoints1) {
        cv::circle(img1_color, pt, 1, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    }
    cv::imshow("Inliers on Second Image", img1_color);
    cv::waitKey(0);


    return 0;
}
