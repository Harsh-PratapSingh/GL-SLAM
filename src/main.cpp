#include "core/lightglue.h"
#include "core/superpoint.h"
#include "core/slam_core.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

// NEW: Compact record carrying original SP indices and 2D locations
struct Match2D2D {
    int idx0;         // SuperPoint index in frame0
    int idx1;         // SuperPoint index in frame1
    cv::Point2f p0;   // 2D point in frame0
    cv::Point2f p1;   // 2D point in frame1
};

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

    std::string posesPath = "/home/tomato/Downloads/data_odometry_gray/dataset/sequences/00/00.txt";
    auto gtPoses = loadKittiPoses4x4(posesPath);

    cv::Mat img0 = cv::imread("temp3.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img1 = cv::imread("temp4.png", cv::IMREAD_GRAYSCALE);
    if (img0.empty() || img1.empty()) {
        throw std::runtime_error("Failed to load image0.png or image1.png");
    }

    SuperPointTRT::Result spRes0 = sp.runInference(img0, img0.rows, img0.cols);
    SuperPointTRT::Result spRes1 = sp.runInference(img1, img1.rows, img1.cols);
    std::cout << "Image0: valid keypoints = " << spRes0.numValid << std::endl;
    std::cout << "Image1: valid keypoints = " << spRes1.numValid << std::endl;

    LightGlueTRT::Result lgRes = lg.run_Direct_Inference(spRes0, spRes1);
    int nMatches = 0;
    const float match_thr = 0.7f;

    // Build compact matches directly
    std::vector<Match2D2D> matches;
    matches.reserve(spRes0.numValid);
    for (int i = 0; i < spRes0.numValid; ++i) {
        int j = lgRes.matches0[i];
        if (j >= 0 && lgRes.mscores0[i] > match_thr) {
            matches.push_back({
                i, j,
                cv::Point2f((float)lgRes.keypoints0[2*i],     (float)lgRes.keypoints0[2*i + 1]),
                cv::Point2f((float)lgRes.keypoints1[2*j],     (float)lgRes.keypoints1[2*j + 1])
            });
            ++nMatches;
        }
    }
    std::cout << "LightGlue matches: " << nMatches << " (of " << spRes0.numValid << " keypoints in image0)" << std::endl;

    if (matches.size() < 8) {
        std::cerr << "Not enough matches for pose estimation." << std::endl;
        return 1;
    }

    // Views for essential matrix
    std::vector<cv::Point2f> points0, points1;
    points0.reserve(matches.size());
    points1.reserve(matches.size());
    for (const auto& m : matches) {
        points0.push_back(m.p0);
        points1.push_back(m.p1);
    }

    // Compute essential matrix
    cv::Mat essentialMat, mask;
    essentialMat = cv::findEssentialMat(points0, points1, cameraMatrix, cv::USAC_MAGSAC, 0.999, 1.0, mask);

    // Recover relative pose
    cv::Mat R, t;
    int inliers = cv::recoverPose(essentialMat, points0, points1, cameraMatrix, R, t, mask);

    // Filter inliers using the mask into compact records
    std::vector<Match2D2D> inliersPairs;
    inliersPairs.reserve(matches.size());
    const uchar* mptr = mask.ptr<uchar>();
    for (size_t k = 0; k < matches.size(); ++k) {
        if (mptr[k]) inliersPairs.push_back(matches[k]);
    }
    std::cout << "Extracted " << inliersPairs.size() << " inlier matches." << std::endl;

    // R = R.t();
    // t = -R * t;
    // Scale translation to GT magnitude (compare against T_w1)
    cv::Mat T_w1 = gtPoses[1];
    cv::Mat t_gt = T_w1(cv::Rect(3,0,1,3)).clone();
    cv::Mat R_gt = T_w1(cv::Rect(0,0,3,3)).clone();
    double t_gt_mag = cv::norm(t_gt);
    t *= (t_gt_mag / cv::norm(t));

    // Projection matrices: P0 = K [I|0], P1 = K [R|t]
    cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat Rt(3, 4, CV_64F), P0(3, 4, CV_64F), P1(3, 4, CV_64F);
    I.copyTo(P0.colRange(0, 3)); // [I|0]
    P0.col(3) = cv::Mat::zeros(3, 1, CV_64F);
    R.copyTo(Rt.colRange(0, 3));
    t.copyTo(Rt.col(3));
    P0 = cameraMatrix * P0;
    P1 = cameraMatrix * Rt;

    // Triangulate using inlier pairs
    std::vector<cv::Point2f> inlierPoints0, inlierPoints1;
    inlierPoints0.reserve(inliersPairs.size());
    inlierPoints1.reserve(inliersPairs.size());
    for (const auto& m : inliersPairs) {
        inlierPoints0.push_back(m.p0);
        inlierPoints1.push_back(m.p1);
    }

    cv::Mat X4;
    cv::triangulatePoints(P0, P1, inlierPoints0, inlierPoints1, X4);
    X4.convertTo(X4, CV_64FC1);

    // Filter triangulated points and carry pairs forward
    std::vector<cv::Point3d> points3d;
    std::vector<Match2D2D> filteredPairs;
    points3d.reserve(X4.cols);
    filteredPairs.reserve(X4.cols);

    for (int i = 0; i < X4.cols; ++i) {
        double w = X4.at<double>(3, i);
        if (std::abs(w) < 1e-9) continue;
        double Z = X4.at<double>(2, i) / w;
        if (Z <= 0 || Z > 100) continue;

        double X = X4.at<double>(0, i) / w;
        double Y = X4.at<double>(1, i) / w;
        points3d.emplace_back(X, Y, Z);

        // Corresponding inlier pair at same index i
        filteredPairs.push_back(inliersPairs[i]);
    }

    std::cout << "Triangulated " << points3d.size() << " 3D points." << std::endl;

    // bookkeeping
    Map map;
    Frame frame0, frame1;
    frame0.id = map.next_keyframe_id++;
    frame1.id = map.next_keyframe_id++;
    frame0.img = img0;
    frame1.img = img1;

    frame0.R = cv::Mat::eye(3,3,CV_64F);
    frame0.t = cv::Mat::zeros(3,1,CV_64F);

    // Convert relative pose to camera1->camera2 then scale translation
    R = R.t();
    t = -R * t;
    // Scale translation to GT magnitude (compare against T_w1)
    // cv::Mat T_w1 = gtPoses[1];
    // cv::Mat t_gt = T_w1(cv::Rect(3,0,1,3)).clone();
    // cv::Mat R_gt = T_w1(cv::Rect(0,0,3,3)).clone();
    // double t_gt_mag = cv::norm(t_gt);
    // t *= (t_gt_mag / cv::norm(t));

    frame1.R = R.clone();
    frame1.t = t.clone();

    // Store SuperPoint outputs in frames (single source of truth)
    frame0.keypoints = spRes0.keypoints;
    frame1.keypoints = spRes1.keypoints;
    frame0.descriptors = spRes0.descriptors;
    frame1.descriptors = spRes1.descriptors;
    frame0.is_keyframe = true;
    frame1.is_keyframe = true;

    map.keyframes[frame0.id] = frame0;
    map.keyframes[frame1.id] = frame1;

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

    // --- Associate triangulated points with MapPoints and Observations ---
    // Size kp_to_mpid by SuperPoint keypoint counts
    map.keyframes[frame0.id].kp_to_mpid.assign(map.keyframes[frame0.id].keypoints.size()/2, -1);
    map.keyframes[frame1.id].kp_to_mpid.assign(map.keyframes[frame1.id].keypoints.size()/2, -1);

    for (size_t i = 0; i < points3d.size(); ++i) {
        const auto& pr = filteredPairs[i];

        MapPoint mp;
        mp.id = map.next_point_id++;
        mp.position = cv::Point3f(points3d[i].x, points3d[i].y, points3d[i].z);

        Observation obs0, obs1;

        // Frame 0 observation
        obs0.keyframe_id = frame0.id;
        obs0.kp_index = pr.idx0;
        {
            const auto& kps0 = map.keyframes[frame0.id].keypoints;
            obs0.point2D = cv::Point2f((float)kps0[2*obs0.kp_index], (float)kps0[2*obs0.kp_index+1]);
        }
        map.keyframes[frame0.id].kp_to_mpid[obs0.kp_index] = mp.id;

        // Frame 1 observation
        obs1.keyframe_id = frame1.id;
        obs1.kp_index = pr.idx1;
        {
            const auto& kps1 = map.keyframes[frame1.id].keypoints;
            obs1.point2D = cv::Point2f((float)kps1[2*obs1.kp_index], (float)kps1[2*obs1.kp_index+1]);
        }
        map.keyframes[frame1.id].kp_to_mpid[obs1.kp_index] = mp.id;

        mp.obs.push_back(obs0);
        mp.obs.push_back(obs1);

        map.map_points[mp.id] = std::move(mp);
        map.keyframes[frame0.id].map_point_ids.push_back(mp.id);
        map.keyframes[frame1.id].map_point_ids.push_back(mp.id);
    }
    std::cout << "Map contains " << map.map_points.size() << " MapPoints and "
              << map.keyframes.size() << " KeyFrames." << std::endl;

    // ===================== PnP on next image (temp5.png) =====================
    cv::Mat img2 = cv::imread("temp5.png", cv::IMREAD_GRAYSCALE);
    if (img2.empty()) {
        throw std::runtime_error("Failed to load temp5.png");
    }

    // 1) Run SuperPoint on temp5
    SuperPointTRT::Result spRes2 = sp.runInference(img2, img2.rows, img2.cols);
    std::cout << "Image2: valid keypoints = " << spRes2.numValid << std::endl;

    // 2) Match frame1 (map-stored SP data) to frame2 (fresh SP)
    LightGlueTRT::Result lgRes12 = lg.run_Direct_Inference(
        // Left: frame1 data from the map (SuperPoint arrays)
        // Wrap into SuperPointTRT::Result-like shallow containers if needed
        // Here we assume run_Direct_Inference accepts a Result struct with .keypoints/.descriptors vectors.
        spRes1,
        spRes2
    );

    // 3) Build 3D-2D correspondences using kp_to_mpid from frame1
    std::vector<cv::Point3f> pts3d; pts3d.reserve(spRes2.numValid);
    std::vector<cv::Point2f> pts2d; pts2d.reserve(spRes2.numValid);

    const auto& kf1 = map.keyframes[frame1.id];
    const float score_thr = 0.7f;

    // For each keypoint in frame1 that matched to frame2, if it has a mapped 3D point, use it
    int used = 0;
    int count = 0;
    for (int i = 0, n1 = (int)kf1.keypoints.size()/2; i < n1; ++i) {
        int j = lgRes12.matches0[i];
        if (j < 0 || lgRes12.mscores0[i] <= score_thr) continue;

        int mpid = -1;
        if (i < (int)kf1.kp_to_mpid.size()) mpid = kf1.kp_to_mpid[i];
        if (mpid < 0) {
            count++;
            continue;
         } // no 3D point for this 2D kp in frame1

        auto mp_it = map.map_points.find(mpid);
        if (mp_it == map.map_points.end()) continue;

        const MapPoint& mp = mp_it->second;
        pts3d.emplace_back(mp.position); // world coords

        // 2D point in frame2 (temp5) from LightGlue keypoints1
        float x2 = (float)lgRes12.keypoints1[2 * j];
        float y2 = (float)lgRes12.keypoints1[2 * j + 1];
        pts2d.emplace_back(x2, y2);
        ++used;
    }

    std::cout << "PnP correspondences: " << used << " (3D-2D)" << count << std::endl;
    if ((int)pts3d.size() < 4) {
        std::cerr << "Not enough correspondences for PnP." << std::endl;
        // You can return or continue with a fallback here
        return 1;
    }

    // 4) Run PnP (world → camera)
    cv::Mat distCoeffs = cv::Mat::zeros(4,1,CV_64F); // replace with real distortion if you have it
    cv::Mat rvec2, tvec2;
    std::vector<int> inliers_pnp;
    bool ok_pnp = cv::solvePnPRansac(
        pts3d, pts2d,
        cameraMatrix, distCoeffs,
        rvec2, tvec2,
        false,         // useExtrinsicGuess
        1000,          // iterationsCount
        1.8,           // reprojectionError (px)
        0.999,         // confidence
        inliers_pnp,
        cv::SOLVEPNP_ITERATIVE
    );

    
    if (!ok_pnp || (int)inliers_pnp.size() < 4) {
        std::cerr << "PnP failed or insufficient inliers: " << inliers_pnp.size() << std::endl;
        return 1;
    }

    // Convert to rotation matrix
    cv::Mat R2;
    cv::Rodrigues(rvec2, R2);
    cv::Mat t2 = tvec2.clone();

    // OpenCV PnP gives world→camera: x_cam = R2*X_world + t2
    // Store frame2 pose
    Frame frame2;
    frame2.id = map.next_keyframe_id++;
    frame2.img = img2;
    R2 = R2.t();
    t2 = -R2 * t2;

    // Scale translation to GT magnitude (compare against T_w1)
    cv::Mat T_w2 = gtPoses[2];
    cv::Mat t_gt2 = T_w2(cv::Rect(3,0,1,3)).clone();
    double t_gt_mag2 = cv::norm(t_gt2);
    // t2 *= (t_gt_mag2 / cv::norm(t2));

    frame2.R = R2.clone();
    frame2.t = t2.clone();
    frame2.keypoints = std::move(spRes2.keypoints);
    frame2.descriptors = std::move(spRes2.descriptors);
    frame2.is_keyframe = false; // mark as you wish

    // Add to map
    map.keyframes[frame2.id] = std::move(frame2);

    std::cout << "Frame2 PnP inliers: " << inliers_pnp.size() << std::endl;
    std::cout << "Frame2 R (world->cam):\n" << R2 << std::endl;
    std::cout << "Frame2 t (world->cam):\n" << t2 << std::endl;

    // 5) Compare with ground truth pose for frame index 2 (third image)
    if (gtPoses.size() > 2) {
        cv::Mat R_gt2 = T_w2(cv::Rect(0,0,3,3)).clone();

        double rot_err_deg_2 = rotationAngleErrorDeg(R2, R_gt2);
        double t_dir_err_deg_2 = angleBetweenVectorsDeg(t2, t_gt2);
        double t_mag_err_2 = std::abs(cv::norm(t2) - cv::norm(t_gt2));

        std::cout << "Frame2 rotation error (deg): " << rot_err_deg_2 << std::endl;
        std::cout << "Frame2 translation direction error (deg): " << t_dir_err_deg_2 << std::endl;
        std::cout << "Frame2 translation magnitude error (m): " << t_mag_err_2 << std::endl;
    } else {
        std::cerr << "Ground-truth does not contain pose for frame 2." << std::endl;
    }

    // // Visualize inliers on the second image (frame1)
    // cv::Mat img1_color;
    // cv::cvtColor(img1, img1_color, cv::COLOR_GRAY2BGR);
    // const auto& kf1 = map.keyframes[frame1.id];
    // for (const auto& pr : filteredPairs) {
    //     float x = (float)kf1.keypoints[2 * pr.idx1];
    //     float y = (float)kf1.keypoints[2 * pr.idx1 + 1];
    //     cv::circle(img1_color, cv::Point2f(x, y), 1, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    // }
    // cv::imshow("Inliers on Second Image", img1_color);
    // cv::waitKey(0);

    return 0;
}
