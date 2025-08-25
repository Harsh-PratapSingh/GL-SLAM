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

#include <chrono> // for time measurement

// NEW: Compact record carrying original SP indices and 2D locations
// struct Match2D2D {
//     int idx0;         // SuperPoint index in frame0
//     int idx1;         // SuperPoint index in frame1
//     cv::Point2f p0;   // 2D point in frame0
//     cv::Point2f p1;   // 2D point in frame1
// };

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
    auto cameraMatrix = slam_core::load_camera_matrix(calibPath);
    std::cout << "Camera Matrix:\n" << cameraMatrix << std::endl;

    std::string posesPath = "/home/tomato/Downloads/data_odometry_gray/dataset/sequences/00/00.txt";
    auto gtPoses = slam_core::load_poses(posesPath);

    cv::Mat img0 = cv::imread("temp3.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img1 = cv::imread("temp4.png", cv::IMREAD_GRAYSCALE);
    if (img0.empty() || img1.empty()) {
        throw std::runtime_error("Failed to load image0.png or image1.png");
    }

    auto spRes0 = sp.runInference(img0, img0.rows, img0.cols);
    auto spRes1 = sp.runInference(img1, img1.rows, img1.cols);
    std::cout << "Image0: valid keypoints = " << spRes0.numValid << std::endl;
    std::cout << "Image1: valid keypoints = " << spRes1.numValid << std::endl;

    auto lgRes = lg.run_Direct_Inference(spRes0, spRes1);
    int nMatches = 0;
    const float match_thr = 0.7f;

    // Build compact matches directly
    auto matches = slam_core::lightglue_score_filter(lgRes, match_thr);
    if (matches.size() < 8) {
        std::cerr << "Not enough matches for pose estimation." << std::endl;
        return 1;
    }

    auto [R, t, mask] = slam_core::pose_estimator(matches, cameraMatrix);

    auto inliersPairs = slam_core::pose_estimator_mask_filter(matches, mask);

    // R = R.t();
    // t = -R * t;
    // Scale translation to GT magnitude (compare against T_w1)
    t = slam_core::adjust_translation_magnitude(gtPoses, t, 1);

    cv::Mat R1 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t1 = cv::Mat::zeros(3, 1, CV_64F);
    auto [points3d, filteredPairs] = slam_core::triangulate_and_filter_3d_points(R1, t1, R, t, cameraMatrix, inliersPairs, 100.0, 0.5 );
    Map map;
    slam_core::update_map_and_keyframe_data(map, img1, R, t, spRes1, points3d,
                                            filteredPairs, spRes0, img0, true, true);


    // ===================== PnP on next image (temp5.png) =====================
    cv::Mat img2 = cv::imread("temp5.png", cv::IMREAD_GRAYSCALE);
    if (img2.empty()) {
        throw std::runtime_error("Failed to load temp5.png");
    }

    // Example timing block around LightGlue
    auto start = std::chrono::high_resolution_clock::now();

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


    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "LightGlue inference took " << duration_ms << " ms" << std::endl;

    // 3) Build 3D-2D correspondences using kp_to_mpid from frame1
    std::vector<cv::Point3f> pts3d; pts3d.reserve(spRes2.numValid);
    std::vector<cv::Point2f> pts2d; pts2d.reserve(spRes2.numValid);

    const auto& kf1 = map.keyframes[1];
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
    frame2.keypoints = spRes2.keypoints;
    frame2.descriptors = spRes2.descriptors;
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

    // Visualize inliers on the second image (frame1)
    // cv::Mat img1_color;
    // cv::cvtColor(img1, img1_color, cv::COLOR_GRAY2BGR);
    // const auto& kf2 = map.keyframes[frame1.id];
    // for (const auto& pr : filteredPairs) {
    //     float x = (float)kf2.keypoints[2 * pr.idx1];
    //     float y = (float)kf2.keypoints[2 * pr.idx1 + 1];
    //     cv::circle(img1_color, cv::Point2f(x, y), 1, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    // }
    // cv::imshow("Inliers on Second Image", img1_color);
    // cv::waitKey(0);


    // ---------------- TRIANGULATE ALL FRAME0 KEYPOINTS THAT HAVE A MATCH ----------------
    // We already have matches and mask from E/pose estimation.
    // If you truly want "all" keypoints of frame0, we will attempt triangulation for every kp
    // that has a LightGlue match in frame1 (even if it was not an inlier of the essential matrix).
    // However, for stability we will prefer using inliers. If you want strictly all matched,
    // replace 'inliersPairs' with 'matches' and skip the mask filtering above.

    // To satisfy instruction (1), we’ll triangulate for every valid LightGlue match (not only inliers):
    std::vector<cv::Point2f> allPts0, allPts1;
    std::vector<int> allIdx0, allIdx1;
    allPts0.reserve(matches.size());
    allPts1.reserve(matches.size());
    allIdx0.reserve(matches.size());
    allIdx1.reserve(matches.size());
    for (const auto& m : matches) {
        allPts0.push_back(m.p0);
        allPts1.push_back(m.p1);
        allIdx0.push_back(m.idx0);
        allIdx1.push_back(m.idx1);
    }

    // Use the same P0, P1 from above (P0 = K[I|0], P1 = K[R|t] with R,t from recoverPose before your R=t() invert)
    // We have already built P0 and P1 earlier prior to inlier triangulation.
    // Rebuild quickly here to be explicit:
    {
        cv::Mat I3 = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat Rt2(3,4,CV_64F), PP0(3,4,CV_64F), PP1(3,4,CV_64F);
        I3.copyTo(PP0.colRange(0,3));            // [I|0]
        PP0.col(3) = cv::Mat::zeros(3,1,CV_64F);
        R = R.t();
        t = -R * t;
        R.copyTo(Rt2.colRange(0,3));              // R from recoverPose (cam0->cam1)
        t.copyTo(Rt2.col(3));                     // t from recoverPose (cam0->cam1)
        PP0 = cameraMatrix * PP0;
        PP1 = cameraMatrix * Rt2;

        cv::Mat X4_all;
        cv::triangulatePoints(PP0, PP1, allPts0, allPts1, X4_all);
        X4_all.convertTo(X4_all, CV_64FC1);

        // Build triangulated 3D points for every matched kp in frame0 (where valid)
        struct TriRec { cv::Point3d X; int idx0; int idx1; };
        std::vector<TriRec> triRecords;
        triRecords.reserve(X4_all.cols);

        for (int i = 0; i < X4_all.cols; ++i) {
            double w = X4_all.at<double>(3, i);
            if (std::abs(w) < 1e-9) continue;
            double X = X4_all.at<double>(0, i) / w;
            double Y = X4_all.at<double>(1, i) / w;
            double Z = X4_all.at<double>(2, i) / w;
            // simple cheirality w.r.t cam0 and cam1 (positive depth in both):
            // For cam0: depth is Z0 = [I|0] -> just Z
            // For cam1: depth Z1 ~ third component of R*X + t; approximate via PP1 third row
            cv::Mat Xw = (cv::Mat_<double>(4,1) << X, Y, Z, 1.0);
            double z0 = Z;
            cv::Mat row2 = PP1.row(2);  // Extract row as cv::Mat
            cv::Mat res = row2 * Xw;    // Perform multiplication
            double z1 = res.at<double>(0, 0);  // Access the element            
            if (z0 <= 0.0 || z1 <= 0.0 || Z > 1e5) continue; // clamp far points

            triRecords.push_back({ cv::Point3d(X, Y, Z), allIdx0[i], allIdx1[i] });
        }

        std::cout << "Triangulated (all matched from frame0) valid count: " << triRecords.size() << std::endl;

        // ---------------- BUILD SYNTHETIC FRAME S (1241x376) AT SAME POSE AS FRAME1 ----------------
        // Synthetic frame S: same pose as frame1 in your final convention.
        // Your final frame1 pose is camera1->world inverted (you flipped R,t below).
        // For projection we need world->cam of frame1. We can use the "cam0->cam1" pose (R,t) before inversion.
        // We'll compute projection with the cam1 (world->cam) from the recoverPose stage: x1 ~ K [R|t] X0
        // Since our X are in the cam0/world implied by P0=K[I|0], we can project with PP1 (camera 1).
        // Synthetic image size:
        const int synthW = 1241, synthH = 376;

        // Prepare synthetic "SuperPoint-like" container with projected keypoints and descriptors from frame0
        SuperPointTRT::Result spResSynth;
        spResSynth.keypoints.clear();
        spResSynth.descriptors.clear();

        spResSynth.keypoints.reserve(triRecords.size() * 2);
        spResSynth.descriptors.reserve(triRecords.size() * 256);

        // Project each triangulated 3D point into synthetic frame S using PP1 (K[R|t])
        int kept = 0;
        for (const auto& rec : triRecords) {
            cv::Mat Xw_(4,1,CV_64F);
            Xw_.at<double>(0)=rec.X.x; Xw_.at<double>(1)=rec.X.y; Xw_.at<double>(2)=rec.X.z; Xw_.at<double>(3)=1.0;
            cv::Mat proj = P1 * Xw_; // P1 was earlier K[R|t], ensure it's consistent with recoverPose stage
            double u = proj.at<double>(0) / proj.at<double>(2);
            double v = proj.at<double>(1) / proj.at<double>(2);
            if (proj.at<double>(2) <= 0.0) continue;
            if (u < 0 || u >= synthW || v < 0 || v >= synthH) continue;

            // push keypoint (x,y) into synthetic keypoint array
            spResSynth.keypoints.push_back((float)u);
            spResSynth.keypoints.push_back((float)v);

            // copy descriptor from frame0 for idx0 (most recent descriptor could also be from frame1)
            // frame0.descriptors is a flat array (N0 * 256). Use rec.idx0 to index
            const float* d0 = &spRes0.descriptors[rec.idx0 * 256];
            spResSynth.descriptors.insert(spResSynth.descriptors.end(), d0, d0 + 256);
            ++kept;
        }
        spResSynth.numValid = kept;
        std::cout << "Synthetic projected points kept in-view: " << spResSynth.numValid << std::endl;

        // ---------------- RUN LIGHTGLUE: synthetic S vs real current frame1 ----------------
        // As requested: "run lightglue with those keypoints on this image and descriptors"
        // We'll match the synthetic frame S to the real frame1 SuperPoint result spRes1.
        auto lg_start = std::chrono::high_resolution_clock::now();
        LightGlueTRT::Result lgRes_S_1 = lg.run_Direct_Inference(spResSynth, spRes0);
        auto lg_end = std::chrono::high_resolution_clock::now();
        auto lg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(lg_end - lg_start).count();
        std::cout << "LightGlue (Synthetic vs frame1) took " << lg_ms << " ms, matches=";

        int mcountS = 0;
        for (int i = 0; i < spResSynth.numValid; ++i) {
            if (lgRes_S_1.matches0[i] >= 0 && lgRes_S_1.mscores0[i] > match_thr) ++mcountS;
        }
        std::cout << mcountS << std::endl;

        // If you want to use these matches for something (e.g., more correspondences for BA or validation),
        // you can extract them similarly as earlier. Example collect matched (S -> frame1):
        std::vector<cv::Point2f> synthPts, realPts1;
        synthPts.reserve(mcountS);
        realPts1.reserve(mcountS);
        for (int i = 0; i < spResSynth.numValid; ++i) {
            int j = lgRes_S_1.matches0[i];
            if (j >= 0 && lgRes_S_1.mscores0[i] > match_thr) {
                float xs = (float)lgRes_S_1.keypoints0[2*i];
                float ys = (float)lgRes_S_1.keypoints0[2*i + 1];
                float xr = (float)lgRes_S_1.keypoints1[2*j];
                float yr = (float)lgRes_S_1.keypoints1[2*j + 1];
                synthPts.emplace_back(xs, ys);
                realPts1.emplace_back(xr, yr);
            }
        }
        std::cout << "Synthetic vs frame1 good matches (score>" << match_thr << "): " << synthPts.size() << std::endl;
    }
    // ---------------- END synthetic matching block ----------------


    return 0;
}
