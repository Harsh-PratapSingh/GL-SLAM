#include "core/lightglue.h"
#include "core/superpoint.h"
#include "core/slam_core.h"
#include "visualization/visualization.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>


#include <chrono> // for time measurement

#include <pangolin/pangolin.h>
#include <thread>
#include <mutex>

#include <unordered_set>
#include <set>

constexpr int SYN_W = 1241;  // KITTI gray left image_0 width
constexpr int SYN_H = 376;   // KITTI gray left image_0 height





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


Map map;
std::mutex map_mutex;  // To synchronize map access
const float match_thr = 0.7f;
std::string img_dir_path = "/home/tomato/Downloads/data_odometry_gray/dataset/sequences/00/image_0/";
std::string calibPath = "/home/tomato/Downloads/data_odometry_gray/dataset/sequences/00/calib.txt";
std::string posesPath = "/home/tomato/Downloads/data_odometry_gray/dataset/sequences/00/00.txt";
    


int main() {
    SuperPointTRT sp;
    LightGlueTRT lg;
    slam_core::superpoint_lightglue_init(sp, lg);

    auto cameraMatrix = slam_core::load_camera_matrix(calibPath);
    auto gtPoses = slam_core::load_poses(posesPath);
    cv::Mat img0 = cv::imread(img_dir_path + "000000.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img1 = cv::imread(img_dir_path + "000001.png", cv::IMREAD_GRAYSCALE);
    auto spRes0 = sp.runInference(img0, img0.rows, img0.cols);
    auto spRes1 = sp.runInference(img1, img1.rows, img1.cols);
    auto lgRes = lg.run_Direct_Inference(spRes0, spRes1);
    auto matches = slam_core::lightglue_score_filter(lgRes, match_thr);
    if (matches.size() < 8) {
        std::cerr << "Not enough matches for pose estimation." << std::endl;
        return 1;
    }
    auto [R, t, mask] = slam_core::pose_estimator(matches, cameraMatrix);
    auto inliersPairs = slam_core::pose_estimator_mask_filter(matches, mask);
    // R = R.t();
    // t = -R * t;
    t = slam_core::adjust_translation_magnitude(gtPoses, t, 1);

    cv::Mat R1 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t1 = cv::Mat::zeros(3, 1, CV_64F);
    auto [points3d, filteredPairs] = slam_core::triangulate_and_filter_3d_points(R1, t1, R, t, cameraMatrix, inliersPairs, 100.0, 0.5 );
    
    std::vector<int> a;
    slam_core::update_map_and_keyframe_data(map, img1, R, t, spRes1, points3d,
                                            filteredPairs, spRes0, img0, a, a, true, true);

    std::cout << "Camera Matrix:\n" << cameraMatrix << std::endl;
    std::cout << "Image0: valid keypoints = " << spRes0.numValid << std::endl;
    std::cout << "Image1: valid keypoints = " << spRes1.numValid << std::endl;

    // After initial map update (bootstrap)
    std::thread viewer_thread = slam_visualization::start_viewer(map, map_mutex);

    // ===================== PnP on next image (temp5.png) =====================

    int prev_kfid = map.next_keyframe_id - 1; 
    int start_idx = map.next_keyframe_id;            
    int max_idx   = 4540;           // max 4540

    auto img_name = [](int idx) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%06d.png", idx);
            return std::string(buf);
        };

    for (int idx = start_idx; idx <= max_idx; ++idx) {

        auto [img_cur, R_cur, t_cur, spRes_cur, restPairs, map_point_id,
            kp_index, skip] = slam_core::run_pnp(map, sp, lg, img_dir_path,
            cameraMatrix, match_thr, 0.5, idx, 6);

        // Compare with GT
        if ((int)gtPoses.size() > idx) {
            const cv::Mat T_wi = gtPoses[idx];
            cv::Mat R_gt = T_wi(cv::Rect(0,0,3,3)).clone();
            cv::Mat t_gt = T_wi(cv::Rect(3,0,1,3)).clone();

            double rot_err = rotationAngleErrorDeg(R_cur, R_gt);
            double t_dir_err = angleBetweenVectorsDeg(t_cur, t_gt);
            double t_mag_err = std::abs(cv::norm(t_cur) - cv::norm(t_gt));
            std::cout << "[PnP-Loop] Frame " << idx << " | rot(deg): " << rot_err
                    << " t_dir(deg): " << t_dir_err << " t_mag(m): " << t_mag_err << "\n";
        }

        


        // 8) Triangulate-and-filter the “rest” using your helper (world->cam convention)
        //    Pprev = K[R_prev|t_prev], Pcur = K[R_cur|t_cur]
        cv::Mat R_prev = map.keyframes[prev_kfid].R;
        cv::Mat t_prev = map.keyframes[prev_kfid].t;
        R_prev = R_prev.t();
        t_prev = -R_prev * t_prev;
        R_cur = R_cur.t();
        t_cur = -R_cur * t_cur;
        


        auto [newPoints3D, newPairs] =
            slam_core::triangulate_and_filter_3d_points(R_prev, t_prev, R_cur, t_cur,
                                                        cameraMatrix, restPairs,
                                                        /*maxZ*/ 100.0, /*minCosParallax*/ 0.1);


        std::cout << "[PnP-Loop] Frame " << idx << " triangulated-new = " << newPoints3D.size() << "\n";


        // 9) Update map & keyframes via your helper (writes sp_res, kp_to_mpid, observations, etc.)
        //    Note: pass prev’s sp_res and the new current sp_res so the helper can wire indices correctly.
        std::lock_guard<std::mutex> lk(map_mutex);
        slam_core::update_map_and_keyframe_data(
            map,
            /*img_cur*/ img_cur,
            /*R_cur*/   R_cur,
            /*t_cur*/   t_cur,
            /*spRes_cur*/ spRes_cur,
            /*points3d*/ newPoints3D,
            /*pairs*/    newPairs,
            /*spRes_prev*/ spRes_cur,
            /*img_prev*/  img_cur,   // ensure kf_prev.img was stored at bootstrap
            /*map_point_obs_id*/ map_point_id,      //exp
            /*obs_kp_index*/ kp_index,
            /*is_first_frame*/ false,
            /*is_cur_kf*/  true
        );


        // 10) Advance: current frame inserted with the next id; set prev_kfid to last inserted
        // If your helper increments ids internally, the last inserted keyframe id should be map.next_keyframe_id - 1
        prev_kfid = map.next_keyframe_id - 1;
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

    viewer_thread.join();




    return 0;
}
