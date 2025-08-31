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
const float match_thr = 0.7f;
const float map_match_thr = 0.7f;
const int map_match_window = 5;
const float mag_filter = 0.00f;
int max_idx   = 1;           // max 4540

Map map;
std::mutex map_mutex;  // To synchronize map access
std::string img_dir_path = "/home/tomato/Downloads/data_odometry_gray/dataset/sequences/00/image_0/";
std::string calibPath = "/home/tomato/Downloads/data_odometry_gray/dataset/sequences/00/calib.txt";
std::string posesPath = "/home/tomato/Downloads/data_odometry_gray/dataset/sequences/00/00.txt";

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


const int ba_window_size = 2;  // Adjust based on performance; e.g., 5-15


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
    auto inliersPairs = slam_core::pose_estimator_mask_filter(matches, mask);// std::vector<int> mapid;
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
    std::thread viewer_thread(slam_visualization::visualize_map_loop, std::ref(map), std::ref(map_mutex));

    OptimizedBAData opt_data;
    if (map.next_keyframe_id >= ba_window_size) {
        opt_data = slam_core::perform_local_ba(map, cameraMatrix, ba_window_size, map.next_keyframe_id - 1);
    }

    // // Lock only for update
    if (!opt_data.optimized_poses.empty()) {
        std::lock_guard<std::mutex> lock(map_mutex);
        for (const auto& [kfid, Rt] : opt_data.optimized_poses) {
            if (map.keyframes.find(kfid) != map.keyframes.end()) {
                map.keyframes[kfid].R = Rt.colRange(0, 3).clone();
                map.keyframes[kfid].t = Rt.col(3).clone();
            }
        }
        for (const auto& [pid, pos] : opt_data.optimized_points) {
            if (map.map_points.find(pid) != map.map_points.end()) {
                map.map_points[pid].position = pos;
            }
        }
    }
    //std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    // ===================== PnP on next image (temp5.png) =====================

    
    const int start_idx = map.next_keyframe_id;            
    
    // auto img_name = [](int idx) {
    //         char buf[32];
    //         std::snprintf(buf, sizeof(buf), "%06d.png", idx);
    //         return std::string(buf);
    //     };

    for (int idx = start_idx; idx <= max_idx; ++idx) {
        const int prev_kfid = map.next_keyframe_id - 1; 
   
        auto [img_cur, R_cur, t_cur, spRes_cur, restPairs, map_point_id,
                kp_index, skip] = slam_core::run_pnp(map, sp, lg, img_dir_path,
                cameraMatrix, match_thr, map_match_thr, idx, map_match_window, false, gtPoses);
        if(skip) continue;

        // R_cur = R_cur.t(); t_cur = -R_cur * t_cur;
        //slam_core::refine_pose_with_g2o(R_cur, t_cur, spRes_cur, map_point_id, kp_index, map, cameraMatrix);
        // R_cur = R_cur.t(); t_cur = -R_cur * t_cur;
        // t_cur = slam_core::adjust_translation_magnitude(gtPoses, t_cur, idx );

        
        double t_mag = std::abs(cv::norm(map.keyframes[prev_kfid].t) - cv::norm(t_cur));
        if(t_mag < mag_filter) continue;

        cv::Mat Rc = R_cur.clone(); cv::Mat tc = t_cur.clone();
        Rc = Rc.t();
        tc = -Rc * tc;

        // Compare with GT
        if (gtPoses.size() > idx) {
            const cv::Mat T_wi = gtPoses[idx];
            cv::Mat R_gt = T_wi(cv::Rect(0,0,3,3)).clone();
            cv::Mat t_gt = T_wi(cv::Rect(3,0,1,3)).clone();

            double rot_err = rotationAngleErrorDeg(Rc, R_gt);
            double t_dir_err = angleBetweenVectorsDeg(tc, t_gt);
            double t_mag_err = std::abs(cv::norm(tc) - cv::norm(t_gt));
            std::cout << "[PnP-Loop] Frame " << idx << " | rot(deg): " << rot_err
                    << " t_dir(deg): " << t_dir_err << " t_mag(m): " << t_mag_err << "\n";
        }

        cv::Mat R_prev = map.keyframes[prev_kfid].R.clone(); cv::Mat t_prev = map.keyframes[prev_kfid].t.clone();
        R_prev = R_prev.t(); t_prev = -R_prev * t_prev;
        // R_cur = R_cur.t(); t_cur = -R_cur * t_cur;

        auto [newPoints3D, newPairs] = slam_core::triangulate_and_filter_3d_points(
            R_prev, t_prev, R_cur, t_cur, cameraMatrix, restPairs,
            /*maxZ*/ 100.0, /*min_repoj_error*/ 0.1);
        std::cout << "[PnP-Loop] Frame " << idx << " triangulated-new = " << newPoints3D.size() << "\n";

        {
            std::lock_guard<std::mutex> lock(map_mutex);
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
        }
        
        // In your loop, after run_pnp and skip/mag_filter checks
        // Perform BA outside lock
        OptimizedBAData opt_data;
        if (map.next_keyframe_id >= ba_window_size) {
            opt_data = slam_core::perform_local_ba(map, cameraMatrix, ba_window_size, map.next_keyframe_id - 1);
        }

        // Lock only for update
        if (!opt_data.optimized_poses.empty()) {
            std::lock_guard<std::mutex> lock(map_mutex);
            for (const auto& [kfid, Rt] : opt_data.optimized_poses) {
                if (map.keyframes.find(kfid) != map.keyframes.end()) {
                    map.keyframes[kfid].R = Rt.colRange(0, 3).clone();
                    map.keyframes[kfid].t = Rt.col(3).clone();
                }
            }
            for (const auto& [pid, pos] : opt_data.optimized_points) {
                if (map.map_points.find(pid) != map.map_points.end()) {
                    map.map_points[pid].position = pos;
                }
            }
        }

        
        // {   
        //     std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        //     // std::lock_guard<std::mutex> lock(map_mutex);
        //     // map.keyframes[map.next_keyframe_id-1].R = R_cur;
        //     // map.keyframes[map.next_keyframe_id-1].t = t_cur;
        // }
       


        //visualize obs(green) and new points(blue)
        cv::Mat img1_color;
        cv::cvtColor(img_cur, img1_color, cv::COLOR_GRAY2BGR);
        const auto& kf2 = map.keyframes[map.next_keyframe_id - 1].sp_res;

        for (const auto& pr : newPairs) {
            float x = (float)kf2.keypoints[2 * pr.idx1];
            float y = (float)kf2.keypoints[2 * pr.idx1 + 1];
            cv::circle(img1_color, cv::Point2f(x, y), 2, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
        }

        for( auto i : kp_index){
            float x = (float)kf2.keypoints[2 * i];
            float y = (float)kf2.keypoints[2 * i + 1];
            cv::circle(img1_color, cv::Point2f(x, y), 2, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
        }
        
        cv::imshow("Inliers on Second Image", img1_color);
        cv::waitKey(1);
    }
    
    // After the main loop, analyze map points
    {
        // std::lock_guard<std::mutex> lock(map_mutex);  // Ensure thread-safe access

        int count_low_error_high_obs = 0;
        int count_low_error_low_obs = 0;
        int count_high_error_high_obs = 0;
        int count_high_error_low_obs = 0;

        const double error_threshold = 3.0;  // Pixels
        const int obs_threshold = 3;

        for (const auto& [point_id, point] : map.map_points) {
            if (point.obs.empty() || point.is_bad) continue;

            // Count observations
            int num_obs = point.obs.size();

            // Compute average reprojection error
            double total_error = 0.0;
            int valid_obs = 0;
            cv::Mat position_mat = (cv::Mat_<double>(3,1) << point.position.x, point.position.y, point.position.z);

            for (const auto& obs : point.obs) {
                int kfid = obs.keyframe_id;
                if (map.keyframes.find(kfid) == map.keyframes.end()) continue;

                const auto& kf = map.keyframes.at(kfid);
                
                cv::Mat R1 = kf.R.clone();
                cv::Mat t1 = kf.t.clone();
                R1 = R1.t();
                t1 = -R1 * t1;
                // Project 3D point to camera coordinates (assuming R/t are camera-to-world)
                cv::Mat camera_point = R1 * position_mat + t1;
                if (camera_point.at<double>(2) <= 0) continue;  // Behind camera

                // Normalize
                double z = camera_point.at<double>(2);
                cv::Mat normalized = (cv::Mat_<double>(3,1) << camera_point.at<double>(0)/z, camera_point.at<double>(1)/z, 1.0);

                // Apply camera matrix for pixel coordinates
                cv::Mat projected_mat = cameraMatrix * normalized;
                cv::Point2d projected(projected_mat.at<double>(0), projected_mat.at<double>(1));

                // Reprojection error
                double error = cv::norm(projected - obs.point2D);
                
                //std::cout << "Error " << valid_obs << " : " << error << std::endl;
                total_error += error;
                valid_obs++;
            }

            if (valid_obs == 0) continue;

            double avg_error = total_error / valid_obs;

            // Categorize
            bool is_low_error = (avg_error < error_threshold);
            bool is_high_obs = (num_obs >= obs_threshold);

            if (is_low_error && is_high_obs) {
                count_low_error_high_obs++;
            } else if (is_low_error && !is_high_obs) {
                count_low_error_low_obs++;
            } else if (!is_low_error && is_high_obs) {
                count_high_error_high_obs++;
            } else {
                count_high_error_low_obs++;
            }
        }

        // Print counts
        std::cout << "Map Point Analysis:" << std::endl;
        std::cout << " - Low error (< " << error_threshold << " px) + High obs (>= " << obs_threshold << "): " << count_low_error_high_obs << std::endl;
        std::cout << " - Low error (< " << error_threshold << " px) + Low obs (< " << obs_threshold << "): " << count_low_error_low_obs << std::endl;
        std::cout << " - High error (>= " << error_threshold << " px) + High obs (>= " << obs_threshold << "): " << count_high_error_high_obs << std::endl;
        std::cout << " - High error (>= " << error_threshold << " px) + Low obs (< " << obs_threshold << "): " << count_high_error_low_obs << std::endl;
        std::cout << "Total map points analyzed: " << map.map_points.size() << std::endl;

        std::cout << "Frame 0 :-" << std::endl;
        std::cout << " R : " << map.keyframes[0].R << std::endl;
        std::cout << " t : " << map.keyframes[0].t << std::endl;
        std::cout << "Frame 1 :-" << std::endl;
        std::cout << " R : " << map.keyframes[1].R << std::endl;
        std::cout << " t : " << map.keyframes[1].t << std::endl;
    }


    viewer_thread.join();

    return 0;
}
