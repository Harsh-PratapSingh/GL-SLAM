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
const int map_match_window = 20;
const int Full_ba_window_size = 15;
const int Full_ba_include_past_optimized_frame_size = 5;
const float mag_filter = 1.0f;
const float rot_filter = 1.0f;
int max_idx   = 4540;           // max 4540

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


#include <ceres/ceres.h>
#include <ceres/rotation.h>

// ... (rest of your existing includes and code)
int stop = 0;
// Define the reprojection error cost function
// Define the reprojection error cost function
struct ReprojectionError {
    ReprojectionError(const cv::Point2d& observed, const cv::Mat& camera_matrix)
        : observed_(observed), camera_matrix_(camera_matrix) {}

    template <typename T>
    bool operator()(const T* const camera,  // 6 params: angle-axis rotation + translation
                    const T* const point,   // 3 params: 3D point
                    T* residuals) const {
        // Camera params: camera[0,1,2] = angle-axis rotation, camera[3,4,5] = translation
        // Assuming pose is camera-to-world: invert to get world-to-camera
        T p_trans[3];
        p_trans[0] = point[0] - camera[3];
        p_trans[1] = point[1] - camera[4];
        p_trans[2] = point[2] - camera[5];

        // Apply inverse rotation (transpose, i.e., rotate by -angle_axis)
        T minus_camera[3] = { -camera[0], -camera[1], -camera[2] };
        T p[3];
        ceres::AngleAxisRotatePoint(minus_camera, p_trans, p);

        // Project to normalized image coordinates
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        // Apply camera matrix (fx, fy, cx, cy; assuming no skew or distortion)
        T fx = T(camera_matrix_.at<double>(0, 0));
        T fy = T(camera_matrix_.at<double>(1, 1));
        T cx = T(camera_matrix_.at<double>(0, 2));
        T cy = T(camera_matrix_.at<double>(1, 2));

        T predicted_x = fx * xp + cx;
        T predicted_y = fy * yp + cy;

        // Residuals
        residuals[0] = predicted_x - T(observed_.x);
        residuals[1] = predicted_y - T(observed_.y);

        return true;
    }

    static ceres::CostFunction* Create(const cv::Point2d& observed, const cv::Mat& camera_matrix) {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
            new ReprojectionError(observed, camera_matrix));
    }

    cv::Point2d observed_;
    cv::Mat camera_matrix_;
};


// Function to compute average reprojection error (extracted from your existing code)
double ComputeAverageReprojectionError(const Map& map, const cv::Mat& cameraMatrix) {
    double total_error = 0.0;
    int valid_obs = 0;

    for (const auto& [point_id, point] : map.map_points) {
        if (point.obs.empty() || point.is_bad) continue;

        cv::Mat position_mat = (cv::Mat_<double>(3,1) << point.position.x, point.position.y, point.position.z);

        for (const auto& obs : point.obs) {
            int kfid = obs.keyframe_id;
            if (map.keyframes.find(kfid) == map.keyframes.end()) continue;
            const auto& kf = map.keyframes.at(kfid);

            cv::Mat R1 = kf.R.clone();
            cv::Mat t1 = kf.t.clone();
            R1 = R1.t();
            t1 = -R1 * t1;

            cv::Mat camera_point = R1 * position_mat + t1;
            if (camera_point.at<double>(2) <= 0) continue;

            double z = camera_point.at<double>(2);
            cv::Mat normalized = (cv::Mat_<double>(3,1) << camera_point.at<double>(0)/z, camera_point.at<double>(1)/z, 1.0);

            cv::Mat projected_mat = cameraMatrix * normalized;
            cv::Point2d projected(projected_mat.at<double>(0), projected_mat.at<double>(1));

            double error = cv::norm(projected - obs.point2D);
            total_error += error;
            valid_obs++;
        }
    }

    if (valid_obs == 0) return 0.0;
    return total_error / valid_obs;
}

const int ba_window_size = 3;  // Adjust based on performance; e.g., 5-15


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


    //std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    // ===================== PnP on next image (temp5.png) =====================

    
    // const int start_idx = map.next_keyframe_id;     
        const int start_idx = 2;            
       
    
    // auto img_name = [](int idx) {
    //         char buf[32];
    //         std::snprintf(buf, sizeof(buf), "%06d.png", idx);
    //         return std::string(buf);
    //     };
    int prev_triangulated_frame = map.next_keyframe_id -1;
    int run_window = -1;
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

        
        double t_mag = std::abs(cv::norm(map.keyframes[prev_triangulated_frame].t) - cv::norm(t_cur));
        
        
        
        

        cv::Mat Rc = R_cur.clone(); cv::Mat tc = t_cur.clone();
        Rc = Rc.t();
        tc = -Rc * tc;

        double r_deg = rotationAngleErrorDeg(Rc, map.keyframes[prev_triangulated_frame].R);
        std::cout << "R_DEG = " << r_deg << std::endl;

        if(t_mag < mag_filter && r_deg < rot_filter) skip = true;
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

        bool run_ba = false;
        int window = 0;
        
        if(skip){
            newPoints3D.clear();
            newPairs.clear();
            if(map.next_keyframe_id - prev_triangulated_frame > Full_ba_window_size + 10  && map.next_keyframe_id - run_window > Full_ba_window_size ){
                run_ba = true;
                window = map.next_keyframe_id - run_window; 
                // run_window = map.next_keyframe_id;
            }
        }
        else{
            if(map.next_keyframe_id - run_window >= Full_ba_window_size){
                run_ba = true;
                window = map.next_keyframe_id - run_window; 
                // run_window = map.next_keyframe_id;
            }
            prev_triangulated_frame = map.next_keyframe_id;
        }

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
        
        // In main(), after the for loop (before the existing map point analysis)
        
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
        std::cout << map.keyframes.size() << std::endl;

        if(run_ba){
            int ba_window = 0;
            if(map.keyframes.size() >= (Full_ba_include_past_optimized_frame_size + window)){
                ba_window = Full_ba_include_past_optimized_frame_size + window;
            }
            else ba_window = window;
            auto done = slam_core::full_ba(map_mutex, map, cameraMatrix, ba_window);
            if(done) run_window = map.next_keyframe_id - 1;
            // prev_triangulated_frame = map.next_keyframe_id;
        }

        if (gtPoses.size() > idx) {
            const cv::Mat T_wi = gtPoses[idx];
            cv::Mat R_gt = T_wi(cv::Rect(0,0,3,3)).clone();
            cv::Mat t_gt = T_wi(cv::Rect(3,0,1,3)).clone();

            double rot_err = rotationAngleErrorDeg(map.keyframes[map.next_keyframe_id-1].R, R_gt);
            double t_dir_err = angleBetweenVectorsDeg(map.keyframes[map.next_keyframe_id-1].t, t_gt);
            double t_mag_err = std::abs(cv::norm(map.keyframes[map.next_keyframe_id-1].t) - cv::norm(t_gt));
            std::cout << "[PnP-Loop] Frame " << idx << " | rot(deg): " << rot_err
                    << " t_dir(deg): " << t_dir_err << " t_mag(m): " << t_mag_err << "\n";
        }
        
    }

    
    // {

    //     // Compute average reprojection error before BA
    //     double avg_error_before = ComputeAverageReprojectionError(map, cameraMatrix);
    //     std::cout << "Average Reprojection Error Before BA: " << avg_error_before << " px" << std::endl;

    //     // Prepare parameters for Ceres
    //     // We need arrays for camera parameters (6 per keyframe: angle-axis rot + trans) and points (3 per point)
    //     std::vector<double> camera_params;
    //     std::vector<double> point_params;

    //     // Map from keyframe id to parameter index
    //     std::unordered_map<int, int> kf_to_param_idx;
    //     int cam_param_size = 6;  // angle-axis (3) + translation (3)

    //     // Collect and convert camera poses
    //     std::unique_lock<std::mutex> lock(map_mutex);
    //     for (const auto& [kfid, kf] : map.keyframes) {
    //         kf_to_param_idx[kfid] = camera_params.size() / cam_param_size;

    //         cv::Mat Rr = kf.R;
    //         cv::Mat Tr = kf.t;

    //         // Rr = Rr.t();
    //         // Tr = -Rr * Tr;
    //         // Convert rotation matrix to angle-axis
    //         cv::Mat angle_axis;
    //         cv::Rodrigues(Rr, angle_axis);  // Assuming R is camera-to-world; adjust if needed

    //         camera_params.push_back(angle_axis.at<double>(0));
    //         camera_params.push_back(angle_axis.at<double>(1));
    //         camera_params.push_back(angle_axis.at<double>(2));
    //         camera_params.push_back(Tr.at<double>(0));
    //         camera_params.push_back(Tr.at<double>(1));
    //         camera_params.push_back(Tr.at<double>(2));
    //     }

    //     // Collect points
    //     std::unordered_map<int, int> point_to_param_idx;
    //     int point_param_size = 3;
    //     for (const auto& [point_id, point] : map.map_points) {
    //         if (point.is_bad || point.obs.empty()) continue;
    //         point_to_param_idx[point_id] = point_params.size() / point_param_size;

    //         point_params.push_back(point.position.x);
    //         point_params.push_back(point.position.y);
    //         point_params.push_back(point.position.z);
    //     }

    //     // Build the problem
    //     ceres::Problem problem;

    //     for (const auto& [point_id, point] : map.map_points) {
    //         if (point.is_bad || point.obs.empty()) continue;
    //         int point_idx = point_to_param_idx[point_id];
    //         // if(point.obs.size() < 3) continue;
    //         for (const auto& obs : point.obs) {
    //             int kfid = obs.keyframe_id;
    //             if (map.keyframes.find(kfid) == map.keyframes.end()) continue;

    //             int cam_idx = kf_to_param_idx[kfid];

    //             ceres::CostFunction* cost_function = ReprojectionError::Create(obs.point2D, cameraMatrix);
    //             ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);  // Scale 1.0; adjust based on expected error magnitude (e.g., pixels)
    //             problem.AddResidualBlock(cost_function, loss_function,
    //                                     &camera_params[cam_idx * cam_param_size],
    //                                     &point_params[point_idx * point_param_size]);
    //         }
    //     }

    //     // Fix the first camera to remove gauge freedom
    //     // if (!map.keyframes.empty()) {
    //     //     int first_kf_id = map.keyframes.begin()->first;
    //     //     int first_cam_idx = kf_to_param_idx[first_kf_id];
    //     //     problem.SetParameterBlockConstant(&camera_params[first_cam_idx * cam_param_size]);
    //     // }

    //     {
    //         const int anchor_kf_id = 0; // known bootstrap keyframe
    //         const int cam_param_size = 6;
    //         const int anchor_cam_idx = kf_to_param_idx.at(anchor_kf_id);
    //         problem.SetParameterBlockConstant(&camera_params[anchor_cam_idx * cam_param_size]);
    //         const int anchor_cam_idx2 = kf_to_param_idx.at(1);
    //         problem.SetParameterBlockConstant(&camera_params[anchor_cam_idx2 * cam_param_size]);
    //         // If only translation should be fixed, use SubsetParameterization instead:
    //         // std::vector<int> fixed = {3,4,5};
    //         // auto* subset = new ceres::SubsetParameterization(6, fixed);
    //         // problem.SetParameterization(&camera_params[anchor_cam_idx * cam_param_size], subset);
    //     }
    //     lock.unlock();

    //     // Solve
    //     ceres::Solver::Options options;
    //     options.linear_solver_type = ceres::SPARSE_SCHUR;  // Or SPARSE_SCHUR for larger problems
    //     options.minimizer_progress_to_stdout = true;
    //     options.max_num_iterations = 300; // increase from default ~50
    //     options.num_threads = 8;  // Adjust to your CPU cores (e.g., std::thread::hardware_concurrency())
    //     ceres::Solver::Summary summary;
    //     ceres::Solve(options, &problem, &summary);
    //     std::cout << summary.FullReport() << std::endl;


    //     std::lock_guard<std::mutex> lk(map_mutex);
    //     // Update the map with optimized values
    //     // Update cameras
    //     for (const auto& [kfid, idx] : kf_to_param_idx) {
    //         double* cam = &camera_params[idx * cam_param_size];
    //         // cv::Mat Rr = (cv::Mat_<double>(3,1) << cam[0], cam[1], cam[2]);
    //         // cv::Mat Tr = (cv::Mat_<double>(3,1) << cam[3], cam[4], cam[5]);

    //         // Rr = Rr.t();
    //         // Tr = -Rr * Tr;
            
    //         // cv::Rodrigues(Rr, map.keyframes[kfid].R);
    //         // map.keyframes[kfid].t = Tr;
    //         cv::Mat angle_axis = (cv::Mat_<double>(3,1) << cam[0], cam[1], cam[2]);

    //         cv::Rodrigues(angle_axis, map.keyframes[kfid].R);
    //         map.keyframes[kfid].t = (cv::Mat_<double>(3,1) << cam[3], cam[4], cam[5]);
    //         // map.keyframes[kfid].R = map.keyframes[kfid].R.t();
    //         // map.keyframes[kfid].t = -map.keyframes[kfid].R * map.keyframes[kfid].t;
    //     }

    //     // Update points
    //     for (const auto& [point_id, idx] : point_to_param_idx) {
    //         double* pt = &point_params[idx * point_param_size];
    //         map.map_points[point_id].position = cv::Point3d(pt[0], pt[1], pt[2]);
    //     }
        

    //     // Compute average reprojection error after BA
    //     double avg_error_after = ComputeAverageReprojectionError(map, cameraMatrix);
    //     std::cout << "Average Reprojection Error After BA: " << avg_error_after << " px" << std::endl;
    //     {const cv::Mat T_wi = gtPoses[1];
    //         cv::Mat R_gt = T_wi(cv::Rect(0,0,3,3)).clone();
    //         cv::Mat t_gt = T_wi(cv::Rect(3,0,1,3)).clone();

    //         double rot_err = rotationAngleErrorDeg(map.keyframes[1].R, R_gt);
    //         double t_dir_err = angleBetweenVectorsDeg(map.keyframes[1].t, t_gt);
    //         double t_mag_err = std::abs(cv::norm(map.keyframes[1].t) - cv::norm(t_gt));
    //         std::cout << "[PnP-Loop] Frame " << 1 << " | rot(deg): " << rot_err
    //                 << " t_dir(deg): " << t_dir_err << " t_mag(m): " << t_mag_err << "\n";}
        
    //     // const cv::Mat T_wi = gtPoses[200];
    //     //     cv::Mat R_gt = T_wi(cv::Rect(0,0,3,3)).clone();
    //     //     cv::Mat t_gt = T_wi(cv::Rect(3,0,1,3)).clone();

    //     //     double rot_err = rotationAngleErrorDeg(map.keyframes[200].R, R_gt);
    //     //     double t_dir_err = angleBetweenVectorsDeg(map.keyframes[200].t, t_gt);
    //     //     double t_mag_err = std::abs(cv::norm(map.keyframes[200].t) - cv::norm(t_gt));
    //     //     std::cout << "[PnP-Loop] Frame " << 200 << " | rot(deg): " << rot_err
    //     //             << " t_dir(deg): " << t_dir_err << " t_mag(m): " << t_mag_err << "\n";
    // }
    // // In main(), after the for loop (before the existing map point analysis)
    // {
    //     std::lock_guard<std::mutex> lock(map_mutex);

    //     // Compute average reprojection error before BA
    //     double avg_error_before = ComputeAverageReprojectionError(map, cameraMatrix);
    //     std::cout << "Average Reprojection Error Before BA: " << avg_error_before << " px" << std::endl;

    //     // Prepare parameters for Ceres
    //     // We need arrays for camera parameters (6 per keyframe: angle-axis rot + trans) and points (3 per point)
    //     std::vector<double> camera_params;
    //     std::vector<double> point_params;

    //     // Map from keyframe id to parameter index
    //     std::unordered_map<int, int> kf_to_param_idx;
    //     int cam_param_size = 6;  // angle-axis (3) + translation (3)

    //     // Collect and convert camera poses
    //     for (const auto& [kfid, kf] : map.keyframes) {
    //         kf_to_param_idx[kfid] = camera_params.size() / cam_param_size;

    //         cv::Mat Rr = kf.R;
    //         cv::Mat Tr = kf.t;

    //         Rr = Rr.t();
    //         Tr = -Rr * Tr;
    //         // Convert rotation matrix to angle-axis
    //         cv::Mat angle_axis;
    //         cv::Rodrigues(Rr, angle_axis);  // Assuming R is camera-to-world; adjust if needed

    //         camera_params.push_back(angle_axis.at<double>(0));
    //         camera_params.push_back(angle_axis.at<double>(1));
    //         camera_params.push_back(angle_axis.at<double>(2));
    //         camera_params.push_back(Tr.at<double>(0));
    //         camera_params.push_back(Tr.at<double>(1));
    //         camera_params.push_back(Tr.at<double>(2));
    //     }

    //     // Collect points
    //     std::unordered_map<int, int> point_to_param_idx;
    //     int point_param_size = 3;
    //     for (const auto& [point_id, point] : map.map_points) {
    //         if (point.is_bad || point.obs.empty()) continue;
    //         point_to_param_idx[point_id] = point_params.size() / point_param_size;

    //         point_params.push_back(point.position.x);
    //         point_params.push_back(point.position.y);
    //         point_params.push_back(point.position.z);
    //     }

    //     // Build the problem
    //     ceres::Problem problem;

    //     for (const auto& [point_id, point] : map.map_points) {
    //         if (point.is_bad || point.obs.empty()) continue;
    //         int point_idx = point_to_param_idx[point_id];

    //         for (const auto& obs : point.obs) {
    //             int kfid = obs.keyframe_id;
    //             if (map.keyframes.find(kfid) == map.keyframes.end()) continue;

    //             int cam_idx = kf_to_param_idx[kfid];

    //             ceres::CostFunction* cost_function = ReprojectionError::Create(obs.point2D, cameraMatrix);
    //             problem.AddResidualBlock(cost_function, nullptr,
    //                                     &camera_params[cam_idx * cam_param_size],
    //                                     &point_params[point_idx * point_param_size]);
    //         }
    //     }

    //     // Fix the first camera to remove gauge freedom
    //     if (!map.keyframes.empty()) {
    //         int first_kf_id = map.keyframes.begin()->first;
    //         int first_cam_idx = kf_to_param_idx[first_kf_id];
    //         problem.SetParameterBlockConstant(&camera_params[first_cam_idx * cam_param_size]);
    //     }

    //     {
    //         const int anchor_kf_id = 0; // known bootstrap keyframe
    //         const int cam_param_size = 6;
    //         const int anchor_cam_idx = kf_to_param_idx.at(anchor_kf_id);
    //         problem.SetParameterBlockConstant(&camera_params[anchor_cam_idx * cam_param_size]);
    //         // If only translation should be fixed, use SubsetParameterization instead:
    //         // std::vector<int> fixed = {3,4,5};
    //         // auto* subset = new ceres::SubsetParameterization(6, fixed);
    //         // problem.SetParameterization(&camera_params[anchor_cam_idx * cam_param_size], subset);
    //     }

    //     // Solve
    //     ceres::Solver::Options options;
    //     options.linear_solver_type = ceres::SPARSE_SCHUR;  // Or SPARSE_SCHUR for larger problems
    //     options.minimizer_progress_to_stdout = true;
    //     options.max_num_iterations = 50; // increase from default ~50
    //     ceres::Solver::Summary summary;
    //     ceres::Solve(options, &problem, &summary);
    //     std::cout << summary.FullReport() << std::endl;

    //     // Update the map with optimized values
    //     // Update cameras
    //     for (const auto& [kfid, idx] : kf_to_param_idx) {
    //         double* cam = &camera_params[idx * cam_param_size];
    //         // cv::Mat Rr = (cv::Mat_<double>(3,1) << cam[0], cam[1], cam[2]);
    //         // cv::Mat Tr = (cv::Mat_<double>(3,1) << cam[3], cam[4], cam[5]);

    //         // Rr = Rr.t();
    //         // Tr = -Rr * Tr;
            
    //         // cv::Rodrigues(Rr, map.keyframes[kfid].R);
    //         // map.keyframes[kfid].t = Tr;
    //         cv::Mat angle_axis = (cv::Mat_<double>(3,1) << cam[0], cam[1], cam[2]);

    //         cv::Rodrigues(angle_axis, map.keyframes[kfid].R);
    //         map.keyframes[kfid].t = (cv::Mat_<double>(3,1) << cam[3], cam[4], cam[5]);
    //         map.keyframes[kfid].R = map.keyframes[kfid].R.t();
    //         map.keyframes[kfid].t = -map.keyframes[kfid].R * map.keyframes[kfid].t;
    //     }

    //     // Update points
    //     for (const auto& [point_id, idx] : point_to_param_idx) {
    //         double* pt = &point_params[idx * point_param_size];
    //         map.map_points[point_id].position = cv::Point3d(pt[0], pt[1], pt[2]);
    //     }

    //     // Compute average reprojection error after BA
    //     double avg_error_after = ComputeAverageReprojectionError(map, cameraMatrix);
    //     std::cout << "Average Reprojection Error After BA: " << avg_error_after << " px" << std::endl;
    // }

    // After the main loop, analyze map points
    {
        // std::lock_guard<std::mutex> lock(map_mutex);  // Ensure thread-safe access

        int count_low_error_high_obs = 0;
        int count_low_error_low_obs = 0;
        int count_high_error_high_obs = 0;
        int count_high_error_low_obs = 0;

        const double error_threshold = 1.0;  // Pixels
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
