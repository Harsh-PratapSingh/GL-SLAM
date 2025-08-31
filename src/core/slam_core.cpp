#include "core/slam_core.h"
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <unordered_set>
#include <memory> 

#include <g2o/solvers/eigen/linear_solver_eigen.h> 
#include <g2o/core/robust_kernel_impl.h> 

namespace slam_core {
    // std::vector<cv::Point3f> triangulatePoints(const cv::Mat& K, const cv::Mat& R1, const cv::Mat& T1, 
    //                                           const cv::Mat& R2, const cv::Mat& T2, 
    //                                           std::vector<cv::Point2f>& points1, 
    //                                           std::vector<cv::Point2f>& points2, 
    //                                           const cv::Mat& mask,
    //                                           const std::vector<int>& exclude_indices) {
    //     cv::Mat R1_64f, T1_64f, R2_64f, T2_64f, K_64f;
    //     R1.convertTo(R1_64f, CV_64F);
    //     T1.convertTo(T1_64f, CV_64F);
    //     R2.convertTo(R2_64f, CV_64F);
    //     T2.convertTo(T2_64f, CV_64F);
    //     K.convertTo(K_64f, CV_64F);

    //     cv::Mat R1_inv = R1_64f.t();
    //     cv::Mat T1_inv = -R1_inv * T1_64f;
    //     cv::Mat R2_inv = R2_64f.t();
    //     cv::Mat T2_inv = -R2_inv * T2_64f;

    //     cv::Mat P1_ext, P2_ext;
    //     cv::hconcat(R1_inv, T1_inv, P1_ext);
    //     cv::Mat P1 = K_64f * P1_ext;
    //     P1.convertTo(P1, CV_64F);
    //     cv::hconcat(R2_inv, T2_inv, P2_ext);
    //     cv::Mat P2 = K_64f * P2_ext;
    //     P2.convertTo(P2, CV_64F);

    //     std::vector<cv::Point2f> points1_inlier, points2_inlier, p1, p2;
    //     std::vector<bool> exclude(points1.size(), false);
    //     for (int idx : exclude_indices) {
    //         if (idx >= 0 && idx < exclude.size()) exclude[idx] = true;
    //     }
    //     for (size_t i = 0; i < points1.size(); ++i) {
    //         if (mask.at<uchar>(i)) {
    //             p1.push_back(points1[i]);
    //             p2.push_back(points2[i]);
    //             if (!exclude[i]) {
    //                 points1_inlier.push_back(points1[i]);
    //                 points2_inlier.push_back(points2[i]);
    //             }
    //         }
    //     }
    //     points1 = points1_inlier;
    //     points2 = points2_inlier;

    //     cv::Mat points4D;
    //     cv::triangulatePoints(P1, P2, points1, points2, points4D);
    //     points4D.convertTo(points4D, CV_64F);

    //     std::vector<cv::Point3f> points3D;
    //     for (int i = 0; i < points4D.cols; ++i) {
    //         cv::Mat x = points4D.col(i);
    //         x /= x.at<double>(3);
    //         cv::Point3f world_point(x.at<double>(0), x.at<double>(1), x.at<double>(2));
    //         points3D.push_back(world_point);
    //     }
    //     return points3D;
    // }

    // Function to parse calib.txt and extract camera matrix from P0
    cv::Mat load_camera_matrix(const std::string& calibPath) {
        std::ifstream file(calibPath);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open calib.txt");
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.find("P0:") == 0) {
                std::istringstream iss(line.substr(3));  // Skip "P0:"
                cv::Mat P(3, 4, CV_64F);
                for (int i = 0; i < 12; ++i) {
                    iss >> P.at<double>(i / 4, i % 4);
                }
                // Camera matrix is the first 3x3 of P0
                return P.colRange(0, 3).clone();
            }
        }
        throw std::runtime_error("P0 not found in calib.txt");
    }

    std::vector<cv::Mat> load_poses(const std::string& path) {
        std::ifstream f(path);
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

    // void estimate_pose(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2,
    //                   const cv::Mat& K, cv::Mat& R, cv::Mat& T) {
    //     cv::Mat mask;
    //     cv::Mat E = cv::findEssentialMat(points1, points2, K, cv::RANSAC, 0.9999, 0.5, mask);
    //     cv::recoverPose(E, points1, points2, K, R, T, mask);
    //     R = R.t();
    //     T = -R * T;
    //     cv::Mat mask_loose;
    //     cv::Mat E_loose = cv::findEssentialMat(points1, points2, K, cv::RANSAC, 0.999, 1.0, mask_loose);
    //     int inliers_loose = cv::countNonZero(mask_loose);
    //     std::cout << "Inliers with loose RANSAC (p=0.999, thresh=1.0): " << inliers_loose << "\n";
    // }

    // void bundleAdjustment(std::vector<cv::Mat>& Rs_est, std::vector<cv::Mat>& Ts_est,
    //                     std::vector<Point3D>& points3D, const cv::Mat& K) {
    //     // Step 1: Initialize G2O optimizer
    //     g2o::SparseOptimizer optimizer;
    //     optimizer.setVerbose(true);
        
    //     // Set up solver
    //     typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    //     typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    //     auto solver = new g2o::OptimizationAlgorithmLevenberg(
    //         std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    //     optimizer.setAlgorithm(solver);

    //     // Step 2: Add camera poses as vertices
    //     std::vector<g2o::VertexSE3Expmap*> camera_vertices;
    //     for (size_t i = 0; i < Rs_est.size(); ++i) {
    //         Eigen::Matrix3d R_eigen;
    //         cv::cv2eigen(Rs_est[i], R_eigen);
    //         Eigen::Vector3d T_eigen;
    //         cv::cv2eigen(Ts_est[i], T_eigen);
    //         g2o::SE3Quat pose(R_eigen, T_eigen);

    //         g2o::VertexSE3Expmap* v_se3 = new g2o::VertexSE3Expmap();
    //         v_se3->setEstimate(pose);
    //         v_se3->setId(i);
    //         if (i == 0) v_se3->setFixed(true); // Fix first camera
    //         optimizer.addVertex(v_se3);
    //         camera_vertices.push_back(v_se3);
    //     }

    //     // Step 3: Add 3D points as vertices
    //     std::vector<g2o::VertexPointXYZ*> point_vertices;
    //     for (size_t i = 0; i < points3D.size(); ++i) {
    //         g2o::VertexPointXYZ* v_point = new g2o::VertexPointXYZ();
    //         Eigen::Vector3d point_eigen(points3D[i].position.x, points3D[i].position.y, points3D[i].position.z);
    //         v_point->setEstimate(point_eigen);
    //         v_point->setId(i + Rs_est.size());
    //         v_point->setMarginalized(true);
    //         optimizer.addVertex(v_point);
    //         point_vertices.push_back(v_point);
    //     }

    //     // Step 4: Add edges for observations
    //     g2o::CameraParameters* cam_params = new g2o::CameraParameters(
    //         K.at<double>(0,0), Eigen::Vector2d(K.at<double>(0,2), K.at<double>(1,2)), 0);
    //     cam_params->setId(0);
    //     optimizer.addParameter(cam_params);

    //     for (size_t i = 0; i < points3D.size(); ++i) {
    //         for (const auto& obs : points3D[i].observations) {
    //             g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
    //             edge->setVertex(0, point_vertices[i]);
    //             edge->setVertex(1, camera_vertices[obs.camera_idx]);
    //             Eigen::Vector2d measurement(obs.point2D.x, obs.point2D.y);
    //             edge->setMeasurement(measurement);
    //             edge->setInformation(Eigen::Matrix2d::Identity());
    //             edge->setParameterId(0, 0);
    //             optimizer.addEdge(edge);
    //         }
    //     }

    //     // Step 5: Optimize
    //     optimizer.initializeOptimization();
    //     optimizer.optimize(30);

    //     // Step 6: Update poses and points
    //     for (size_t i = 0; i < camera_vertices.size(); ++i) {
    //         g2o::SE3Quat optimized_pose = camera_vertices[i]->estimate();
    //         cv::eigen2cv(optimized_pose.rotation().toRotationMatrix(), Rs_est[i]);
    //         cv::eigen2cv(optimized_pose.translation(), Ts_est[i]);
    //     }
    //     for (size_t i = 0; i < point_vertices.size(); ++i) {
    //         Eigen::Vector3d optimized_point = point_vertices[i]->estimate();
    //         points3D[i].position = cv::Point3f(optimized_point[0], optimized_point[1], optimized_point[2]);
    //     }
    // }


    // double compute_rotation_error(const cv::Mat& R_est, const cv::Mat& R_gt) {
    //     double trace = cv::trace(R_est.t() * R_gt)[0];
    //     return acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / CV_PI;
    // }

    // double compute_translation_error(const cv::Mat& T_est, const cv::Mat& T_gt) {
    //     cv::Mat T_est_3 = T_est(cv::Rect(0, 0, 1, 3));
    //     cv::Mat T_gt_3 = T_gt(cv::Rect(0, 0, 1, 3));
    //     double cos_phi = T_est_3.dot(T_gt_3) / (cv::norm(T_est_3) * cv::norm(T_gt_3));
    //     return acos(std::clamp(cos_phi, -1.0, 1.0)) * 180.0 / CV_PI;
    // }

    void process_keypoints(TensorRTInference& infer, const cv::Mat& img1, const cv::Mat& img2,
                          std::vector<cv::Point2f>& img1_points_combined, std::vector<cv::Point2f>& img2_points_combined
                          ) {


        // Image 1: crop halves
        cv::Mat img1_left_src  = img1(cv::Rect(0,   0, 620, 376));  // left half
        cv::Mat img1_right_src = img1(cv::Rect(620, 0, 621, 376));  // right half

        // Step 1: pad top only (648 pixels)
        cv::Mat img1_left_top, img1_right_top;
        copyMakeBorder(img1_left_src,  img1_left_top,  648, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
        copyMakeBorder(img1_right_src, img1_right_top, 648, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0));

        // Step 2: pad right for left half (404 pixels), left for right half (403 pixels)
        cv::Mat img1_left, img1_right;
        copyMakeBorder(img1_left_top,  img1_left,  0, 0, 0, 404, cv::BORDER_CONSTANT, cv::Scalar(0));
        copyMakeBorder(img1_right_top, img1_right, 0, 0, 403, 0, cv::BORDER_CONSTANT, cv::Scalar(0));

        // Image 2: repeat same steps
        cv::Mat img2_left_src  = img2(cv::Rect(0,   0, 620, 376));
        cv::Mat img2_right_src = img2(cv::Rect(620, 0, 621, 376));

        cv::Mat img2_left_top, img2_right_top;
        copyMakeBorder(img2_left_src,  img2_left_top,  648, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
        copyMakeBorder(img2_right_src, img2_right_top, 648, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0));

        cv::Mat img2_left, img2_right;
        copyMakeBorder(img2_left_top,  img2_left,  0, 0, 0, 404, cv::BORDER_CONSTANT, cv::Scalar(0));
        copyMakeBorder(img2_right_top, img2_right, 0, 0, 403, 0, cv::BORDER_CONSTANT, cv::Scalar(0));



        

        // Inference on LEFT pair
        std::vector<int64_t> keypoints_left, matches_left;
        std::vector<float> scores_left;
        infer.runInference("img1_left.png", "img2_left.png", keypoints_left, matches_left, scores_left);

        // Inference on RIGHT pair
        std::vector<int64_t> keypoints_right, matches_right;
        std::vector<float> scores_right;
        infer.runInference("img1_right.png", "img2_right.png", keypoints_right, matches_right, scores_right);


        uint16_t i = 0;


        for (size_t m = 0; m < scores_left.size() && m * 3 + 2 < matches_left.size(); ++m) {
            if (scores_left[m] >= SCORE_THRESHOLD && matches_left[m * 3] == 0) {
                int first_img_idx = matches_left[m * 3 + 1];
                int second_img_idx = matches_left[m * 3 + 2];
                if (first_img_idx >= 0 && first_img_idx < MODEL_IMAGE_SIZE && second_img_idx >= 0 && second_img_idx < MODEL_IMAGE_SIZE) {
                    float x1 = keypoints_left[first_img_idx * 2];
                    float y1 = keypoints_left[first_img_idx * 2 + 1];
                    float x2 = keypoints_left[MODEL_IMAGE_SIZE * 2 + second_img_idx * 2];
                    float y2 = keypoints_left[MODEL_IMAGE_SIZE * 2 + second_img_idx * 2 + 1];

                    float x1o = x1;
                    float y1o = y1 - 648.0f;
                    float x2o = x2; 
                    float y2o = y2 - 648.0f;
                    

                    //removed middle line
                    if(x1o<618 && x2o < 618){
                        img1_points_combined.emplace_back(x1o, y1o);
                        img2_points_combined.emplace_back(x2o, y2o);
                        i++;
                    }
                    

                }
            }
        }

        for (size_t m = 0; m < scores_right.size() && m * 3 + 2 < matches_right.size(); ++m) {
            if (scores_right[m] >= SCORE_THRESHOLD && matches_right[m * 3] == 0) {
                int first_img_idx = matches_right[m * 3 + 1];
                int second_img_idx = matches_right[m * 3 + 2];
                if (first_img_idx >= 0 && first_img_idx < MODEL_IMAGE_SIZE && second_img_idx >= 0 && second_img_idx < MODEL_IMAGE_SIZE) {
                    float x1 = keypoints_right[first_img_idx * 2];
                    float y1 = keypoints_right[first_img_idx * 2 + 1];
                    float x2 = keypoints_right[MODEL_IMAGE_SIZE * 2 + second_img_idx * 2];
                    float y2 = keypoints_right[MODEL_IMAGE_SIZE * 2 + second_img_idx * 2 + 1];

                    float x1o = x1 + 217.0f;
                    float y1o = y1 - 648.0f;
                    float x2o = x2 + 217.0f; 
                    float y2o = y2 - 648.0f;
                    
                    //removed middle line
                    if (x1o > 621 && x2o > 621) {
                        img1_points_combined.emplace_back(x1o, y1o);
                        img2_points_combined.emplace_back(x2o, y2o);
                        i++;
                    }

                }
            }
        }

        std::cout << "Matches : " << i << "\n";

    //     std::vector<cv::Point2f> filtered_points1, filtered_points2;
    //     std::vector<float> distances;
    //     int c = 0;
    //     for (size_t j = 0; j < points1.size(); ++j) {
    //         float dx = points2[j].x - points1[j].x;
    //         float dy = points2[j].y - points1[j].y;
    //         float distance = std::sqrt(dx * dx + dy * dy);
    //         distances.push_back(distance);
    //     }
    //     if (!distances.empty()) {
    //         float variable_threshold = 100.0f;
    //         for (size_t j = 0; j < points1.size(); ++j) {
    //             if (distances[j] <= variable_threshold) {
    //                 filtered_points1.push_back(points1[j]);
    //                 filtered_points2.push_back(points2[j]);
    //                 c++;
    //             }
    //         }
    //     } else {
    //         filtered_points1 = points1;
    //         filtered_points2 = points2;
    //     }
    //     std::cout << "filtered " << points1.size() - c << " keypoints \n";
    //     points1 = filtered_points1;
    //     points2 = filtered_points2;
    }

    // void processSlidingWindowBA(const int i, const int window_size, const cv::Mat K, std::vector<cv::Mat> &Rs_est, 
    //                             std::vector<cv::Mat> &Ts_est, std::vector<Point3D> &global_points3D) {
    //     if (i < window_size - 1) return; // Wait until we have enough frames
    
    //     // Step 1: Define the window (last window_size frames)
    //     int start_idx = std::max(0, i + 1 - window_size); // Ensure we don’t go below 0
    //     int end_idx = i + 1;                              // Include the current frame
    //     std::vector<int> window_indices;
    //     for (int j = start_idx; j <= end_idx; ++j) {
    //         window_indices.push_back(j);
    //     }
    
    //     // Step 2: Extract poses for the window
    //     std::vector<cv::Mat> Rs_window, Ts_window;
    //     for (int idx : window_indices) {
    //         cv::Mat temp1 = Rs_est[idx].clone();
    //         cv::Mat temp2 = Ts_est[idx].clone();
    //         temp1 = temp1.t();
    //         temp2 = -temp1 * temp2;
    //         Rs_window.push_back(temp1); // Clone to avoid modifying global data yet
    //         Ts_window.push_back(temp2);
    //     }
    
    //     // Step 3: Identify 3D points observed in the window
    //     std::unordered_set<int> relevant_point_indices;
    //     for (size_t pt_idx = 0; pt_idx < global_points3D.size(); ++pt_idx) {
    //         for (const auto& obs : global_points3D[pt_idx].observations) {
    //             if (std::find(window_indices.begin(), window_indices.end(), obs.camera_idx) != window_indices.end()) {
    //                 relevant_point_indices.insert(pt_idx);
    //                 break; // Point is relevant if seen in any window frame
    //             }
    //         }
    //     }
    
    //     // Step 4: Create window-specific points with remapped observations
    //     std::vector<Point3D> points_window;
    //     std::unordered_map<int, int> global_to_local_pt_idx; // Maps global point index to local
    //     int local_pt_idx = 0;
    //     for (int pt_idx : relevant_point_indices) {
    //         Point3D local_pt;
    //         local_pt.position = global_points3D[pt_idx].position;
    //         for (const auto& obs : global_points3D[pt_idx].observations) {
    //             auto it = std::find(window_indices.begin(), window_indices.end(), obs.camera_idx);
    //             if (it != window_indices.end()) {
    //                 // Remap global camera index to local window index (0-based in window)
    //                 int local_cam_idx = std::distance(window_indices.begin(), it);
    //                 local_pt.observations.push_back({local_cam_idx, obs.point2D});
    //             }
    //         }
    //         if (!local_pt.observations.empty()) {
    //             points_window.push_back(local_pt);
    //             global_to_local_pt_idx[pt_idx] = local_pt_idx++;
    //         }
    //     }
    
    //     // Step 5: Call the original bundle adjustment function
    //     std::cout << "Running sliding window BA for frames " << start_idx << " to " << end_idx << "...\n";
    //     slam_core::bundleAdjustment(Rs_window, Ts_window, points_window, K);
    //     std::cout << "Sliding window BA completed.\n";
    
    //     // Step 6: Update global poses with optimized values
    //     for (size_t j = 0; j < window_indices.size(); ++j) {
    //         int global_idx = window_indices[j];
    //         cv::Mat temp1 = Rs_window[j].clone();
    //         cv::Mat temp2 = Ts_window[j].clone();
    //         temp1 = temp1.t();
    //         temp2 = -temp1 * temp2;
    //         Rs_est[global_idx] = temp1;
    //         Ts_est[global_idx] = temp2;
    //     }
    
    //     // Step 7: Update global points with optimized values
    //     for (const auto& pair : global_to_local_pt_idx) {
    //         int global_pt_idx = pair.first;
    //         int local_pt_idx = pair.second;
    //         global_points3D[global_pt_idx].position = points_window[local_pt_idx].position;
    //     }
    // }


    void superpoint_lightglue_init(SuperPointTRT& sp, LightGlueTRT& lg){

        sp.setWorkspaceSizeBytes(2ULL << 30);
        sp.setMaxKeypoints(2048);
        sp.setScoreThreshold(0.0f);
        const int spH = 376;
        const int spW = 1241;
        if (!sp.init("superpoint_2048.onnx", "superpoint_2048.engine", spH, spW)) {
            throw std::runtime_error("SuperPoint init failed");
        }
        if (!lg.init("superpoint_lightglue.onnx", "superpoint_lightglue.engine")) {
            throw std::runtime_error("LightGlueTRT init failed");
        }
    }

    std::vector<Match2D2D> lightglue_score_filter(LightGlueTRT::Result& result, const float& score){
 
        std::vector<Match2D2D> matches;
        uint16_t lg_matches = result.matches0.size();
        matches.reserve(lg_matches);

        for (int i = 0; i < lg_matches; ++i) {
            int j = result.matches0[i];
            if (j >= 0 && result.mscores0[i] > score) {
                matches.push_back({
                    i, j,
                    cv::Point2d((double)result.keypoints0[2*i],     (double)result.keypoints0[2*i + 1]),
                    cv::Point2d((double)result.keypoints1[2*j],     (double)result.keypoints1[2*j + 1])
                });
            }
        }

        matches.shrink_to_fit();
        std::cout << "Matches(Score = " << score << " ):" << matches.size() << " out of " << lg_matches << std::endl;

        return matches;
    }

    std::tuple<cv::Mat, cv::Mat, cv::Mat> pose_estimator(std::vector<Match2D2D>& matches, cv::Mat& K){
        
        std::vector<cv::Point2d> points0, points1;
        points0.reserve(matches.size());
        points1.reserve(matches.size());
        for (const auto& m : matches) {
            points0.push_back(m.p0);
            points1.push_back(m.p1);
        }

        cv::Mat essentialMat, mask, R, t;
        essentialMat = cv::findEssentialMat(points0, points1, K, cv::USAC_MAGSAC, 0.999, 1.0, mask);
        int inliers = cv::recoverPose(essentialMat, points0, points1, K, R, t, mask);

        return std::make_tuple(R, t, mask);
    }

    std::vector<Match2D2D> pose_estimator_mask_filter(std::vector<Match2D2D>& matches, cv::Mat mask){

        std::vector<Match2D2D> inliersPairs;
        inliersPairs.reserve(matches.size());
        const uchar* mptr = mask.ptr<uchar>();
        for (size_t k = 0; k < matches.size(); ++k) {
            if (mptr[k]) inliersPairs.push_back(matches[k]);
        }
        std::cout << "Extracted " << inliersPairs.size() << " inlier matches." << std::endl;

        return inliersPairs;
    }

    cv::Mat adjust_translation_magnitude(std::vector<cv::Mat>& gtPoses, cv::Mat& t, int frame){
        
        double t_gt_mag = cv::norm(gtPoses[frame](cv::Rect(3,0,1,3)));
        cv::Mat T = t*(t_gt_mag / cv::norm(t));
        return T;

    }

    std::tuple<std::vector<cv::Point3d>, std::vector<Match2D2D>> triangulate_and_filter_3d_points(
        cv::Mat& R1, cv::Mat& t1, cv::Mat& R2, cv::Mat& t2, cv::Mat& K, std::vector<Match2D2D> matches,
        const float& distance_threshold, const float& reprojection_threshold){

            cv::Mat P0(3, 4, CV_64F), P1(3, 4, CV_64F);
            R1.copyTo(P0.colRange(0, 3));
            t1.copyTo(P0.col(3));
            R2.copyTo(P1.colRange(0, 3));
            t2.copyTo(P1.col(3));
            P0 = K * P0;
            P1 = K * P1;

            std::vector<cv::Point2d> inlierPoints0, inlierPoints1;
            inlierPoints0.reserve(matches.size());
            inlierPoints1.reserve(matches.size());
            for (const auto& m : matches) {
                inlierPoints0.push_back(m.p0);
                inlierPoints1.push_back(m.p1);
            }

            cv::Mat X4;
            cv::triangulatePoints(P0, P1, inlierPoints0, inlierPoints1, X4);

            std::vector<cv::Point3d> points3d;
            std::vector<Match2D2D> filteredPairs;
            points3d.reserve(X4.cols);
            filteredPairs.reserve(X4.cols);

            cv::Mat T1 = cv::Mat::eye(4, 4, CV_64F);  
            R1.copyTo(T1(cv::Rect(0, 0, 3, 3)));       
            t1.copyTo(T1(cv::Rect(3, 0, 1, 3))); 

            cv::Mat T2 = cv::Mat::eye(4, 4, CV_64F);  
            R2.copyTo(T2(cv::Rect(0, 0, 3, 3)));       
            t2.copyTo(T2(cv::Rect(3, 0, 1, 3))); 

            for (int i = 0; i < X4.cols; ++i) {
                double w = X4.at<double>(3, i);
                if (std::abs(w) < 1e-9) continue; // Removed degenerate cases

                cv::Mat X4_cam1 = T1 * X4.col(i); // Transform point i into cam1 frame
                double Z_cam1 = X4_cam1.at<double>(2, 0) / w;
                if (Z_cam1 <= 0 || Z_cam1 > distance_threshold) continue; // checked if point is in front of the camera1

                cv::Mat X4_cam2 = T2 * X4.col(i); // Transform point i into cam2 frame
                double Z_cam2 = X4_cam2.at<double>(2, 0) / w;
                if (Z_cam2 <= 0 || Z_cam2 > distance_threshold) continue; // checked if point is in front of the camera2

                //reprojection error filter for cam1 
                cv::Point2d observed_uv = matches[i].p0;
                cv::Mat uv_homogeneous = K * (X4_cam1.rowRange(0, 3) / w);

                double u = uv_homogeneous.at<double>(0, 0) / uv_homogeneous.at<double>(2, 0); 
                double v = uv_homogeneous.at<double>(1, 0) / uv_homogeneous.at<double>(2, 0);

                double reproj_error = cv::norm(cv::Point2d(u, v) - observed_uv);
                if (reproj_error > reprojection_threshold) continue;

                //reprojection error filter for cam2
                observed_uv = matches[i].p1;
                uv_homogeneous = K * (X4_cam2.rowRange(0, 3) / w);         

                u = uv_homogeneous.at<double>(0, 0) / uv_homogeneous.at<double>(2, 0);
                v = uv_homogeneous.at<double>(1, 0) / uv_homogeneous.at<double>(2, 0);

                reproj_error = cv::norm(cv::Point2d(u, v) - observed_uv);
                if (reproj_error > reprojection_threshold) continue;


                double Z = X4.at<double>(2, i) / w;
                double X = X4.at<double>(0, i) / w;
                double Y = X4.at<double>(1, i) / w;

                points3d.emplace_back(X, Y, Z);

                // Corresponding inlier pair at same index i
                filteredPairs.push_back(matches[i]);
            }

            std::cout << "Triangulated " << points3d.size() << " 3D points." << std::endl;

            return std::make_tuple(points3d, filteredPairs);

        }

    void update_map_and_keyframe_data(Map& map, cv::Mat& img, cv::Mat& R, cv::Mat t,
        SuperPointTRT::Result& Result, std::vector<cv::Point3d>& points3d,
        std::vector<Match2D2D>& filteredPairs, SuperPointTRT::Result& f_res,
        cv::Mat& f_img, std::vector<int>& map_point_id, std::vector<int>& kp_index, bool if_first_frame = false, bool if_R_t_inversed = false){

        if(if_first_frame){
            Frame first;
            first.id = map.next_keyframe_id++;
            first.img = f_img;
            first.R = cv::Mat::eye(3,3,CV_64F);
            first.t = cv::Mat::zeros(3,1,CV_64F);
            
            first.sp_res = f_res;
            first.is_keyframe = true;

            map.keyframes[first.id] = first;
            map.keyframes[first.id].kp_to_mpid.assign(map.keyframes[first.id].sp_res.keypoints.size()/2, -1);

        }

        Frame frame;

        frame.id = map.next_keyframe_id++;
        frame.img = img;

        if(if_R_t_inversed){

            auto R1 = map.keyframes[frame.id-1].R;       
            auto t1 = map.keyframes[frame.id-1].t; 
            
            R = R.t();
            t = -R * t;

            // cv::Mat R2 = R1 * R;
            // cv::Mat t2 = t1 + R1 * t;

            frame.R = R.clone();
            frame.t = t.clone();
            frame.sp_res = Result;
            frame.is_keyframe = true;

            map.keyframes[frame.id] = frame;

        }else{

            auto R1 = map.keyframes[frame.id-1].R;       
            auto t1 = map.keyframes[frame.id-1].t; 

            // cv::Mat R2 = R1 * R;
            // cv::Mat t2 = t1 + R1 * t;

            frame.R = R.clone();
            frame.t = t.clone();
            frame.sp_res = Result;
            frame.is_keyframe = true;

            map.keyframes[frame.id] = frame;
        }

        //update map point data
        // if(if_first_frame){
        //     map.keyframes[frame.id-1].kp_to_mpid.assign(map.keyframes[frame.id-1].sp_res.keypoints.size()/2, -1);
        // }
        map.keyframes[frame.id].kp_to_mpid.assign(map.keyframes[frame.id].sp_res.keypoints.size()/2, -1);

        for (size_t i = 0; i < points3d.size(); ++i) {
            const auto& pr = filteredPairs[i];

            MapPoint mp;
            mp.id = map.next_point_id++;
            mp.position = cv::Point3d(points3d[i].x, points3d[i].y, points3d[i].z);

            Observation obs0, obs1;

            // Frame 0 observation
            obs0.keyframe_id = frame.id-1;
            obs0.kp_index = pr.idx0;
            const auto& kps0 = map.keyframes[frame.id-1].sp_res.keypoints;
            obs0.point2D = cv::Point2d(static_cast<double>(kps0[2*obs0.kp_index]), static_cast<double>(kps0[2*obs0.kp_index+1]));
            map.keyframes[frame.id-1].kp_to_mpid[obs0.kp_index] = mp.id;

            // Frame 1 observation
            obs1.keyframe_id = frame.id;
            obs1.kp_index = pr.idx1;
            const auto& kps1 = map.keyframes[frame.id].sp_res.keypoints;
            obs1.point2D = cv::Point2d(static_cast<double>(kps1[2*obs1.kp_index]), static_cast<double>(kps1[2*obs1.kp_index+1]));
            map.keyframes[frame.id].kp_to_mpid[obs1.kp_index] = mp.id;

            mp.obs.push_back(obs0);
            mp.obs.push_back(obs1);
            map.keyframes[frame.id-1].map_point_ids.push_back(mp.id);
            map.keyframes[frame.id].map_point_ids.push_back(mp.id);

            map.map_points[mp.id] = mp;
            
        }
        
        int obs1 =0 ;
        if((!map_point_id.empty() || !kp_index.empty()) && !if_first_frame){
            for(int i = 0; i < map_point_id.size(); ++i){
                Observation obs;
                obs.keyframe_id = frame.id;
                obs.kp_index = kp_index[i];
                const auto& kps = map.keyframes[frame.id].sp_res.keypoints;
                obs.point2D = cv::Point2d(static_cast<double>(kps[2*obs.kp_index]), static_cast<double>(kps[2*obs.kp_index+1]));
                map.keyframes[frame.id].kp_to_mpid[obs.kp_index] = map_point_id[i];
                map.map_points[map_point_id[i]].obs.push_back(obs);
                ++obs1;
            }
            
        }

        std::cout << "Updated " << obs1 << " observations" << std::endl;

        std::cout << "Map contains " << map.map_points.size() << " MapPoints and "
              << map.keyframes.size() << " KeyFrames." << std::endl;

    }

    std::unordered_map<int, SyntheticMatch> get_matches_from_previous_frames(
        LightGlueTRT& lg, Map& map, int prev_frame_id, int i, cv::Mat& K,
        SuperPointTRT::Result& sp_res2, float score)
    {
        const int win = i;
        const int min_kfid = std::max(0, prev_frame_id - win);
        const int W = 1241, H = 376;

        // Collect candidate mpids from recent keyframes
        std::unordered_set<int> candidate_mpids;
        for (int kfid = min_kfid; kfid < prev_frame_id; ++kfid) {
            const auto& kf = map.keyframes[kfid];
            for (int mpid : kf.kp_to_mpid)
                if (mpid >= 0 && !map.map_points[mpid].is_bad) candidate_mpids.insert(mpid);
        }
        // Exclude those already seen in prev frame
        for (int mpid : map.keyframes[prev_frame_id].kp_to_mpid)
            if (mpid >= 0) candidate_mpids.erase(mpid);

        std::cout << "Candidate mpid = " << candidate_mpids.size() << std::endl;

        // Projection setup (world -> prev camera)
        const auto& prev_kf = map.keyframes[prev_frame_id];
        cv::Mat Rcw = prev_kf.R.t();
        cv::Mat tcw = -Rcw * prev_kf.t;

        const double fx = K.at<double>(0,0), cx = K.at<double>(0,2);
        const double fy = K.at<double>(1,1), cy = K.at<double>(1,2);

        std::vector<cv::Point2d> proj_uv;
        std::vector<const float*> proj_desc_ptrs;
        std::vector<int> mapid;
        std::set<std::pair<int,int>> occupied_px;

        proj_uv.reserve(candidate_mpids.size());
        proj_desc_ptrs.reserve(candidate_mpids.size());
        mapid.reserve(candidate_mpids.size());

        for (int mpid : candidate_mpids) {
            const auto& mp = map.map_points[mpid];

            // Latest observation within [min_kfid, prev_frame_id)
            int latest_kfid = -1, latest_kpidx = -1;
            for (const auto& ob : mp.obs) {
                if (ob.keyframe_id >= min_kfid && ob.keyframe_id < prev_frame_id) {
                    if (ob.keyframe_id > latest_kfid) {
                        latest_kfid = ob.keyframe_id;
                        latest_kpidx = ob.kp_index;
                    }
                }
            }
            if (latest_kfid < 0 || latest_kpidx < 0) continue;

            const auto& obs_kf = map.keyframes[latest_kfid];
            const size_t desc_sz = obs_kf.sp_res.descriptors.size();
            if ((latest_kpidx + 1) * 256 > desc_sz) continue;

            const float* desc_ptr = obs_kf.sp_res.descriptors.data() + latest_kpidx * 256;

            // Project into prev frame
            cv::Mat Pw = (cv::Mat_<double>(3,1) << mp.position.x, mp.position.y, mp.position.z);
            cv::Mat Pc = Rcw * Pw + tcw;
            double Z = Pc.at<double>(2);
            if (Z <= 0.0) continue;

            double x = Pc.at<double>(0) / Z, y = Pc.at<double>(1) / Z;
            double u = fx * x + cx, v = fy * y + cy;
            if (u < 0 || u >= W || v < 0 || v >= H) continue;

            auto pix = std::pair{(int)std::lround(u), (int)std::lround(v)};
            if (!occupied_px.insert(pix).second) continue;

            proj_uv.emplace_back(u, v);
            proj_desc_ptrs.push_back(desc_ptr);
            mapid.push_back(mpid);
        }

        std::cout << "proj_uv = " << proj_uv.size() << std::endl;

        std::unordered_map<int, SyntheticMatch> SynMatches;
        if (proj_uv.empty()) {
            std::cout << "[Synthetic] No valid projected points to match." << std::endl;
            return SynMatches;
        }

        // Build synthetic SuperPoint and match
        SuperPointTRT::Result synth;
        const int N = static_cast<int>(proj_uv.size());
        synth.numValid = N;
        synth.keypoints.resize(2 * N);
        synth.scores.assign(N, 1.0f);
        synth.descriptors.resize(256 * N);

        for (int j = 0; j < N; ++j) {
            synth.keypoints[2*j]     = static_cast<int64_t>(std::lround(proj_uv[j].x));
            synth.keypoints[2*j + 1] = static_cast<int64_t>(std::lround(proj_uv[j].y));
            std::copy(proj_desc_ptrs[j], proj_desc_ptrs[j] + 256, synth.descriptors.begin() + j * 256);
        }

        auto lgRes     = lg.run_Direct_Inference(synth, sp_res2);
        auto lgMatches = slam_core::lightglue_score_filter(lgRes, score);

        SynMatches.reserve(lgMatches.size());
        for (const auto& m : lgMatches) {
            if (m.idx1 >= 0 && m.idx0 >= 0 && m.idx0 < (int)mapid.size()) {
                SynMatches[m.idx1] = SyntheticMatch{m.idx1, mapid[m.idx0]};
                // SynMatches.push_back(SyntheticMatch{m.idx1, mapid[m.idx0]});
            }
        }

        std::cout << "[Synthetic] proj=" << N << " matches=" << SynMatches.size() << std::endl;
        return SynMatches;
    }

    std::tuple<cv::Mat, cv::Mat, cv::Mat, SuperPointTRT::Result,
        std::vector<Match2D2D>, std::vector<int>, std::vector<int>, bool> 
        run_pnp(Map& map, SuperPointTRT& sp, LightGlueTRT& lg,
            std::string& img_dir_path, cv::Mat& cameraMatrix, float match_thr,
            float map_match_thr, int idx, int window, bool get_inliner, std::vector<cv::Mat>& gtPoses){

        int prev_kfid = map.next_keyframe_id - 1; 
        auto img_name = [](int idx) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%06d.png", idx);
            return std::string(buf);
        };
        bool skip = false;

        std::string img_path = img_dir_path + img_name(idx);
        cv::Mat img_cur = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (img_cur.empty()) {
            std::cerr << "[PnP-Loop] Could not load " << img_path << ", stopping.\n";
            skip = true;
        }
        auto spRes_cur = sp.runInference(img_cur, img_cur.rows, img_cur.cols);
        const auto& kf_prev = map.keyframes.at(prev_kfid);
        auto lgRes_prev_cur = lg.run_Direct_Inference(kf_prev.sp_res, spRes_cur);
        auto all_pairs = slam_core::lightglue_score_filter(lgRes_prev_cur, match_thr);
        auto map_matches = slam_core::get_matches_from_previous_frames(
            lg, map, prev_kfid, window, cameraMatrix, spRes_cur, map_match_thr);
        
        // 4) Build 3D–2D (from prev’s kp_to_mpid) for PnP
        std::vector<cv::Point3d> p3d_pnp;
        std::vector<cv::Point2d> p2d_pnp;
        std::vector<int> map_point_id;          
        std::vector<int> kp_index;  
        std::vector<Match2D2D> restPairs;
        p3d_pnp.reserve(all_pairs.size());
        p2d_pnp.reserve(all_pairs.size());
        map_point_id.reserve(all_pairs.size());
        kp_index.reserve(all_pairs.size());
        restPairs.reserve(all_pairs.size());
                    
        int used3d = 0, skipped_no3d = 0;
        int x = 0;
        auto emplace = [](auto& map, auto mpid, auto& p3d_pnp,
                auto& p2d_pnp, auto& map_point_id, auto& kp_index,
                auto& used3d, auto& m){
                p3d_pnp.emplace_back(map.map_points[mpid].position);
                p2d_pnp.emplace_back(m.p1);
                map_point_id.push_back(mpid);
                kp_index.push_back(m.idx1);
                used3d++;
            };

        for (const auto& m : all_pairs) {
            int mpid = kf_prev.kp_to_mpid[m.idx0];
            if(mpid > 0){
                emplace(map, mpid, p3d_pnp, p2d_pnp, map_point_id,
                kp_index, used3d, m);
            }else if (map_matches.find(m.idx1) != map_matches.end()){
                x++;
                mpid = map_matches[m.idx1].mpid;
                emplace(map, mpid, p3d_pnp, p2d_pnp, map_point_id,
                kp_index, used3d, m);
            }else{
                restPairs.push_back(m);
                skipped_no3d++;
            }
        }

        if ((int)p3d_pnp.size() < 4) {
            std::cerr << "[PnP-Loop] Not enough 3D–2D; skipping frame " << idx << "\n";
            skip = true;
        }

        // 5) PnP (world->camera)
        cv::Mat rvec, tvec, R_cur, t_cur;
        std::vector<int> inliers_pnp;
        cv::Mat distCoeffs = cv::Mat::zeros(4,1,CV_64F);
        if(!skip){
            bool ok_pnp = cv::solvePnPRansac(
                p3d_pnp, p2d_pnp, cameraMatrix, distCoeffs,
                rvec, tvec, false, 1000, 1.0, 0.999, inliers_pnp, cv::USAC_MAGSAC
            );
            if (!ok_pnp || (int)inliers_pnp.size() < 4) {
                std::cerr << "[PnP-Loop] PnP failed/low inliers at frame " << idx << "\n";
                skip = true;
            }
            if (get_inliner){
                std::vector<int> mapid;
                std::vector<int> keyid;
                for (int idx : inliers_pnp) {
                    mapid.push_back(map_point_id[idx]);
                    keyid.push_back(kp_index[idx]);
                }
                map_point_id = mapid;
                kp_index = keyid;
            }
            cv::Rodrigues(rvec, R_cur);
            t_cur = tvec.clone(); 
            R_cur.convertTo(R_cur, CV_64F);
            t_cur.convertTo(t_cur, CV_64F);
            // R_cur = R_cur.t();
            // t_cur = -R_cur * t_cur;
        }
        

        // t_cur = slam_core::adjust_translation_magnitude(gtPoses, t_cur, idx );

        std::cout << "[PnP-Loop] Map matches = " << x << std::endl;
        std::cout << "[PnP-Loop] Matches without Map points = " << restPairs.size() << std::endl;
        std::cout << "[PnP-Loop] Frame " << idx << ": 3D-2D for PnP = " << used3d
                << " (no-3D=" << skipped_no3d << ")\n";
        std::cerr << "[PnP-Loop] PnP inliers at frame " << idx << " = " << (int)inliers_pnp.size() << " , " << map_point_id.size()  << "\n";

        return std::make_tuple(img_cur, R_cur, t_cur, spRes_cur, restPairs, map_point_id, kp_index, skip);
    }   

    // Function to refine pose with motion-only BA
    void refine_pose_with_g2o(cv::Mat& R_cur, cv::Mat& t_cur, SuperPointTRT::Result& spRes_cur,
                            std::vector<int>& map_point_id, std::vector<int>& kp_index,
                            Map& map, cv::Mat& cameraMatrix) {
        
        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        optimizer.setVerbose(true);
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
        typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;

        auto linearSolver = std::make_unique<LinearSolverType>();
        auto blockSolver = std::make_unique<BlockSolverType>(std::move(linearSolver));
        auto algorithm = std::make_unique<g2o::OptimizationAlgorithmLevenberg>(std::move(blockSolver));
        optimizer.setAlgorithm(algorithm.release());  // Transfer ownership to optimizer

        // Add current pose vertex (optimizable)
        g2o::VertexSE3Expmap* pose_vertex = new g2o::VertexSE3Expmap();
        g2o::SE3Quat initial_pose;
        Eigen::Matrix3d R_eig;
        cv::cv2eigen(R_cur, R_eig);
        Eigen::Vector3d t_eig;
        cv::cv2eigen(t_cur, t_eig);
        initial_pose = g2o::SE3Quat(R_eig, t_eig);
        pose_vertex->setEstimate(initial_pose);
        pose_vertex->setId(0);
        pose_vertex->setFixed(false);
        optimizer.addVertex(pose_vertex);

        // Camera parameters
        const double fx = cameraMatrix.at<double>(0, 0);
        const double fy = cameraMatrix.at<double>(1, 1);
        const double cx = cameraMatrix.at<double>(0, 2);
        const double cy = cameraMatrix.at<double>(1, 2);
        const double thHuber = sqrt(5.99);  // Chi-squared for 2 DoF at 95%

        // Add edges (unary, only connected to pose; fixed points set via Xw)
        for (size_t i = 0; i < kp_index.size(); ++i) {
            int point_idx = map_point_id[i];
            int kp_idx = kp_index[i];

            // Get 3D point from map (adjust to your Map struct, e.g., if map.points is std::vector<cv::Point3f>)
            cv::Point3d pt3d = map.map_points[point_idx].position;
            Eigen::Vector3d point3d(pt3d.x, pt3d.y, pt3d.z);

            // Get 2D observation
            double x = spRes_cur.keypoints[2 * kp_idx];
            double y = spRes_cur.keypoints[2 * kp_idx + 1];
            Eigen::Vector2d obs(x, y);

            // Add edge: optimizes only pose, with fixed point in Xw
            g2o::EdgeSE3ProjectXYZOnlyPose* edge = new g2o::EdgeSE3ProjectXYZOnlyPose();
            edge->setVertex(0, pose_vertex);
            edge->setMeasurement(obs);
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->fx = fx;
            edge->fy = fy;
            edge->cx = cx;
            edge->cy = cy;
            edge->Xw = point3d;  // Fixed world point (no separate vertex needed)
            edge->setRobustKernel(new g2o::RobustKernelHuber());
            edge->robustKernel()->setDelta(thHuber);
            optimizer.addEdge(edge);
        }

        // Optimize
        optimizer.initializeOptimization();
        optimizer.optimize(100);  // Adjust iterations as needed

        // Extract refined pose
        g2o::SE3Quat refined_pose = pose_vertex->estimate();
        Eigen::Matrix3d refined_R = refined_pose.rotation().toRotationMatrix();
        Eigen::Vector3d refined_t = refined_pose.translation();
        cv::eigen2cv(refined_R, R_cur);
        cv::eigen2cv(refined_t, t_cur);

        // No manual cleanup needed; optimizer handles algorithm

    }

    
    OptimizedBAData perform_local_ba(const Map& map, const cv::Mat& cameraMatrix, int window_size, int current_kfid) {
        OptimizedBAData result;

        // Collect window keyframes
        int start_kfid = std::max(0, current_kfid - window_size + 1);
        std::vector<int> window_kfs;
        for (int i = start_kfid; i <= current_kfid; ++i) {
            if (map.keyframes.find(i) != map.keyframes.end()) {
                window_kfs.push_back(i);
            }
        }
        if (window_kfs.size() < 2) return result;  // Skip if too few frames

        // Collect unique visible map points
        std::unordered_set<int> visible_point_ids;
        for (int kfid : window_kfs) {
            const auto& kf = map.keyframes.at(kfid);
            for (int mpid : kf.map_point_ids) {
                visible_point_ids.insert(mpid);
            }
        }
        if (visible_point_ids.size() < 8) return result;  // Skip if too few points
        std::vector<int> point_list(visible_point_ids.begin(), visible_point_ids.end());

        // Set up g2o optimizer
        using BlockSolverT = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
        std::unique_ptr<BlockSolverT::LinearSolverType> linear_solver(new g2o::LinearSolverEigen<BlockSolverT::PoseMatrixType>());
        std::unique_ptr<BlockSolverT> block_solver(new BlockSolverT(std::move(linear_solver)));
        g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(algorithm);
        optimizer.setVerbose(true);  // Enable for debugging: shows chi2 and iterations

        // Camera parameters
        double fx = cameraMatrix.at<double>(0, 0);
        double fy = cameraMatrix.at<double>(1, 1);
        double cx = cameraMatrix.at<double>(0, 2);
        double cy = cameraMatrix.at<double>(1, 2);
        g2o::CameraParameters* cam_params = new g2o::CameraParameters(fx, Eigen::Vector2d(cx, cy), 0);
        cam_params->setId(0);
        optimizer.addParameter(cam_params);

        // Add pose vertices (with inversion before setting estimate)
        int vertex_id = 0;
        std::unordered_map<int, int> kf_to_vertex;  // kfid -> vertex ID
        for (size_t i = 0; i < window_kfs.size(); ++i) {
            int kfid = window_kfs[i];
            const auto& kf = map.keyframes.at(kfid);
            // Invert pose before setting in g2o
            cv::Mat R_inv = kf.R.t();
            cv::Mat t_inv = -R_inv * kf.t;
            // cv::Mat R_inv = kf.R;
            // cv::Mat t_inv = kf.t;
            Eigen::Matrix3d eigR;
            cv::cv2eigen(R_inv, eigR);
            Eigen::Vector3d eigt;
            cv::cv2eigen(t_inv, eigt);
            g2o::SE3Quat pose(eigR, eigt);

            g2o::VertexSE3Expmap* v_se3 = new g2o::VertexSE3Expmap();
            v_se3->setId(vertex_id);
            v_se3->setEstimate(pose);
            if (i == 0) v_se3->setFixed(true);  // Fix oldest pose
            optimizer.addVertex(v_se3);
            kf_to_vertex[kfid] = vertex_id++;
        }

        // Add point vertices
        std::unordered_map<int, int> point_to_vertex;  // mpid -> vertex ID
        for (int pid : point_list) {
            const auto& mp = map.map_points.at(pid);
            if (mp.is_bad) continue;
            Eigen::Vector3d eig_pt(mp.position.x, mp.position.y, mp.position.z);

            g2o::VertexPointXYZ* v_pt = new g2o::VertexPointXYZ();
            v_pt->setId(vertex_id);
            v_pt->setEstimate(eig_pt);
            v_pt->setMarginalized(true);  // Essential for point optimization
            optimizer.addVertex(v_pt);
            point_to_vertex[pid] = vertex_id++;
        }

        // Add reprojection error edges with robust kernel and scaled information
        int edge_count = 0;
        const float thHuber = std::sqrt(5.99f);  // Chi2 threshold for 2 DoF
        const double pixel_variance = 3.0;  // Adjust based on your keypoint accuracy (e.g., 1-5 pixels)
        for (int kfid : window_kfs) {
            const auto& kf = map.keyframes.at(kfid);
            for (size_t kp_idx = 0; kp_idx < kf.kp_to_mpid.size(); ++kp_idx) {
                int mpid = kf.kp_to_mpid[kp_idx];
                auto it = point_to_vertex.find(mpid);
                if (mpid == -1 || it == point_to_vertex.end()) continue;

                float x = kf.sp_res.keypoints[2 * kp_idx];
                float y = kf.sp_res.keypoints[2 * kp_idx + 1];
                Eigen::Vector2d measurement(x, y);

                g2o::EdgeSE3ProjectXYZ* edge = new g2o::EdgeSE3ProjectXYZ();
                edge->setVertex(0, optimizer.vertex(it->second));  // Point (vertex 0)
                edge->setVertex(1, optimizer.vertex(kf_to_vertex[kfid]));  // Pose (vertex 1)
                edge->setMeasurement(measurement);
                Eigen::Matrix2d info = Eigen::Matrix2d::Identity() / pixel_variance;  // Scaled for better weighting
                edge->setInformation(info);
                edge->setParameterId(0, 0);

                // Add robust kernel
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();
                rk->setDelta(thHuber);
                edge->setRobustKernel(rk);

                optimizer.addEdge(edge);
                ++edge_count;
            }
        }
        std::cout << "[BA] Added " << edge_count << " edges for " << point_to_vertex.size() << " points." << std::endl;
        if (edge_count < 8) return result;  // Skip if too few edges

        // Optimize and log chi2
        optimizer.initializeOptimization();
        double initial_chi2 = optimizer.chi2();
        std::cout << "[BA] Initial chi2: " << initial_chi2 << std::endl;
        optimizer.optimize(50);  // More iterations for better convergence
        double final_chi2 = optimizer.chi2();
        std::cout << "[BA] Final chi2: " << final_chi2 << " (delta: " << (initial_chi2 - final_chi2) << ")" << std::endl;

        // Extract optimized poses (with inversion after extraction)
        for (size_t i = 0; i < window_kfs.size(); ++i) {
            int kfid = window_kfs[i];
            auto v_se3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(kf_to_vertex[kfid]));
            g2o::SE3Quat opt_pose = v_se3->estimate();
            Eigen::Matrix3d eig_opt_R = opt_pose.rotation().toRotationMatrix();
            Eigen::Vector3d eig_opt_t = opt_pose.translation();
            cv::Mat opt_R, opt_t;
            cv::eigen2cv(eig_opt_R, opt_R);
            cv::eigen2cv(eig_opt_t, opt_t);
            // Invert back after optimization
            // opt_R = opt_R.t();
            // opt_t = -opt_R * opt_t;
            cv::Mat Rt(3, 4, CV_64F);
            opt_R.copyTo(Rt.colRange(0, 3));
            opt_t.copyTo(Rt.col(3));
            result.optimized_poses.emplace_back(kfid, Rt);
        }

        // Extract optimized points and check for movement (debug)
        for (auto& [pid, vid] : point_to_vertex) {
            auto v_pt = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(vid));
            Eigen::Vector3d opt_pt = v_pt->estimate();
            cv::Point3d opt_position(opt_pt.x(), opt_pt.y(), opt_pt.z());

            // Debug: Check delta for first point
            if (!point_list.empty() && pid == point_list.front()) {
                const auto& orig_mp = map.map_points.at(pid);
                double delta = cv::norm(cv::Point3d(orig_mp.position.x, orig_mp.position.y, orig_mp.position.z) - opt_position);
                std::cout << "[BA] Sample point " << pid << " delta: " << delta << std::endl;
            }

            result.optimized_points.emplace_back(pid, opt_position);
        }

        return result;
    }



}