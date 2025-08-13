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

    cv::Mat load_calibration(const std::string& path) {
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) {
            if (line.substr(0, 2) == "P0") {
                std::istringstream iss(line.substr(4));
                cv::Mat P0(3, 4, CV_64F);
                for (int i = 0; i < 12; ++i) iss >> P0.at<double>(i / 4, i % 4);
                cv::Mat K = P0(cv::Rect(0, 0, 3, 3));
                std::cout << "Loaded calibration matrix K:\n" << K << "\n";
                return K;
            }
        }
        std::cerr << "P0 not found\n";
        exit(-1);
    }

    // std::vector<cv::Mat> load_poses(const std::string& path, int num_poses) {
    //     std::ifstream file(path);
    //     std::vector<cv::Mat> poses;
    //     std::string line;
    //     for (int i = 0; i < num_poses && std::getline(file, line); ++i) {
    //         std::istringstream iss(line);
    //         cv::Mat pose(3, 4, CV_64F);
    //         for (int j = 0; j < 12; ++j) iss >> pose.at<double>(j / 4, j % 4);
    //         poses.push_back(pose);
    //     }
    //     if (poses.size() < num_poses) {
    //         std::cerr << "Failed to load " << num_poses << " poses\n";
    //         exit(-1);
    //     }
    //     return poses;
    // }

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

}