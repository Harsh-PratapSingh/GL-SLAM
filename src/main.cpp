#include "core/slam_core.h"
#include "visualization/visualization.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>

const int num_images = 500;
const int window_size = 100; // Sliding window size //tested 30 and 100 to be ok

std::vector<cv::Mat> Rs_gt(num_images*3), Ts_gt(num_images*3);
std::vector<cv::Mat> Rs_est(num_images), Ts_est(num_images);
std::vector<Point3D> global_points3D;
std::vector<Point3D> temp_points3D;
cv::Mat K;


int main() {
    std::string dir_path = "/home/tomato/Downloads/data_odometry_gray/dataset/sequences/00/";
    TensorRTInference infer("../third_party/Superpoint_Lightglue/superpoint_lightglue_pipeline.trt.onnx",
                           "../third_party/Superpoint_Lightglue/superpoint_lightglue_pipeline.trt");

    K = slam_core::load_calibration(dir_path + "calib.txt");

    auto poses = slam_core::load_poses(dir_path + "00.txt", num_images*3);
    
    for (int i = 0; i < num_images*3; ++i) {
        Rs_gt[i] = poses[i](cv::Rect(0, 0, 3, 3));
        Ts_gt[i] = poses[i](cv::Rect(3, 0, 1, 3));
    }

    
    Rs_est[0] = cv::Mat::eye(3, 3, CV_64F);
    Ts_est[0] = cv::Mat::zeros(3, 1, CV_64F);

    
    std::vector<cv::Point2f> prev_points2;
    std::vector<int> prev_points2_indices;
    int gidx = 0;
    int c =0 ;
    int last_valid_frame2 = 1;
    int last_valid_frame1 = 0;

    for (int i = 0; i < num_images - 1; ++i) {
        std::ostringstream oss1, oss2;
        oss1 << dir_path << "image_0/" << std::setw(6) << std::setfill('0') << last_valid_frame1 << ".png";
        oss2 << dir_path << "image_0/" << std::setw(6) << std::setfill('0') << last_valid_frame2 << ".png";

        cv::Mat img1 = cv::imread(oss1.str(), cv::IMREAD_GRAYSCALE);
        cv::Mat img2 = cv::imread(oss2.str(), cv::IMREAD_GRAYSCALE);
        if (img1.empty() || img2.empty()) {
            std::cerr << "Failed to load images: " << oss1.str() << " or " << oss2.str() << "\n";
            return -1;
        }

        std::vector<cv::Point2f> points1, points2;
        slam_core::process_keypoints(infer, img1, img2, points1, points2, i);

        // Optical Flow Magnitude Check
        double total_displacement = 0.0;
        for (size_t j = 0; j < points1.size(); ++j) {
            total_displacement += cv::norm(points2[j] - points1[j]);
        }
        double avg_displacement = total_displacement / points1.size();
        if (avg_displacement < 10.0) {
            std::cout << "Skipping frame " << last_valid_frame2 << " ,avg_displacement=" << avg_displacement << " pixels)\n";
            i--;
            last_valid_frame2++;
            continue;
        }

        cv::Mat R_rel, T_rel;
        cv::Mat mask;
        cv::Mat E = cv::findEssentialMat(points1, points2, K, cv::RANSAC, 0.9999, 0.5, mask);
        cv::recoverPose(E, points1, points2, K, R_rel, T_rel, mask);
        R_rel = R_rel.t();
        T_rel = -R_rel * T_rel;

        Rs_est[i + 1] = Rs_est[i] * R_rel;
        Ts_est[i + 1] = Ts_est[i] + Rs_est[i] * T_rel;

        int a = 0;
        std::vector<int> exclude_indices;
        std::vector<int> temp;
        const float match_threshold = 1.0f;
        if (i > 0 && !prev_points2.empty()) {
            for (size_t j = 0; j < points1.size(); ++j) {
                if (mask.at<uchar>(j)) {
                    for (int idx : prev_points2_indices) {
                        for (const auto& obs : global_points3D[idx].observations) {
                            if (obs.camera_idx == i && cv::norm(points1[j] - obs.point2D) < match_threshold) {
                                global_points3D[idx].observations.push_back({i + 1, points2[j]});
                                exclude_indices.push_back(j);
                                temp.push_back(idx);
                                a++;
                                break;
                            }
                        }
                    }
                }
            }
            prev_points2_indices.clear();
            prev_points2_indices = temp;
            temp.clear();
            std::cout << "Updated observations for " << a << " keypoints that match previous frame.\n";
        }
        prev_points2 = points2;

        std::vector<cv::Point2f> projected_points;
        if (i == num_images - 2) {
            for (const auto& point3D : global_points3D) {
                for (const auto& obs : point3D.observations) {
                    if (obs.camera_idx == i + 1) {
                        cv::Point3f X = point3D.position;
                        cv::Mat R = Rs_est[i + 1].t();
                        cv::Mat t = -R * Ts_est[i + 1];
                        cv::Mat X_hom = (cv::Mat_<double>(4, 1) << X.x, X.y, X.z, 1.0);
                        cv::Mat P = K * (cv::Mat_<double>(3, 4) << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
                                                                  R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
                                                                  R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2));
                        cv::Mat x_proj_hom = P * X_hom;
                        cv::Point2f x_proj(x_proj_hom.at<double>(0) / x_proj_hom.at<double>(2),
                                          x_proj_hom.at<double>(1) / x_proj_hom.at<double>(2));
                        projected_points.push_back(x_proj);
                    }
                }
            }
        }
        slam_visualization::visualize_optical_flow(img2, points1, points2, mask, i + 1, projected_points);

        std::vector<cv::Point3f> points3D = slam_core::triangulatePoints(K, Rs_est[i], Ts_est[i], Rs_est[i + 1], Ts_est[i + 1],
                                                                        points1, points2, mask, exclude_indices);

        const double x_min = -50.0, x_max = 50.0;
        const double y_min = -5.0, y_max = 0.5;
        const double z_min = 0.0, z_max = 50.0;
        int filtered_points = 0;
        for (size_t k = 0; k < points3D.size(); ++k) {
            cv::Point3f world_point = points3D[k];
            cv::Mat P_w = (cv::Mat_<double>(3, 1) << world_point.x, world_point.y, world_point.z);
            cv::Mat R2_64f, T2_64f;
            Rs_est[i + 1].convertTo(R2_64f, CV_64F);
            Ts_est[i + 1].convertTo(T2_64f, CV_64F);
            cv::Mat R2_inv = R2_64f.t();
            cv::Mat P_c;
            cv::subtract(P_w, T2_64f, P_c, cv::noArray(), CV_64F);
            P_c = R2_inv * P_c;

            double x_coord = P_c.at<double>(0);
            double y_coord = P_c.at<double>(1);
            double z_coord = P_c.at<double>(2);

            if (x_coord >= x_min && x_coord <= x_max &&
                y_coord >= y_min && y_coord <= y_max &&
                z_coord >= z_min && z_coord <= z_max) {
                Point3D new_point;
                new_point.position = world_point;
                new_point.observations.push_back({i, points1[k]});
                new_point.observations.push_back({i + 1, points2[k]});
                global_points3D.push_back(new_point);
                prev_points2_indices.push_back(gidx);
                gidx++;
            } else {
                filtered_points++;
                // std::cout << "Filtered point " << k << ": camera coords (" 
                //          << x_coord << ", " << y_coord << ", " << z_coord << ")\n";
            }
        }
        std::cout << "Total points filtered by spatial bounds: " << filtered_points << "\n";

        std::cout << "New points: " << points3D.size() - filtered_points << " \n";
        std::cout << "Mask: " << points1.size() << " \n";
        std::cout << "Image " << i + 1 << ":\n";
        std::cout << "Estimated T: " << Ts_est[i + 1].t() << "\n";
        std::cout << "Ground Truth T: " << Ts_gt[last_valid_frame2].t() << "\n";
        std::cout << "Rotation Error: " << slam_core::compute_rotation_error(Rs_est[i + 1], Rs_gt[last_valid_frame2]) << " deg\n";
        std::cout << "Translation Error: " << slam_core::compute_translation_error(Ts_est[i + 1], Ts_gt[last_valid_frame2]) << " deg\n";
        
        if(c >= window_size/3){
            slam_core::processSlidingWindowBA(i, window_size, K, Rs_est, Ts_est, global_points3D);
            c =0;
        }
        c++;
        if(last_valid_frame2-last_valid_frame1 > 1){
            last_valid_frame1 = last_valid_frame2 ;
            last_valid_frame2++;
        }
        else{
            last_valid_frame1++;
            last_valid_frame2++;
        }
        
    }


    int p = 0;
    for (size_t k = 0; k < global_points3D.size(); ++k) {
        int l = 0;
        for (const auto& obs : global_points3D[k].observations) {
            l++;
        }
        if (l > 2) {
            temp_points3D.push_back(global_points3D[k]);
            p++;
        }
    }
    global_points3D = temp_points3D;
    temp_points3D.clear();
    std::cout << "points " << p << " in over 4 Camera\n";

    // Compute reprojection errors
    double total_error = 0.0;
    int num_observations = 0;
    int bad_obs = 0;
    int good_obs = 0;
    int worse_obs = 0;
    int multi_obs = 0;
    std::vector<double> reprojection_errors;
    for (size_t point_idx = 0; point_idx < global_points3D.size(); ++point_idx) {
        int o = 0;
        const auto& point3D = global_points3D[point_idx];
        cv::Point3f X = point3D.position;
        for (const auto& obs : point3D.observations) {
            int cam_idx = obs.camera_idx;
            cv::Point2f x_obs = obs.point2D;

            cv::Mat R = Rs_est[cam_idx].t();
            cv::Mat t = -R * Ts_est[cam_idx];

            cv::Mat X_hom = (cv::Mat_<double>(4, 1) << X.x, X.y, X.z, 1.0);
            cv::Mat P = K * (cv::Mat_<double>(3, 4) << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
                                                      R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
                                                      R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2));
            cv::Mat x_proj_hom = P * X_hom;
            cv::Point2f x_proj(x_proj_hom.at<double>(0) / x_proj_hom.at<double>(2),
                              x_proj_hom.at<double>(1) / x_proj_hom.at<double>(2));

            cv::Point2f error = x_obs - x_proj;
            double err = cv::norm(error);
            total_error += err;
            num_observations++;
            reprojection_errors.push_back(err);
            if(err >= 2){
                bad_obs++;
            }
            if (err <= 0.3){
                good_obs++;
            }
            if (err >= 4){
                worse_obs++;
            }
            o++;
        }
        if (o >=5){
            multi_obs++;
        }
    }

    if (num_observations > 0) {
        double avg_error = total_error / num_observations;
        std::cout << "Average reprojection error: " << avg_error << " pixels\n";
        std::cout << "Total observations checked: " << num_observations << "\n";
        double max_error = *std::max_element(reprojection_errors.begin(), reprojection_errors.end());
        double min_error = *std::min_element(reprojection_errors.begin(), reprojection_errors.end());
        std::cout << "Max reprojection error: " << max_error << " pixels\n";
        std::cout << "Min reprojection error: " << min_error << " pixels\n";
        std::cout << "Bad observations > 2 pixels: " << bad_obs << " \n";
        std::cout << "Worse observations > 4 pixels: " << worse_obs << " \n";
        std::cout << "good observations < 0.5 pixels: " << good_obs << " \n";
        std::cout << "No. of points:" << global_points3D.size() << " \n";
        std::cout << "points observer in cameras >=5 " << multi_obs << "\n";
    } else {
        std::cout << "No observations available to compute reprojection error.\n";
    }

    slam_visualization::visualize_poses(Rs_est, Ts_est, Rs_gt, Ts_gt, global_points3D);

    return 0;
}