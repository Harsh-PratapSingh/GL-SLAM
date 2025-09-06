#include "core/slam_core.h"
// #include "visualization/visualization.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_set>

cv::Mat K;


int main() { 

    std::string dir_path = "/home/tomato/Downloads/data_odometry_gray/dataset/sequences/00/";
    TensorRTInference infer("../third_party/Superpoint_Lightglue/superpoint_lightglue_pipeline.trt.onnx",
                           "../third_party/Superpoint_Lightglue/superpoint_lightglue_pipeline.trt");

    std::ostringstream oss1, oss2;
    oss1 << dir_path << "image_0/" << std::setw(6) << std::setfill('0') << 0 << ".png";
    oss2 << dir_path << "image_0/" << std::setw(6) << std::setfill('0') << 1 << ".png";

    cv::Mat img1 = cv::imread(oss1.str(), cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(oss2.str(), cv::IMREAD_GRAYSCALE);

    std::vector<cv::Point2f> img1_points_combined; img1_points_combined.reserve(4000);
    std::vector<cv::Point2f> img2_points_combined; img2_points_combined.reserve(4000);

    slam_core::process_keypoints(infer, img1, img2, img1_points_combined, img2_points_combined);

    K = slam_core::load_calibration(dir_path + "calib.txt");
    
    // 1) Ensure points are in float and have enough correspondences
    std::vector<cv::Point2f> pts1 = img1_points_combined;
    std::vector<cv::Point2f> pts2 = img2_points_combined;

    if (pts1.size() < 8 || pts2.size() < 8 || pts1.size() != pts2.size()) {
        std::cout << "Not enough or unequal correspondences: " << pts1.size() << " vs " << pts2.size() << "\n";
        return 0;
    }

    // 2) Find Essential matrix with RANSAC
    double ransac_prob = 0.9999;
    double ransac_thresh = 0.3; // in pixels, adjust if needed
    cv::Mat inlier_mask;
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, ransac_prob, ransac_thresh, inlier_mask);

    // 3) Recover pose (R,t) from image1 to image2 using inliers
    cv::Mat R_12, t_12;
    int inliers = cv::recoverPose(E, pts1, pts2, K, R_12, t_12, inlier_mask);
    cv::Mat R_21, t_21;
    R_21 = R_12.t();
    t_21 = -R_21 * t_12;
    

    // 4) World frame setup: fix first camera as world
    // T_w_c0 = [I | 0]
    cv::Mat R_w_c0 = cv::Mat::eye(3,3,CV_64F);
    cv::Mat t_w_c0 = cv::Mat::zeros(3,1,CV_64F);

    // 5) Pose of second camera in world: T_w_c1 = T_w_c0 * T_c0_c1 (with c0 as reference)
    // c0->c1 is (R_12, t_12). World frame aligns with c0, so T_w_c1 == T_c0_c1.
    cv::Mat R_w_c1 = R_21.clone();
    cv::Mat t_w_c1 = t_21.clone();

    // 6) Optional: scale handling (monocular ambiguity)
    // For now, leave t as unit-baseline; introduce scale later via triangulation or external cues.

    // 7) Print diagnostics
    std::cout << "Inliers: " << inliers << " / " << pts1.size() << "\n";
    std::cout << "R_21:\n" << R_21 << "\n";
    std::cout << "t_21 (unit scale):\n" << t_21.t() << "\n";

    // 8) Draw inlier matches on side-by-side visualization
    cv::Mat vis; 
    {
        cv::Mat side(img1.rows, img1.cols + img2.cols, CV_8UC3);
        cv::Mat L = side(cv::Rect(0, 0, img1.cols, img1.rows));
        cv::Mat R = side(cv::Rect(img1.cols, 0, img2.cols, img2.rows));
        cv::cvtColor(img1, L, cv::COLOR_GRAY2BGR);
        cv::cvtColor(img2, R, cv::COLOR_GRAY2BGR);

        for (size_t i = 0; i < pts1.size(); ++i) {
            if (inlier_mask.at<uchar>(static_cast<int>(i)) == 0) continue;
            cv::Point2f p1 = pts1[i];
            cv::Point2f p2 = pts2[i]; p2.x += img1.cols;

            cv::Scalar col(0, 255, 0);
            cv::circle(side, p1, 3, col, -1);
            cv::circle(side, p2, 3, col, -1);
            cv::line(side, p1, p2, cv::Scalar(255, 0, 0), 1);
        }
        vis = side;
        cv::imshow("Pose Inlier Matches", vis);
        cv::waitKey(0);
        cv::imwrite("pose_inlier_matches.png", vis);
    }

    // 9) Store into your Frame bookkeeping if desired
    Frame f0, f1;
    f0.id = 0; f0.R = R_w_c0.clone(); f0.t = t_w_c0.clone();
    f1.id = 1; f1.R = R_w_c1.clone(); f1.t = t_w_c1.clone();

    // Example prints to confirm world-fixed first pose and second pose
    std::cout << "T_w_c0: R=I, t=0\n";
    std::cout << "T_w_c1:\nR=\n" << f1.R << "\nt=\n" << f1.t.t() << "\n";

    return 0;
}