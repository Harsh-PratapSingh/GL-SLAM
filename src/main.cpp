#include "core/lightglue.h"
#include "core/superpoint.h"
#include "core/keypt2subpx.h"
#include "core/slam_core.h"
#include "visualization/visualization.h"
#include "threading/thread_pool.h"
#include <chrono> 
#include <thread>

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

// New function to compute fundamental matrix from R, t, K
cv::Mat computeFundamentalMatrix(const cv::Mat& R, const cv::Mat& t, const cv::Mat& K) {
    // Compute essential matrix E = [t]_x * R
    cv::Mat skew_t = (cv::Mat_<double>(3,3) << 0, -t.at<double>(2), t.at<double>(1),
                                                t.at<double>(2), 0, -t.at<double>(0),
                                                -t.at<double>(1), t.at<double>(0), 0);
    cv::Mat E = skew_t * R;

    // Fundamental matrix F = K^{-T} * E * K^{-1}
    cv::Mat K_inv = K.inv();
    cv::Mat F = K_inv.t() * E * K_inv;

    return F;
}

// New function to calculate average matched point to epipolar line distance
double calculateAvgEpipolarDistance(const std::vector<Match2D2D>& pairs, const cv::Mat& F) {
    double total_dist = 0.0;
    int count = 0;

    for (const auto& pr : pairs) {
        cv::Point2d p1 = pr.p0;  // Assuming MatchPair has pt1 and pt2 as cv::Point2d
        cv::Point2d p2 = pr.p1;

        // Homogeneous points
        cv::Mat p1h = (cv::Mat_<double>(3,1) << p1.x, p1.y, 1.0);
        cv::Mat p2h = (cv::Mat_<double>(3,1) << p2.x, p2.y, 1.0);

        // Epipolar line in image 2 for p1: F * p1
        cv::Mat line2 = F * p1h;
        double denom2 = std::sqrt(line2.at<double>(0)*line2.at<double>(0) + line2.at<double>(1)*line2.at<double>(1));
        double dist2 = std::abs(line2.dot(p2h)) / (denom2 + 1e-9);  // Avoid division by zero

        // Epipolar line in image 1 for p2: F^T * p2
        cv::Mat line1 = F.t() * p2h;
        double denom1 = std::sqrt(line1.at<double>(0)*line1.at<double>(0) + line1.at<double>(1)*line1.at<double>(1));
        double dist1 = std::abs(line1.dot(p1h)) / (denom1 + 1e-9);

        // Symmetric distance
        total_dist += (dist1 + dist2) / 2.0;
        count++;
    }

    return (count > 0) ? (total_dist / count) : 0.0;
}


int main() {

    std::thread viewer_thread(slam_visualization::visualize_map_loop, std::ref(slam_types::map), std::ref(slam_types::map_mutex));
    std::thread tracking_thread(thread_pool::tracking_thread);
    // slam_types::run_tracking.notify_one();
    std::thread local_ba_thread(thread_pool::map_optimizing_thread);
    viewer_thread.join();
    
    tracking_thread.join();
    local_ba_thread.join();

    // //std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        
    // }

    // {

    //     int count_low_error_high_obs = 0;
    //     int count_low_error_low_obs = 0;
    //     int count_high_error_high_obs = 0;
    //     int count_high_error_low_obs = 0;

    //     const double error_threshold = 1.0;  // Pixels
    //     const int obs_threshold = 3;

    //     for (const auto& [point_id, point] : map.map_points) {
    //         if (point.obs.empty() || point.is_bad) continue;

    //         // Count observations
    //         int num_obs = point.obs.size();

    //         // Compute average reprojection error
    //         double total_error = 0.0;
    //         int valid_obs = 0;
    //         cv::Mat position_mat = (cv::Mat_<double>(3,1) << point.position.x, point.position.y, point.position.z);

    //         for (const auto& obs : point.obs) {
    //             int kfid = obs.keyframe_id;
    //             if (map.keyframes.find(kfid) == map.keyframes.end()) continue;

    //             const auto& kf = map.keyframes.at(kfid);
                
    //             cv::Mat R1 = kf.R.clone();
    //             cv::Mat t1 = kf.t.clone();
    //             R1 = R1.t();
    //             t1 = -R1 * t1;
    //             // Project 3D point to camera coordinates (assuming R/t are camera-to-world)
    //             cv::Mat camera_point = R1 * position_mat + t1;
    //             if (camera_point.at<double>(2) <= 0) continue;  // Behind camera

    //             // Normalize
    //             double z = camera_point.at<double>(2);
    //             cv::Mat normalized = (cv::Mat_<double>(3,1) << camera_point.at<double>(0)/z, camera_point.at<double>(1)/z, 1.0);

    //             // Apply camera matrix for pixel coordinates
    //             cv::Mat projected_mat = cameraMatrix * normalized;
    //             cv::Point2d projected(projected_mat.at<double>(0), projected_mat.at<double>(1));

    //             // Reprojection error
    //             double error = cv::norm(projected - obs.point2D);
                
    //             //std::cout << "Error " << valid_obs << " : " << error << std::endl;
    //             total_error += error;
    //             valid_obs++;
    //         }

    //         if (valid_obs == 0) continue;

    //         double avg_error = total_error / valid_obs;

    //         // Categorize
    //         bool is_low_error = (avg_error < error_threshold);
    //         bool is_high_obs = (num_obs >= obs_threshold);

    //         if (is_low_error && is_high_obs) {
    //             count_low_error_high_obs++;
    //         } else if (is_low_error && !is_high_obs) {
    //             count_low_error_low_obs++;
    //         } else if (!is_low_error && is_high_obs) {
    //             count_high_error_high_obs++;
    //         } else {
    //             count_high_error_low_obs++;
    //         }
    //     }

    //     // Print counts
    //     std::cout << "Map Point Analysis:" << std::endl;
    //     std::cout << " - Low error (< " << error_threshold << " px) + High obs (>= " << obs_threshold << "): " << count_low_error_high_obs << std::endl;
    //     std::cout << " - Low error (< " << error_threshold << " px) + Low obs (< " << obs_threshold << "): " << count_low_error_low_obs << std::endl;
    //     std::cout << " - High error (>= " << error_threshold << " px) + High obs (>= " << obs_threshold << "): " << count_high_error_high_obs << std::endl;
    //     std::cout << " - High error (>= " << error_threshold << " px) + Low obs (< " << obs_threshold << "): " << count_high_error_low_obs << std::endl;
    //     std::cout << "Total map points analyzed: " << map.map_points.size() << std::endl;

    //     std::cout << "Frame 0 :-" << std::endl;
    //     std::cout << " R : " << map.keyframes[0].R << std::endl;
    //     std::cout << " t : " << map.keyframes[0].t << std::endl;
    //     std::cout << "Frame 1 :-" << std::endl;
    //     std::cout << " R : " << map.keyframes[1].R << std::endl;
    //     std::cout << " t : " << map.keyframes[1].t << std::endl;

    //     {
    //         const cv::Mat T_wi = gtPoses[1];
    //         cv::Mat R_gt = T_wi(cv::Rect(0,0,3,3)).clone();
    //         cv::Mat t_gt = T_wi(cv::Rect(3,0,1,3)).clone();

    //         double rot_err = rotationAngleErrorDeg(map.keyframes[1].R, R_gt);
    //         double t_dir_err = angleBetweenVectorsDeg(map.keyframes[1].t, t_gt);
    //         double t_mag_err = std::abs(cv::norm(map.keyframes[1].t) - cv::norm(t_gt));
    //         std::cout << "[PnP-Loop] Frame " << 1 << " | rot(deg): " << rot_err
    //                 << " t_dir(deg): " << t_dir_err << " t_mag(m): " << t_mag_err << "\n";
    //     }
    // }

    return 0;
}
