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


std::mutex map_mutex;  // To synchronize map access

void visualize_map(const Map& map) {
    pangolin::CreateWindowAndBind("SLAM Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    auto& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        // Lock mutex to safely read map
        {
            std::lock_guard<std::mutex> lock(map_mutex);

            // Draw map points as a point cloud
            glPointSize(1);
            glBegin(GL_POINTS);
            glColor3f(0.0, 0.0, 1.0);  // Black points
            for (const auto& [mpid, mp] : map.map_points) {
                if (!mp.is_bad) {
                    // if(mp.position.y > 0.5 || mp.position.y < -1)continue;

                    glVertex3f(mp.position.x, mp.position.y, mp.position.z);
                }
            }
            glEnd();

            // Draw camera poses (as small pyramids or frames)
            for (const auto& [kfid, kf] : map.keyframes) {
                cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
                kf.R.copyTo(T(cv::Rect(0, 0, 3, 3)));
                kf.t.copyTo(T(cv::Rect(3, 0, 1, 3)));

                // Invert to get world-to-camera if needed (adjust based on your convention)
                T = invertSE3(T);

                // Draw camera frame (simple axes: red X, green Y, blue Z)
                float sz = 1.0f;  // Camera size
                glLineWidth(3);
                glBegin(GL_LINES);
                // X axis (red)
                glColor3f(1.0, 0.0, 0.0);
                glVertex3f(T.at<double>(0,3), T.at<double>(1,3), T.at<double>(2,3));
                glVertex3f(T.at<double>(0,3) + sz * T.at<double>(0,0),
                           T.at<double>(1,3) + sz * T.at<double>(1,0),
                           T.at<double>(2,3) + sz * T.at<double>(2,0));
                // Y axis (green)
                glColor3f(0.0, 1.0, 0.0);
                glVertex3f(T.at<double>(0,3), T.at<double>(1,3), T.at<double>(2,3));
                glVertex3f(T.at<double>(0,3) + sz * T.at<double>(0,1),
                           T.at<double>(1,3) + sz * T.at<double>(1,1),
                           T.at<double>(2,3) + sz * T.at<double>(2,1));
                // Z axis (blue)
                glColor3f(0.0, 0.0, 1.0);
                glVertex3f(T.at<double>(0,3), T.at<double>(1,3), T.at<double>(2,3));
                glVertex3f(T.at<double>(0,3) + sz * T.at<double>(0,2),
                           T.at<double>(1,3) + sz * T.at<double>(1,2),
                           T.at<double>(2,3) + sz * T.at<double>(2,2));
                glEnd();
            }
        }

        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));  // Update rate
    }
}


int main() {
    SuperPointTRT sp;
    LightGlueTRT lg;
    slam_core::superpoint_lightglue_init(sp, lg);


    std::string img_dir_path = "/home/tomato/Downloads/data_odometry_gray/dataset/sequences/00/image_0/";
    
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
    // t = slam_core::adjust_translation_magnitude(gtPoses, t, 1);


    cv::Mat R1 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t1 = cv::Mat::zeros(3, 1, CV_64F);
    auto [points3d, filteredPairs] = slam_core::triangulate_and_filter_3d_points(R1, t1, R, t, cameraMatrix, inliersPairs, 100.0, 0.5 );
    Map map;
    std::vector<int> a;
    slam_core::update_map_and_keyframe_data(map, img1, R, t, spRes1, points3d,
                                            filteredPairs, spRes0, img0, a, a, true, true);


    // After initial map update (bootstrap)
    std::thread viewer_thread(visualize_map, std::cref(map));


    // ===================== PnP on next image (temp5.png) =====================
    
    // After your two-view bootstrap and initial map update:
    auto fmt_name = [](int idx) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%06d.png", idx);
        return std::string(buf);
    };


    // const float match_thr = 0.7f;
    int prev_kfid = 1;            // last keyframe inserted during bootstrap (frame 1)
    int start_idx = 2;            // third image
    int max_idx   = 4000;           // or drive by gtPoses.size()-1


    for (int idx = start_idx; idx <= max_idx; ++idx) {
        // 1) Load current image and SuperPoint
        std::string img_path = img_dir_path + fmt_name(idx);
        cv::Mat img_cur = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (img_cur.empty()) {
            std::cerr << "[PnP-Loop] Could not load " << img_path << ", stopping.\n";
            break;
        }
        auto spRes_cur = sp.runInference(img_cur, img_cur.rows, img_cur.cols);


        // 2) Prev keyframe (features live in sp_res)
        auto kf_prev = map.keyframes[prev_kfid];


        // 3) Match prev ↔ cur with LightGlue
        auto lgRes_prev_cur = lg.run_Direct_Inference(kf_prev.sp_res, spRes_cur);


        auto all_pairs = slam_core::lightglue_score_filter(lgRes_prev_cur, match_thr);

        auto map_matches = slam_core::get_matches_from_previous_frames(
            lg, map, idx-1, 6, cameraMatrix, spRes_cur);
        // ---- Past-frames projection → synthetic SP result → LightGlue vs current ----
    

        // 4) Build 3D–2D (from prev’s kp_to_mpid) for PnP
        std::vector<cv::Point3f> p3d_pnp;
        std::vector<cv::Point2f> p2d_pnp;
        p3d_pnp.reserve(spRes_cur.numValid);
        p2d_pnp.reserve(spRes_cur.numValid);
        std::vector<int> map_point_id;          //exp
        std::vector<int> kp_index;              //exp

        int used3d = 0, skipped_no3d = 0;
        const auto kp2mp = kf_prev.kp_to_mpid;
        int n_prev = kf_prev.sp_res.numValid;

        int x = 0;
        for (int i = 0; i < n_prev; ++i) {
            int j = lgRes_prev_cur.matches0[i];
            
            if (j < 0 || lgRes_prev_cur.mscores0[i] <= match_thr) continue;


            int mpid = (i < (int)kp2mp.size()) ? kp2mp[i] : -1;
            if (mpid < 0) { 
                skipped_no3d++; 
                if(map_matches.count(j)){
                    x++;
                }
                continue; 
            }


            auto it = map.map_points.find(mpid);
            if (it == map.map_points.end() || it->second.is_bad) continue;


            const cv::Point3f Pw = it->second.position;
            p3d_pnp.emplace_back(Pw);


            float x = (float)lgRes_prev_cur.keypoints1[2 * j];
            float y = (float)lgRes_prev_cur.keypoints1[2 * j + 1];
            p2d_pnp.emplace_back(x, y);

            map_point_id.push_back(mpid);
            kp_index.push_back(j);
            
            used3d++;
        }
        std::cout << "[PnP-Loop] Frame " << idx << ": 3D-2D for PnP = " << used3d
                << " (no-3D=" << skipped_no3d << ")\n";
        std::cout << "map_matches" << x << std::endl;

        if ((int)p3d_pnp.size() < 4) {
            std::cerr << "[PnP-Loop] Not enough 3D–2D; skipping frame " << idx << "\n";
            continue;
        }


        // 5) PnP (world->camera)
        cv::Mat rvec, tvec, R_cur;
        std::vector<int> inliers_pnp;
        cv::Mat distCoeffs = cv::Mat::zeros(4,1,CV_64F);
        bool ok_pnp = cv::solvePnPRansac(
            p3d_pnp, p2d_pnp, cameraMatrix, distCoeffs,
            rvec, tvec, false, 1000, 1.8, 0.999, inliers_pnp, cv::SOLVEPNP_ITERATIVE
        );
        if (!ok_pnp || (int)inliers_pnp.size() < 4) {
            std::cerr << "[PnP-Loop] PnP failed/low inliers at frame " << idx << "\n";
            continue;
        }
        else{
            std::cerr << "[PnP-Loop] PnP inliers at frame " << idx << " = " << (int)inliers_pnp.size() << "\n";
        }
        cv::Rodrigues(rvec, R_cur);
        cv::Mat t_cur = tvec.clone(); // x_cam = R_cur * X_world + t_cur (world->cam)


        R_cur = R_cur.t();
        t_cur = -R_cur * t_cur;

        //t_cur = slam_core::adjust_translation_magnitude(gtPoses, t_cur, idx );



        // 6) Compare with GT
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


        // 7) Build the “rest” 2D–2D pairs (no 3D yet in prev) for triangulation
        // Use your LightGlue score filter to keep a clean set first
        // auto all_pairs = slam_core::lightglue_score_filter(lgRes_prev_cur, match_thr);


        // Split into: restPairs (idx0/idx1/p0/p1) where prev has no 3D
        std::vector<Match2D2D> restPairs;
        restPairs.reserve(all_pairs.size());
        for (const auto& m : all_pairs) {
            int i_prev = m.idx0;
            if (i_prev < 0 || i_prev >= n_prev) continue;
            int mpid = (i_prev < (int)kp2mp.size()) ? kp2mp[i_prev] : -1;
            if (mpid >= 0) continue;  // already has 3D
            restPairs.push_back(m);
        }


        std::cout << "Rest = " << restPairs.size() << std::endl;


        // 8) Triangulate-and-filter the “rest” using your helper (world->cam convention)
        //    Pprev = K[R_prev|t_prev], Pcur = K[R_cur|t_cur]
        cv::Mat R_prev = kf_prev.R;
        cv::Mat t_prev = kf_prev.t;
        R_prev = R_prev.t();
        t_prev = -R_prev * t_prev;
        R_cur = R_cur.t();
        t_cur = -R_cur * t_cur;
        


        auto [newPoints3D, newPairs] =
            slam_core::triangulate_and_filter_3d_points(R_prev, t_prev, R_cur, t_cur,
                                                        cameraMatrix, restPairs,
                                                        /*maxZ*/ 100.0, /*minCosParallax*/ 0.3);


        std::cout << "[PnP-Loop] Frame " << idx << " triangulated-new = " << newPoints3D.size() << "\n";


        // 9) Update map & keyframes via your helper (writes sp_res, kp_to_mpid, observations, etc.)
        //    Note: pass prev’s sp_res and the new current sp_res so the helper can wire indices correctly.
        std::lock_guard<std::mutex> lock(map_mutex);
        slam_core::update_map_and_keyframe_data(
            map,
            /*img_cur*/ img_cur,
            /*R_cur*/   R_cur,
            /*t_cur*/   t_cur,
            /*spRes_cur*/ spRes_cur,
            /*points3d*/ newPoints3D,
            /*pairs*/    newPairs,
            /*spRes_prev*/ kf_prev.sp_res,
            /*img_prev*/  kf_prev.img,   // ensure kf_prev.img was stored at bootstrap
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
