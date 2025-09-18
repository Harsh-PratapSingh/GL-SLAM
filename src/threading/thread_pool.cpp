#include "threading/thread_pool.h"

namespace thread_pool {

    auto img_name = [](int idx) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%06d.png", idx);
            return std::string(buf);
        };

    void tracking_thread()
    {
        slam_core::superpoint_lightglue_init(slam_types::sp, slam_types::lg);
        if (!slam_types::ks.init("../third_party/Keypt2Subpx/keypt2subpx_splg.onnx", "keypt2subpx_splg.engine")) {
            std::cerr << "Failed to initialize Keypt2SubpxTRT" << std::endl;
            exit;
        }
        auto cameraMatrix = slam_core::load_camera_matrix(slam_types::calibPath);
        auto gtPoses = slam_core::load_poses(slam_types::posesPath);
        {
            cv::Mat img0 = cv::imread(slam_types::img_dir_path + "000000.png", cv::IMREAD_GRAYSCALE);
            cv::Mat img1 = cv::imread(slam_types::img_dir_path + "000001.png", cv::IMREAD_GRAYSCALE);
    
            auto spRes0 = slam_types::sp.runInference(img0, img0.rows, img0.cols);
            auto spRes1 = slam_types::sp.runInference(img1, img1.rows, img1.cols);
            auto lgRes = slam_types::lg.run_Direct_Inference(spRes0, spRes1);
            
            auto k2sRes = slam_types::ks.run_Direct_Inference(lgRes, img0, img1);
    
            auto matches = slam_core::lightglue_score_filter(lgRes, k2sRes, slam_types::match_thr);
            if (matches.size() < 8) {
                std::cerr << "Not enough matches for pose estimation." << std::endl;
                exit;
            }
            auto [R, t, mask] = slam_core::pose_estimator(matches, cameraMatrix);
            auto inliersPairs = slam_core::pose_estimator_mask_filter(matches, mask);
            // R = R.t(); t = -R * t;
            t = slam_core::adjust_translation_magnitude(gtPoses, t, 1);
    
            cv::Mat R1 = cv::Mat::eye(3, 3, CV_64F);
            cv::Mat t1 = cv::Mat::zeros(3, 1, CV_64F);
            auto [points3d, filteredPairs] = slam_core::triangulate_and_filter_3d_points(R1, t1, R, t, cameraMatrix, inliersPairs, 100.0, 0.5);
            std::vector<ObsPairs> a;
            slam_core::update_map_and_keyframe_data(slam_types::map, img1, R, t, spRes1, points3d,
                                                    filteredPairs, spRes0, img0, a, true, true);
    
            cv::Mat Ra = slam_types::map.keyframes[1].R.clone();
            cv::Mat ta = slam_types::map.keyframes[1].t.clone();
            // while(slam_types::tracking_frame){};
            // cv::Mat delta_R = R_before * R_after.inv();
            cv::Mat delta_R = R1 * Ra.t();
            cv::Mat delta_t = t1 - delta_R * ta;
    
            std::cout << "Delta R = " << delta_R <<std::endl;
            std::cout << "Delta t = " << delta_t <<std::endl;
            cv::Mat Rnew = delta_R * Ra;
            cv::Mat tnew = delta_R * ta + delta_t;
            // slam_types::map.keyframes[1].R = Rnew;
            // slam_types::map.keyframes[1].t = tnew;
    
            {
                // cv::Mat F_bootstrap = computeFundamentalMatrix(R, t, cameraMatrix);
                // double avg_dist_bootstrap = calculateAvgEpipolarDistance(matches, F_bootstrap);
                // std::cout << "Bootstrap Average Epipolar Distance: " << avg_dist_bootstrap << " px" << std::endl;
            }
            
        }

        int prev_triangulated_frame = slam_types::map.next_keyframe_id -1;
        int prev_idx = 1;
        for (int idx = 2; idx <= slam_types::max_idx; ++idx) 
        {   
            std::unique_lock<std::mutex> tracking_lock(slam_types::tracking_mutex);
            std::cout << "TRACKING" << std::endl;
            // const int prev_kfid = slam_types::map.next_keyframe_id - 1; 
            double t_mag_err;
            if (gtPoses.size() > idx) {
                const cv::Mat T_1 = gtPoses[prev_idx];
                cv::Mat R_1 = T_1(cv::Rect(0,0,3,3)).clone();
                cv::Mat t_1 = T_1(cv::Rect(3,0,1,3)).clone();
                const cv::Mat T_2 = gtPoses[idx];
                cv::Mat R_2 = T_2(cv::Rect(0,0,3,3)).clone();
                cv::Mat t_2 = T_2(cv::Rect(3,0,1,3)).clone();

                double rot_err = slam_core::rotationAngleErrorDeg(R_1, R_2);
                // double t_dir_err = slam_core::angleBetweenVectorsDeg(tc, t_gt);
                t_mag_err = std::abs(cv::norm(t_1) - cv::norm(t_2));
                std::cout << "[PnP-Loop] Frame " << idx << " | t_mag(m): " << t_mag_err << "\n";
                if (t_mag_err < slam_types::mag_filter && rot_err < slam_types::rot_filter) continue;
                // cv::Mat delta_R = Rc * slam_types::map.keyframes[1].R.t();
                // cv::Mat delta_t = tc - delta_R * slam_types::map.keyframes[1].t;

                // std::cout << "Delta R = " << delta_R <<std::endl;
                // std::cout << "Delta t = " << delta_t <<std::endl;
            }
            const int prev_kfid = prev_triangulated_frame; 

            bool skip = false;

            std::string img_path = slam_types::img_dir_path + img_name(idx);
            cv::Mat img_cur = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
            if (img_cur.empty()) {
                std::cerr << "[PnP-Loop] Could not load " << img_path << ", stopping.\n";
                continue;
            }

            auto spRes_cur = slam_types::sp.runInference(img_cur, img_cur.rows, img_cur.cols);
            const auto& kf_prev = slam_types::map.keyframes.at(prev_kfid);
            auto lgRes_prev_cur = slam_types::lg.run_Direct_Inference(kf_prev.sp_res, spRes_cur);
            auto lgRes = lgRes_prev_cur;
            auto img0 = kf_prev.img.clone();
            auto img1 = img_cur.clone();
            // Keypt2SubpxTRT::Result k2sRes;
            auto k2sRes = slam_types::ks.run_Direct_Inference(lgRes, img0, img1);
            // std::cout << k2sRes.refined_keypt0.size() << std::endl;
            auto all_pairs = slam_core::lightglue_score_filter(lgRes_prev_cur, k2sRes, slam_types::match_thr);
            auto map_matches = slam_core::get_matches_from_previous_frames(
                slam_types::lg, slam_types::map, prev_kfid, slam_types::map_match_window, cameraMatrix, spRes_cur, slam_types::map_match_thr);
            if (all_pairs.size() < 8) {
                std::cerr << "Not enough matches for pose estimation." << std::endl;
                exit;
            }
            auto [R, t, mask] = slam_core::pose_estimator(all_pairs, cameraMatrix);
            // auto inliersPairs = slam_core::pose_estimator_mask_filter(all_pairs, mask);
            // if (inliersPairs.size() > 0) {
            //     std::cerr << "Inlier Pairs" << inliersPairs.size() << std::endl;
            // }
            
            
            int used3d = 0, skipped_no3d = 0;
            int x = 0;

            std::vector<ObsPairs> obsPairs;
            std::vector<Match2D2D> restPairs;
            restPairs.reserve(all_pairs.size());
            obsPairs.reserve(all_pairs.size());
            const uchar* mptr = mask.ptr<uchar>();
            int k = 0;
            for (const auto& m : all_pairs) {
                int mpid = kf_prev.kp_to_mpid[m.idx0];
                if(mpid > -1){
                    obsPairs.push_back({mpid,m.idx1,m.p1});
                    used3d++;
                } else if (map_matches.find(m.idx1) != map_matches.end()){
                    x++;
                    mpid = map_matches[m.idx1].mpid;
                    obsPairs.push_back({mpid,m.idx1,m.p1});
                    used3d++;
                }else if (mptr[k]){
                    restPairs.push_back(m);
                    skipped_no3d++;
                }
                k++;
            }
            std::cout << "Rest Pairs" << restPairs.size() << std::endl;

            R = R.t(); t = -R * t;
            // t = t*(t_mag_err/cv::norm(t));
            cv::Mat R_prev = slam_types::map.keyframes[prev_kfid].R.clone();
            cv::Mat t_prev = slam_types::map.keyframes[prev_kfid].t.clone();
            // std::cout << "R_prev_kfid = " << R_prev << std::endl;
            // std::cout << "t_prev_kfid = " << t_prev << std::endl;
            cv::Mat R_cur = R_prev * R;
            cv::Mat t_cur = t_prev + R_prev * t;

            auto pose_ba_done = slam_core::pose_only_ba(R_cur, t_cur, p3d, p2d, cameraMatrix);
            
            // std::cout << "R_cur = " << R_cur << std::endl;
            // std::cout << "t_cur = " << t_cur << std::endl;
            R_cur = R_cur.t();
            t_cur = -R_cur * t_cur;
            // t_cur = slam_core::adjust_translation_magnitude(gtPoses, t_cur, idx);


            cv::Mat Rc = R_cur.clone(); cv::Mat tc = t_cur.clone();
            Rc = Rc.t();
            tc = -Rc * tc;
            double t_mag = std::abs(cv::norm(slam_types::map.keyframes[prev_triangulated_frame].t) - cv::norm(tc));
            double r_deg = slam_core::rotationAngleErrorDeg(Rc, slam_types::map.keyframes[prev_triangulated_frame].R);
            // std::cout << "R_relative_for orthonormal_checks = " << R << std::endl;
            // std::cout << "R_DEG = " << r_deg << std::endl;
            if(t_mag < slam_types::mag_filter && r_deg < slam_types::rot_filter) skip = true;
            

            // cv::Mat R_prev = slam_types::map.keyframes[prev_kfid].R.clone(); 
            // cv::Mat t_prev = slam_types::map.keyframes[prev_kfid].t.clone();
            std::vector<cv::Point3d> newPoints3D;
            std::vector<Match2D2D> newPairs;
            if(restPairs.size() > 0)
            {
                R_prev = R_prev.t(); t_prev = -R_prev * t_prev;
                auto [a, b] = slam_core::triangulate_and_filter_3d_points(
                    R_prev, t_prev, R_cur, t_cur, cameraMatrix, restPairs,
                    /*maxZ*/ 100.0, /*min_repoj_error*/ 0.1);
                newPoints3D = a;
                newPairs = b;
            }
            
            bool run_ba = false;
            int window = 0;
            skip = false;
            if(skip){
                newPoints3D.clear();
                newPairs.clear();
                if(slam_types::map.next_keyframe_id - prev_triangulated_frame > slam_types::Full_ba_window_size + 10  && slam_types::map.next_keyframe_id - slam_types::run_window > slam_types::Full_ba_window_size ){
                    run_ba = true;
                    window = slam_types::map.next_keyframe_id - slam_types::run_window; 
                    // slam_types::run_window = map.next_keyframe_id;
                }
            }
            else{
                if(slam_types::map.next_keyframe_id - slam_types::run_window >= slam_types::Full_ba_window_size){
                    run_ba = true;
                    window = slam_types::map.next_keyframe_id - slam_types::run_window; 
                    // slam_types::run_window = map.next_keyframe_id;
                }
                prev_triangulated_frame = slam_types::map.next_keyframe_id;
                prev_idx = idx;
            }
            
            std::cout << "[PnP-Loop] Frame " << idx << " triangulated-new = " << newPoints3D.size() << "\n";

            {
                std::lock_guard<std::mutex> lock(slam_types::map_mutex);
                slam_core::update_map_and_keyframe_data(
                    slam_types::map,
                    /*img_cur*/ img_cur,
                    /*R_cur*/   R_cur,
                    /*t_cur*/   t_cur,
                    /*spRes_cur*/ spRes_cur,
                    /*points3d*/ newPoints3D,
                    /*pairs*/    newPairs,
                    /*spRes_prev*/ spRes_cur,
                    /*img_prev*/  img_cur,  
                    /*map_point_obs_id*/ obsPairs,      
                    /*is_first_frame*/ false,
                    /*is_inversed*/  true
                );
            }
            tracking_lock.unlock();

            if (gtPoses.size() > idx) {
                const cv::Mat T_wi = gtPoses[idx];
                cv::Mat R_gt = T_wi(cv::Rect(0,0,3,3)).clone();
                cv::Mat t_gt = T_wi(cv::Rect(3,0,1,3)).clone();

                double rot_err = slam_core::rotationAngleErrorDeg(Rc, R_gt);
                double t_dir_err = slam_core::angleBetweenVectorsDeg(tc, t_gt);
                double t_mag_err = std::abs(cv::norm(tc) - cv::norm(t_gt));
                // std::cout << "R_cur = " << Rc << std::endl;
                std::cout << "[PnP-Loop] Frame " << idx << " | rot(deg): " << rot_err
                        << " t_dir(deg): " << t_dir_err << " t_mag(m): " << t_mag_err << "\n";

                // cv::Mat delta_R = Rc * slam_types::map.keyframes[1].R.t();
                // cv::Mat delta_t = tc - delta_R * slam_types::map.keyframes[1].t;

                // std::cout << "Delta R = " << delta_R <<std::endl;
                // std::cout << "Delta t = " << delta_t <<std::endl;
            }

            cv::Mat img1_color;
            cv::cvtColor(img_cur, img1_color, cv::COLOR_GRAY2BGR);
            const auto& kf2 = slam_types::map.keyframes[slam_types::map.next_keyframe_id - 1].sp_res;

            for (const auto& pr : newPairs) {
                float x = (float)kf2.keypoints[2 * pr.idx1];
                float y = (float)kf2.keypoints[2 * pr.idx1 + 1];
                cv::circle(img1_color, cv::Point2f(x, y), 3, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
            }

            for( auto i : obsPairs){
                float x = (float)i.p1.x;
                float y = (float)i.p1.y;
                cv::circle(img1_color, cv::Point2f(x, y), 2, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
            }
            
            cv::imshow("Inliers on Second Image", img1_color);
            cv::waitKey(1);

            if(run_ba){
                std::unique_lock<std::mutex> local_ba_lock(slam_types::local_ba_mutex);
                
                int ba_window = 0;
                if(slam_types::map.keyframes.size() >= (slam_types::Full_ba_include_past_optimized_frame_size + window)){
                    ba_window = slam_types::Full_ba_include_past_optimized_frame_size + window;
                }
                else ba_window = window;
                // slam_types::run_window = slam_types::map.next_keyframe_id - 1;
                // std::cout << "Run_window = " << slam_types::run_window << std::endl;
                // slam_types::local_ba_done = slam_core::full_ba(slam_types::map_mutex, slam_types::map, cameraMatrix, ba_window);
                
                slam_types::local_ba_window = ba_window;
                slam_types::run_window = slam_types::map.next_keyframe_id - 1;
                slam_types::kpid_to_correct.clear();
                slam_types::mpid_to_correct.clear();
                std::cout << "Run_window = " << slam_types::run_window << std::endl;
                std::cout << "ba_window = " << ba_window << std::endl;
                slam_types::local_ba_start = true;
                slam_types::cv_local_ba.notify_one();
            }

        }

    }


    void map_optimizing_thread()
    {
        auto cameraMatrix = slam_core::load_camera_matrix(slam_types::calibPath);
        while(true)
        {
            std::unique_lock<std::mutex> local_ba_lock(slam_types::local_ba_mutex);
            slam_types::cv_local_ba.wait(local_ba_lock, [] { return slam_types::local_ba_start; });
            
            slam_types::local_ba_done = slam_core::full_ba(slam_types::map_mutex, slam_types::map, cameraMatrix, slam_types::local_ba_window);
            
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            slam_types::local_ba_start = false;

            // //Apply Map Point Culling here - just make it a bad point so that It is not further optimized or used
            // int culled_points = 0;
            // std::unordered_set<int> mpids;
            // for(int i = (slam_types::run_window - slam_types::local_ba_window); i <= (slam_types::run_window - 4); ++i) 
            // {
            //     if(i == -1) continue;
            //     const auto& kf = slam_types::map.keyframes[i];
            //     for(auto& mp : kf.map_point_ids){
            //         const int earliest_kfid = slam_types::map.map_points[mp].obs.front().keyframe_id;
            //         if(earliest_kfid == i) mpids.insert(mp);
            //     } 
            //     std::cout << "keyframe to cull mpid = " << i << std::endl;
            //     std::cout << "map point to check size = " << mpids.size() << std::endl;
            // }
            // for (int id : mpids) 
            // {
            //     auto& mp = slam_types::map.map_points[id];
            //     if (mp.is_bad) continue;

            //     double total_error = 0.0;
            //     int valid_obs = 0;
            //     cv::Mat position_mat = (cv::Mat_<double>(3,1) << mp.position.x, mp.position.y, mp.position.z);
            //     for (const auto& obs : mp.obs) {
            //         const int kfid = obs.keyframe_id;
                                        
            //         const auto& kf = slam_types::map.keyframes.at(kfid);

            //         cv::Mat R1 = kf.R.clone();
            //         cv::Mat t1 = kf.t.clone();
            //         R1 = R1.t();
            //         t1 = -R1 * t1;

            //         cv::Mat camera_point = R1 * position_mat + t1;
            //         if (camera_point.at<double>(2) <= 0)
            //         {
            //             mp.is_bad = true;
            //             break;
            //         }

            //         double z = camera_point.at<double>(2);
            //         cv::Mat normalized = (cv::Mat_<double>(3,1) << camera_point.at<double>(0)/z, camera_point.at<double>(1)/z, 1.0);

            //         cv::Mat projected_mat = cameraMatrix * normalized;
            //         cv::Point2d projected(projected_mat.at<double>(0), projected_mat.at<double>(1));

            //         double error = cv::norm(projected - obs.point2D);
            //         total_error += error;
            //         valid_obs++;
            //     }
            //     if (mp.is_bad) continue;
            //     double avg_error = total_error/valid_obs;
            //     if(valid_obs < slam_types::obs_count_threshold_for_old_points || avg_error > slam_types::reprog_error_threshold_for_old_points) 
            //     {
            //         mp.is_bad = true;
            //         culled_points++;
            //     }

            // }
            // std::cout << "bad point size = " << culled_points << std::endl;


        }
    }

}