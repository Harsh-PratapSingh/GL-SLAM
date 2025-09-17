#include "core/slam_core.h"
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <iostream>
#include <sstream>
#include <fstream>

#include <memory> 

#include <g2o/solvers/eigen/linear_solver_eigen.h> 
#include <g2o/core/robust_kernel_impl.h> 

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace slam_core {

    double angleBetweenVectorsDeg(const cv::Mat& a, const cv::Mat& b) {
        cv::Mat af, bf;
        a.convertTo(af, CV_64F);
        b.convertTo(bf, CV_64F);
        double na = cv::norm(af), nb = cv::norm(bf);
        if (na < 1e-9 || nb < 1e-9) return 0.0;
        double cosang = af.dot(bf) / (na * nb);
        cosang = std::max(-1.0, std::min(1.0, cosang));
        return std::acos(cosang) * 180.0 / CV_PI;
    }

    double rotationAngleErrorDeg(const cv::Mat& R_est, const cv::Mat& R_gt) {
        cv::Mat R_err = R_gt.t() * R_est;
        double tr = std::max(-1.0, std::min(1.0, (R_err.at<double>(0,0) + R_err.at<double>(1,1) + R_err.at<double>(2,2) - 1.0) * 0.5));
        return std::acos(tr) * 180.0 / CV_PI;
    }

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

    void superpoint_lightglue_init(SuperPointTRT& sp, LightGlueTRT& lg){

        sp.setWorkspaceSizeBytes(2ULL << 30);
        sp.setMaxKeypoints(2048);
        sp.setScoreThreshold(0.1f);
        const int spH = 376;
        const int spW = 1241;
        if (!sp.init("../third_party/Superpoint_Lightglue/superpoint_2048.onnx", "superpoint_2048.engine", spH, spW)) {
            throw std::runtime_error("SuperPoint init failed");
        }
        if (!lg.init("../third_party/Superpoint_Lightglue/superpoint_lightglue.onnx", "superpoint_lightglue.engine")) {
            throw std::runtime_error("LightGlueTRT init failed");
        }
    }

    std::vector<Match2D2D> lightglue_score_filter(LightGlueTRT::Result& result,Keypt2SubpxTRT::Result& f_result, const float& score){
 
        std::vector<Match2D2D> matches;
        uint16_t lg_matches = result.matches0.size();
        matches.reserve(lg_matches);
        int x = 0;
        if(!f_result.refined_keypt0.empty()){
            for (int i = 0; i < lg_matches; ++i) {
                int j = result.matches0[i];
                if (j >= 0 && result.mscores0[i] > score) {
                    matches.push_back({
                        i, j,
                        cv::Point2d((double)f_result.refined_keypt0[2*x],     (double)f_result.refined_keypt0[2*x + 1]),
                        cv::Point2d((double)f_result.refined_keypt1[2*x],     (double)f_result.refined_keypt1[2*x + 1])
                    });
                    // if(true){
                    //     std::cout << "i = " << i << std::endl;
                    //     std::cout << "key0 = " << result.keypoints1[2*j] << std::endl;
                    //     std::cout << "r_key0 = " << f_result.refined_keypt1[2*x] << std::endl;
                    // }
                }
                if(j>=0) x++;
            }
        }
        else{
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
        }
        // matches.shrink_to_fit();
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
        essentialMat = cv::findEssentialMat(points0, points1, K, cv::USAC_MAGSAC, 0.9999, 0.5, mask);
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
        cv::Mat& f_img, std::vector<ObsPairs>& obsPairs, bool if_first_frame = false, bool if_R_t_inversed = false){

        if(if_first_frame){
            Frame first;
            first.id = map.next_keyframe_id++;
            first.img = f_img;
            cv::Mat Rf = cv::Mat::eye(3,3,CV_64F);
            cv::Mat tf = cv::Mat::zeros(3,1,CV_64F);
            Rf = Rf.t();
            tf = -Rf * tf;
            first.R = Rf;
            first.t = tf;

            first.sp_res = f_res;
            first.is_keyframe = true;

            map.keyframes[first.id] = first;
            map.keyframes[first.id].kp_to_mpid.assign(map.keyframes[first.id].sp_res.keypoints.size()/2, -1);

        }

        Frame frame;

        frame.id = map.next_keyframe_id++;
        frame.img = img;

        bool unoptimized = false;
        if(slam_types::run_window < frame.id && slam_types::run_window != -1) unoptimized = true;

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
            obs0.point2D = pr.p0;
            map.keyframes[frame.id-1].kp_to_mpid[obs0.kp_index] = mp.id;

            // Frame 1 observation
            obs1.keyframe_id = frame.id;
            obs1.kp_index = pr.idx1;
            obs1.point2D = pr.p1;
            map.keyframes[frame.id].kp_to_mpid[obs1.kp_index] = mp.id;

            mp.obs.push_back(obs0);
            mp.obs.push_back(obs1);
            map.keyframes[frame.id-1].map_point_ids.push_back(mp.id);
            map.keyframes[frame.id].map_point_ids.push_back(mp.id);

            map.map_points[mp.id] = mp;
            if(unoptimized) slam_types::mpid_to_correct.push_back(mp.id);
            
        }
        
        int obs1 =0 ;
        if(!obsPairs.empty() && !if_first_frame){
            for(const auto& m : obsPairs){
                Observation obs;
                obs.keyframe_id = frame.id;
                obs.kp_index = m.idx1;
                obs.point2D = m.p1;
                map.keyframes[frame.id].kp_to_mpid[obs.kp_index] = m.mpid;
                map.keyframes[frame.id].map_point_ids.push_back(m.mpid);
                map.map_points[m.mpid].obs.push_back(obs);
                ++obs1;
            }
            
        }
        if(unoptimized) slam_types::kpid_to_correct.push_back(frame.id);

        std::cout << "Updated " << obs1 << " for frame " << frame.id << " observations" << std::endl;
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
        Keypt2SubpxTRT::Result R;
        auto lgMatches = slam_core::lightglue_score_filter(lgRes, R, score);

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
        std::vector<Match2D2D>, std::vector<ObsPairs>, bool> 
        run_pnp(Map& map, SuperPointTRT& sp, LightGlueTRT& lg, Keypt2SubpxTRT& k2s,
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
        auto lgRes = lgRes_prev_cur;
        auto img0 = kf_prev.img.clone();
        auto img1 = img_cur.clone();
        // Keypt2SubpxTRT::Result k2sRes;
        auto k2sRes = k2s.run_Direct_Inference(lgRes, img0, img1);
        std::cout << k2sRes.refined_keypt0.size() << std::endl;
        auto all_pairs = slam_core::lightglue_score_filter(lgRes_prev_cur, k2sRes, match_thr);

        auto map_matches = slam_core::get_matches_from_previous_frames(
            lg, map, prev_kfid, window, cameraMatrix, spRes_cur, map_match_thr);
        
        // 4) Build 3D–2D (from prev’s kp_to_mpid) for PnP
        std::vector<cv::Point3d> p3d_pnp;
        std::vector<cv::Point2d> p2d_pnp;
        std::vector<int> map_point_id;          
        std::vector<int> kp_index;  
        std::vector<Match2D2D> restPairs;
        std::vector<ObsPairs> obsPairs;
        p3d_pnp.reserve(all_pairs.size());
        p2d_pnp.reserve(all_pairs.size());
        map_point_id.reserve(all_pairs.size());
        kp_index.reserve(all_pairs.size());
        restPairs.reserve(all_pairs.size());
        obsPairs.reserve(all_pairs.size());
                    
        int used3d = 0, skipped_no3d = 0;
        int x = 0;
        auto emplace = [](auto& map, auto mpid, auto& p3d_pnp,
                auto& p2d_pnp, auto& map_point_id, auto& kp_index,
                auto& used3d, auto& m, auto& obsPairs){
                p3d_pnp.emplace_back(map.map_points[mpid].position);
                p2d_pnp.emplace_back(m.p1);
                map_point_id.push_back(mpid);
                kp_index.push_back(m.idx1);
                obsPairs.push_back({mpid, m.idx1, m.p1});
                used3d++;
                // std::cout << "m.p1 =" <<m.p1<<std::endl;
                // std::cout << "position =" <<map.map_points[mpid].position<<std::endl;
                
            };

        for (const auto& m : all_pairs) {
            int mpid = kf_prev.kp_to_mpid[m.idx0];
            if(mpid > 0){
                emplace(map, mpid, p3d_pnp, p2d_pnp, map_point_id,
                kp_index, used3d, m, obsPairs);
            }else if (map_matches.find(m.idx1) != map_matches.end()){
                x++;
                mpid = map_matches[m.idx1].mpid;
                emplace(map, mpid, p3d_pnp, p2d_pnp, map_point_id,
                kp_index, used3d, m, obsPairs);
            }else{
                restPairs.push_back(m);
                skipped_no3d++;
            }
        }

        std::cout << "all pairs size = " <<all_pairs.size() << std::endl;

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
                rvec, tvec, false, 1000, 1.8, 0.999, inliers_pnp, cv::USAC_MAGSAC
            );
            
            if (!ok_pnp || (int)inliers_pnp.size() < 4) {
                std::cerr << "[PnP-Loop] PnP failed/low inliers at frame " << idx << "\n";
                skip = true;
            }
            if (get_inliner){
                std::vector<int> mapid;
                std::vector<int> keyid;
                std::vector<ObsPairs> inlier_obs;
                for (int idx : inliers_pnp) {
                    mapid.push_back(map_point_id[idx]);
                    keyid.push_back(kp_index[idx]);
                    inlier_obs.push_back(obsPairs[idx]);
                }
                // for (auto& m : inliersPairs) {
                //     auto idx = m.idx1;
                //     mapid.push_back(map_point_id[idx]);
                //     keyid.push_back(kp_index[idx]);
                //     inlier_obs.push_back(obsPairs[idx]);
                // }
                map_point_id = mapid;
                kp_index = keyid;
                obsPairs = inlier_obs;
            }
            cv::Rodrigues(rvec, R_cur);
            t_cur = tvec.clone(); 
            R_cur.convertTo(R_cur, CV_64F);
            t_cur.convertTo(t_cur, CV_64F);
            // auto [R, t, mask] = slam_core::pose_estimator(all_pairs, cameraMatrix);
            // auto inliersPairs = slam_core::pose_estimator_mask_filter(all_pairs, mask);
            // R = R.t(); t = -R * t;
            // cv::Mat R_prev = slam_types::map.keyframes[idx-1].R.clone();
            // cv::Mat t_prev = slam_types::map.keyframes[idx-1].t.clone();
            // R_cur = R * R_prev;
            // t_cur = R * t_prev + t;

            // R_cur = R_cur.t();
            // t_cur = -R_cur * t_cur;

            // t_cur = slam_core::adjust_translation_magnitude(gtPoses, t_cur, idx );

        }
        

        
        std::cout << "[PnP-Loop] Map matches = " << x << std::endl;
        std::cout << "[PnP-Loop] Matches without Map points = " << restPairs.size() << std::endl;
        std::cout << "[PnP-Loop] Frame " << idx << ": 3D-2D for PnP = " << used3d
                << " (no-3D=" << skipped_no3d << ")\n";
        std::cerr << "[PnP-Loop] PnP inliers at frame " << idx << " = " << (int)inliers_pnp.size() << " , " << map_point_id.size()  << "\n";

        return std::make_tuple(img_cur, R_cur, t_cur, spRes_cur, restPairs, obsPairs, skip);
    }   

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

    bool full_ba(std::mutex& map_mutex, Map& map, cv::Mat& cameraMatrix, int window){
        if(map.keyframes.size() < window || window == 1) return false;
        std::vector<double> camera_params;
        std::vector<double> point_params;
        std::unordered_map<int, int> kf_to_param_idx;
        int cam_param_size = 6;  // angle-axis (3) + translation (3)

        // Collect and convert camera poses
        // std::unique_lock<std::mutex> lock(map_mutex);
        std::cout << 1 << std::endl;
        int first_frame_idx = slam_types::run_window + 1 - window;
        for(int i = first_frame_idx; i < first_frame_idx + window; ++i){
            const auto& kf = map.keyframes.at(i);
            kf_to_param_idx[i] = camera_params.size() / cam_param_size;
            std::cout << "kf_index: " << i << std::endl;
            cv::Mat Rr = kf.R;
            cv::Mat Tr = kf.t;
            // Rr = Rr.t();
            // Tr = -Rr * Tr;

            cv::Mat angle_axis;
            cv::Rodrigues(Rr, angle_axis);  // Assuming R is camera-to-world; adjust if needed

            camera_params.push_back(angle_axis.at<double>(0));
            camera_params.push_back(angle_axis.at<double>(1));
            camera_params.push_back(angle_axis.at<double>(2));
            camera_params.push_back(Tr.at<double>(0));
            camera_params.push_back(Tr.at<double>(1));
            camera_params.push_back(Tr.at<double>(2));
        }
        std::cout << 2 << std::endl;
        std::unordered_map<int, int> point_to_param_idx;
        std::unordered_set<int> map_points;
        int point_param_size = 3;
        for(int i = first_frame_idx; i < first_frame_idx + window; ++i){
            const auto& kf = map.keyframes.at(i);
            for(const auto& mpid : kf.map_point_ids){
                map_points.insert(mpid);
            }
        }
        std::cout << 3 << std::endl;
        for (const auto& mpid : map_points) {
            const auto& map_point = map.map_points.at(mpid);
            if (map_point.is_bad || map_point.obs.empty()) continue;
            point_to_param_idx[mpid] = point_params.size() / point_param_size;

            point_params.push_back(map_point.position.x);
            point_params.push_back(map_point.position.y);
            point_params.push_back(map_point.position.z);
        }

        ceres::Problem problem;
        std::cout << 4 << std::endl;
        for (const auto& mpid : map_points) {
            const auto& map_point = map.map_points.at(mpid);
            if (map_point.is_bad || map_point.obs.empty()) continue;
            int point_idx = point_to_param_idx[mpid];
            // if(point.obs.size() < 3) continue;
            for (const auto& obs : map_point.obs) {
                int kfid = obs.keyframe_id;
                if(kfid < first_frame_idx || kfid > (first_frame_idx + window)) continue;

                int cam_idx = kf_to_param_idx[kfid];
                // std::cout << "cam_idx: " << cam_idx << std::endl;

                ceres::CostFunction* cost_function = ReprojectionError::Create(obs.point2D, cameraMatrix);
                ceres::LossFunction* loss_function = new ceres::CauchyLoss(1.0);  // Scale 1.0; adjust based on expected error magnitude (e.g., pixels)
                problem.AddResidualBlock(cost_function, loss_function,
                                        &camera_params[cam_idx * cam_param_size],
                                        &point_params[point_idx * point_param_size]);
            }
        }
        std::cout << 5 << std::endl;

        // Fix the first camera to remove gauge freedom
        // if (!map.keyframes.empty()) {
        //     int first_kf_id = map.keyframes.begin()->first;
        //     int first_cam_idx = kf_to_param_idx[first_kf_id];
        //     problem.SetParameterBlockConstant(&camera_params[first_cam_idx * cam_param_size]);
        // }

        {
            // const int cam_param_size = 6;
            problem.SetParameterBlockConstant(&camera_params[0 * cam_param_size]);
            const int anchor_cam_idx2 = kf_to_param_idx.at(first_frame_idx + 1);
            problem.SetParameterBlockConstant(&camera_params[(1) * cam_param_size]);
            // If only translation should be fixed, use SubsetParameterization instead:
            // std::vector<int> fixed = {3,4,5};
            // auto* subset = new ceres::SubsetParameterization(6, fixed);
            // problem.SetParameterization(&camera_params[anchor_cam_idx * cam_param_size], subset);
        }
        std::cout << 6 << std::endl;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_SCHUR;  // Or SPARSE_SCHUR for larger problems
        options.preconditioner_type = ceres::CLUSTER_JACOBI;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 30; // increase from default ~50
        options.num_threads = 8;  // Adjust to your CPU cores (e.g., std::thread::hardware_concurrency())
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        // std::cout << summary.FullReport() << std::endl;


        cv::Mat R_before = slam_types::map.keyframes[slam_types::run_window].R.clone();
        cv::Mat t_before = slam_types::map.keyframes[slam_types::run_window].t.clone();
        std::cout << 7 << std::endl;
        {
            std::lock_guard<std::mutex> tracking_lock(slam_types::tracking_mutex);
            std::lock_guard<std::mutex> lk(slam_types::map_mutex);
            for (const auto& [kfid, idx] : kf_to_param_idx) {
                double* cam = &camera_params[idx * cam_param_size];
                cv::Mat angle_axis = (cv::Mat_<double>(3,1) << cam[0], cam[1], cam[2]);
                cv::Rodrigues(angle_axis, map.keyframes[kfid].R);
                map.keyframes[kfid].t = (cv::Mat_<double>(3,1) << cam[3], cam[4], cam[5]);
                // map.keyframes[kfid].R = map.keyframes[kfid].R.t();
                // map.keyframes[kfid].t = -map.keyframes[kfid].R * map.keyframes[kfid].t;
            }
            std::cout << 8 << std::endl;
            for (const auto& [point_id, idx] : point_to_param_idx) {
                double* pt = &point_params[idx * point_param_size];
                map.map_points[point_id].position = cv::Point3d(pt[0], pt[1], pt[2]);
            }
            std::cout << 9 << std::endl;
            post_ba_map_update_for_new_keyframes(R_before, t_before);
            std::cout << 10 << std::endl;
        }
        post_ba_map_point_culling(cameraMatrix);
        std::cout << 11 << std::endl;
        return true;
    }

    cv::Mat ProjectToSO3(const cv::Mat& R_in) {
        cv::Mat U, W, Vt;
        cv::SVD::compute(R_in, W, U, Vt);
        cv::Mat R = U * Vt;
        // Enforce det(R)=+1
        double det = cv::determinant(R);
        if (det < 0.0) {
            // Flip last column of U and recompute
            U.col(2) *= -1.0;
            R = U * Vt;
        }
        return R;
    }

    void ComputeDeltaPose_SO3(const cv::Mat& Rb_in, const cv::Mat& tb,
                                    const cv::Mat& Ra_in, const cv::Mat& ta,
                                    cv::Mat& dR_out, cv::Mat& dt_out) {
        // Project before/after to SO(3) to remove numerical drift
        cv::Mat Rb = ProjectToSO3(Rb_in);
        cv::Mat Ra = ProjectToSO3(Ra_in);
        // Clean Delta R
        cv::Mat dR = Ra * Rb.t();
        dR = ProjectToSO3(dR);
        // Delta t
        cv::Mat dt = ta - dR * tb;
        dR_out = dR.clone();
        dt_out = dt.clone();
    }

    void post_ba_map_update_for_new_keyframes(cv::Mat& R_before, cv::Mat& t_before)
    {
        
        std::cout << "STOP" << std::endl;
        cv::Mat R_after = slam_types::map.keyframes[slam_types::run_window].R.clone();
        cv::Mat t_after = slam_types::map.keyframes[slam_types::run_window].t.clone();
        cv::Mat delta_R;
        cv::Mat delta_t;
        slam_core::ComputeDeltaPose_SO3(R_before, t_before, R_after, t_after, delta_R, delta_t);  // clean Delta
        // std::cout << "ba_last_R_before = " << R_before << std::endl;
        // std::cout << "ba_last_t_before = " << t_before << std::endl;
        // std::cout << "ba_last_R_after = " << R_after << std::endl;
        // std::cout << "ba_last_t_after = " << t_after << std::endl;
        // std::cout << "Delta R = " << delta_R <<std::endl;
        // std::cout << "Delta t = " << delta_t <<std::endl;
        while(!slam_types::mpid_to_correct.empty())
        {
                auto mpid_new = slam_types::mpid_to_correct.back();

                cv::Point3d point = slam_types::map.map_points[mpid_new].position;
                cv::Mat point_mat = (cv::Mat_<double>(3, 1) << point.x, point.y, point.z);
                cv::Mat updated_point_mat = delta_R * point_mat + delta_t;
                cv::Point3d updated_point(updated_point_mat.at<double>(0), 
                                        updated_point_mat.at<double>(1), 
                                        updated_point_mat.at<double>(2));
                
                slam_types::map.map_points[mpid_new].position = updated_point;

                slam_types::mpid_to_correct.pop_back();
        }
        while(!slam_types::kpid_to_correct.empty())
        {  
                auto kpid_new = slam_types::kpid_to_correct.back();
            
                cv::Mat R_new = slam_types::map.keyframes[kpid_new].R.clone();
                cv::Mat t_new = slam_types::map.keyframes[kpid_new].t.clone();
                // std::cout << "R_before = " << R_new << std::endl;
                // std::cout << "t_before = " << t_new << std::endl;
                cv::Mat Newr = delta_R * R_new;
                
                // R_new_updated = R_new_updated.t();
                cv::Mat t_new_updated = delta_R * t_new + delta_t;
                delta_R.convertTo(delta_R, CV_64F);
                R_new.convertTo(R_new, CV_64F);
                cv::Mat R_new_updated = delta_R * R_new;
                // std::cout << "R_updated = " << R_new_updated << std::endl;
                // std::cout << "t_updated = " << t_new_updated << std::endl;

                
                slam_types::map.keyframes[kpid_new].R = R_new_updated;
                slam_types::map.keyframes[kpid_new].t = t_new_updated;
                // std::cout << "kpid_new = " << kpid_new << std::endl;

                slam_types::kpid_to_correct.pop_back();
        }
        
        std::cout << "DONE" << std::endl;
    }

    void post_ba_map_point_culling(cv::Mat& cameraMatrix)
    {
        //Apply Map Point Culling here - just make it a bad point so that It is not further optimized or used
        int culled_points = 0;
        std::unordered_set<int> mpids;
        for(int i = (slam_types::run_window - slam_types::local_ba_window); i <= (slam_types::run_window - 4); ++i) 
        {
            if(i == -1) continue;
            const auto& kf = slam_types::map.keyframes[i];
            for(auto& mp : kf.map_point_ids){
                const int earliest_kfid = slam_types::map.map_points[mp].obs.front().keyframe_id;
                if(earliest_kfid == i) mpids.insert(mp);
            } 
            std::cout << "keyframe to cull mpid = " << i << std::endl;
            std::cout << "map point to check size = " << mpids.size() << std::endl;
        }
        for (int id : mpids) 
        {
            auto& mp = slam_types::map.map_points[id];
            if (mp.is_bad) continue;

            double total_error = 0.0;
            int valid_obs = 0;
            cv::Mat position_mat = (cv::Mat_<double>(3,1) << mp.position.x, mp.position.y, mp.position.z);
            for (const auto& obs : mp.obs) {
                const int kfid = obs.keyframe_id;
                                    
                const auto& kf = slam_types::map.keyframes.at(kfid);

                cv::Mat R1 = kf.R.clone();
                cv::Mat t1 = kf.t.clone();
                R1 = R1.t();
                t1 = -R1 * t1;

                cv::Mat camera_point = R1 * position_mat + t1;
                if (camera_point.at<double>(2) <= 0)
                {
                    mp.is_bad = true;
                    break;
                }

                double z = camera_point.at<double>(2);
                cv::Mat normalized = (cv::Mat_<double>(3,1) << camera_point.at<double>(0)/z, camera_point.at<double>(1)/z, 1.0);

                cv::Mat projected_mat = cameraMatrix * normalized;
                cv::Point2d projected(projected_mat.at<double>(0), projected_mat.at<double>(1));

                double error = cv::norm(projected - obs.point2D);
                total_error += error;
                valid_obs++;
            }
            if (mp.is_bad) continue;
            double avg_error = total_error/valid_obs;
            if(valid_obs < slam_types::obs_count_threshold_for_old_points || avg_error > slam_types::reprog_error_threshold_for_old_points) 
            {
                mp.is_bad = true;
                culled_points++;
            }

        }
        std::cout << "bad point size = " << culled_points << std::endl;
    }

}