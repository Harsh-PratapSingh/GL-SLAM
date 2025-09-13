#include "visualization/visualization.h"

namespace slam_visualization {
    
    static inline pangolin::OpenGlMatrix CvToGl(const cv::Mat& T) {
        pangolin::OpenGlMatrix M;
        // Pangolin is column-major; fill in column-major order
        const double* t = T.ptr<double>(0);
        M.m[0]  = t[0];  M.m[4]  = t[1];  M.m[8]  = t[2];   M.m[12] = t[3];
        M.m[1]  = t[4];  M.m[5]  = t[5];  M.m[9]  = t[6];   M.m[13] = t[7];
        M.m[2]  = t[8];  M.m[6]  = t[9];  M.m[10] = t[10];  M.m[14] = t[11];
        M.m[3]  = t[12]; M.m[7]  = t[13]; M.m[11] = t[14];  M.m[15] = t[15];
        return M;
    }
    
    static inline cv::Mat GlToCv(const pangolin::OpenGlMatrix& M) { 
        cv::Mat T = cv::Mat::eye(4,4,CV_64F);
        double* t = T.ptr<double>(0);
        // Convert column-major back to row-major
        t[0]  = M.m[0];  t[1]  = M.m[4];  t[2]  = M.m[8];   t[3]  = M.m[12];
        t[4]  = M.m[1];  t[5]  = M.m[5];  t[6]  = M.m[9];   t[7]  = M.m[13];
        t[8]  = M.m[2];  t[9]  = M.m[6];  t[10] = M.m[10];  t[11] = M.m[14];
        t[12] = M.m[3];  t[13] = M.m[7];  t[14] = M.m[11];  t[15] = M.m[15];
        return T;
    }

    void visualize_map_loop(Map& map, std::mutex& map_mutex) {
        pangolin::CreateWindowAndBind("SLAM Viewer", 1024, 768);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024,768,500,500,512,389,0.1,1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
        );
        auto& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

        bool follow = true;
        cv::Mat Tcw_prev = cv::Mat::eye(4,4,CV_64F);
        cv::Mat Mrel     = cv::Mat::eye(4,4,CV_64F);
        bool rel_init    = false;

        while (!pangolin::ShouldQuit()) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // 1) Fetch latest pose snapshot under lock
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_64F);
            {
                std::lock_guard<std::mutex> lock(map_mutex);
                if (!map.keyframes.empty()) {
                    auto it = std::max_element(map.keyframes.begin(), map.keyframes.end(),
                                            [](const auto& a, const auto& b){ return a.first < b.first; });
                    const auto& kf = it->second;
                    cv::Mat Rcw = kf.R.t();
                    cv::Mat tcw = -Rcw * kf.t;
                    Rcw.copyTo(Tcw(cv::Rect(0,0,3,3)));
                    tcw.copyTo(Tcw(cv::Rect(3,0,1,3)));
                }
            }

            // 2) Relative-follow logic
            if (follow) {
                cv::Mat MV = GlToCv(s_cam.GetModelViewMatrix());
                if (!rel_init) { Mrel = MV * Tcw.inv(); Tcw_prev = Tcw.clone(); rel_init = true; }
                else           { Mrel = MV * Tcw_prev.inv(); }
                s_cam.SetModelViewMatrix(CvToGl(Mrel * Tcw));
                Tcw_prev = Tcw.clone();
            }

            // 3) Draw scene using a snapshot under lock
            d_cam.Activate(s_cam);
            glClearColor(1,1,1,1);

            {
                std::lock_guard<std::mutex> lock(map_mutex);

                glPointSize(1);
                glBegin(GL_POINTS);
                glColor3f(0,0,1);
                for (const auto& kv : map.map_points)
                    if (!kv.second.is_bad)
                        glVertex3f(kv.second.position.x, kv.second.position.y, kv.second.position.z);
                glEnd();

                for (const auto& kv : map.keyframes) {
                    const auto& kf = kv.second;
                    cv::Mat Rcw = kf.R, tcw = kf.t;
                    cv::Mat T = cv::Mat::eye(4,4,CV_64F);
                    Rcw.copyTo(T(cv::Rect(0,0,3,3)));
                    tcw.copyTo(T(cv::Rect(3,0,1,3)));

                    float sz = 1.0f;
                    if(kf.id == 0){
                        sz = 2.0f;
                    }
                    
                    glLineWidth(3);
                    glBegin(GL_LINES);
                    double ox = T.at<double>(0,3), oy = T.at<double>(1,3), oz = T.at<double>(2,3);
                    double x0 = T.at<double>(0,0), x1 = T.at<double>(1,0), x2 = T.at<double>(2,0);
                    double y0 = T.at<double>(0,1), y1 = T.at<double>(1,1), y2 = T.at<double>(2,1);
                    double z0 = T.at<double>(0,2), z1 = T.at<double>(1,2), z2 = T.at<double>(2,2);
                    glColor3f(1,0,0); glVertex3f(ox,oy,oz); glVertex3f(ox+sz*x0, oy+sz*x1, oz+sz*x2);
                    glColor3f(0,1,0); glVertex3f(ox,oy,oz); glVertex3f(ox+sz*y0, oy+sz*y1, oz+sz*y2);
                    glColor3f(0,0,1); glVertex3f(ox,oy,oz); glVertex3f(ox+sz*z0, oy+sz*z1, oz+sz*z2);
                    glEnd();
                }
            }

            pangolin::FinishFrame();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }



}