#include "visualization/visualization.h"
#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <iostream>

namespace slam_visualization {
    // Static variables to persist Pangolin context
    static bool initialized = false;
    static pangolin::OpenGlRenderState s_cam;
    static pangolin::View* d_cam = nullptr;
    static std::mutex viz_mutex;

    void initialize_pangolin() {
        std::lock_guard<std::mutex> lock(viz_mutex);
        if (!initialized) {
            try {
                std::cout << "Initializing Pangolin window...\n";
                pangolin::CreateWindowAndBind("Pose Visualization", 1024, 768);
                glEnable(GL_DEPTH_TEST);
                s_cam = pangolin::OpenGlRenderState(
                    pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
                    pangolin::ModelViewLookAt(0, -10, -20, 0, 0, 0, 0.0, -1.0, 0.0)
                );
                d_cam = &pangolin::CreateDisplay()
                    .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                    .SetHandler(new pangolin::Handler3D(s_cam));
                initialized = true;
                std::cout << "Pangolin window initialized successfully.\n";
            } catch (const std::exception& e) {
                std::cerr << "Error initializing Pangolin: " << e.what() << "\n";
                initialized = false;
            }
        }
    }

    void draw_frustum(float scale, float r, float g, float b) {
        glLineWidth(2.0f);
        glBegin(GL_LINES);
        glColor3f(r, g, b);
        float z = 0.8f * scale, s = 0.5f * scale;
        glVertex3f(-s, -s, z); glVertex3f(s, -s, z);
        glVertex3f(s, -s, z);  glVertex3f(s, s, z);
        glVertex3f(s, s, z);   glVertex3f(-s, s, z);
        glVertex3f(-s, s, z);  glVertex3f(-s, -s, z);
        glVertex3f(0, 0, 0);   glVertex3f(-s, -s, z);
        glVertex3f(0, 0, 0);   glVertex3f(s, -s, z);
        glVertex3f(0, 0, 0);   glVertex3f(s, s, z);
        glVertex3f(0, 0, 0);   glVertex3f(-s, s, z);
        glEnd();
    }

    void visualize_poses(const std::vector<cv::Mat>& Rs_est, const std::vector<cv::Mat>& Ts_est, 
                        const std::vector<cv::Mat>& Rs_gt, const std::vector<cv::Mat>& Ts_gt,
                        const std::vector<Point3D>& points3D, bool interactive) {
        std::lock_guard<std::mutex> lock(viz_mutex);
        if (!initialized) {
            std::cerr << "Pangolin not initialized, skipping visualization.\n";
            return;
        }

        // Validate input data
        std::cout << "Visualizing poses: Rs_est=" << Rs_est.size() << ", Ts_est=" << Ts_est.size() 
                  << ", Rs_gt=" << Rs_gt.size() << ", Ts_gt=" << Ts_gt.size() 
                  << ", points3D=" << points3D.size() << "\n";
        for (size_t i = 0; i < Rs_est.size(); ++i) {
            if (Rs_est[i].empty() || Ts_est[i].empty() || Rs_est[i].type() != CV_64F || Ts_est[i].type() != CV_64F) {
                std::cerr << "Invalid Rs_est or Ts_est at index " << i << "\n";
                return;
            }
            // Check for NaN
            for (int r = 0; r < Rs_est[i].rows; ++r) {
                for (int c = 0; c < Rs_est[i].cols; ++c) {
                    if (std::isnan(Rs_est[i].at<double>(r, c))) {
                        std::cerr << "NaN in Rs_est[" << i << "] at (" << r << "," << c << ")\n";
                        return;
                    }
                }
            }
            for (int r = 0; r < Ts_est[i].rows; ++r) {
                if (std::isnan(Ts_est[i].at<double>(r, 0))) {
                    std::cerr << "NaN in Ts_est[" << i << "] at (" << r << ",0)\n";
                    return;
                }
            }
        }
        for (const auto& pt : points3D) {
            if (std::isnan(pt.position.x) || std::isnan(pt.position.y) || std::isnan(pt.position.z)) {
                std::cerr << "NaN in point3D position\n";
                return;
            }
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam->Activate(s_cam);

        // Draw estimated poses (red)
        for (size_t i = 0; i < Rs_est.size(); ++i) {
            float T_est_mat[16] = {
                float(Rs_est[i].at<double>(0,0)), float(Rs_est[i].at<double>(1,0)), float(Rs_est[i].at<double>(2,0)), 0,
                float(Rs_est[i].at<double>(0,1)), float(Rs_est[i].at<double>(1,1)), float(Rs_est[i].at<double>(2,1)), 0,
                float(Rs_est[i].at<double>(0,2)), float(Rs_est[i].at<double>(1,2)), float(Rs_est[i].at<double>(2,2)), 0,
                float(Ts_est[i].at<double>(0)), float(Ts_est[i].at<double>(1)), float(Ts_est[i].at<double>(2)), 1
            };
            glPushMatrix();
            glMultMatrixf(T_est_mat);
            draw_frustum(1.0f, 1.0f, 0.0f, 0.0f); // Red
            glPopMatrix();
        }

        // Draw ground truth poses (green)
        for (size_t i = 0; i < Rs_gt.size(); ++i) {
            float T_gt_mat[16] = {
                float(Rs_gt[i].at<double>(0,0)), float(Rs_gt[i].at<double>(1,0)), float(Rs_gt[i].at<double>(2,0)), 0,
                float(Rs_gt[i].at<double>(0,1)), float(Rs_gt[i].at<double>(1,1)), float(Rs_gt[i].at<double>(2,1)), 0,
                float(Rs_gt[i].at<double>(0,2)), float(Rs_gt[i].at<double>(1,2)), float(Rs_gt[i].at<double>(2,2)), 0,
                float(Ts_gt[i].at<double>(0)), float(Ts_gt[i].at<double>(1)), float(Ts_gt[i].at<double>(2)), 1
            };
            glPushMatrix();
            glMultMatrixf(T_gt_mat);
            draw_frustum(1.0f, 0.0f, 1.0f, 0.0f); // Green
            glPopMatrix();
        }

        // Draw 3D points (green)
        glPointSize(2.0f);
        glBegin(GL_POINTS);
        glColor3f(0.0f, 1.0f, 0.0f); // Green
        for (const auto& pt : points3D) {
            glVertex3f(pt.position.x, pt.position.y, pt.position.z);
        }
        glEnd();

        pangolin::FinishFrame();
        std::cout << "Rendered Pangolin frame.\n";

        if (interactive) {
            std::cout << "Entering interactive visualization mode...\n";
            while (!pangolin::ShouldQuit()) {
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                d_cam->Activate(s_cam);

                for (size_t i = 0; i < Rs_est.size(); ++i) {
                    float T_est_mat[16] = {
                        float(Rs_est[i].at<double>(0,0)), float(Rs_est[i].at<double>(1,0)), float(Rs_est[i].at<double>(2,0)), 0,
                        float(Rs_est[i].at<double>(0,1)), float(Rs_est[i].at<double>(1,1)), float(Rs_est[i].at<double>(2,1)), 0,
                        float(Rs_est[i].at<double>(0,2)), float(Rs_est[i].at<double>(1,2)), float(Rs_est[i].at<double>(2,2)), 0,
                        float(Ts_est[i].at<double>(0)), float(Ts_est[i].at<double>(1)), float(Ts_est[i].at<double>(2)), 1
                    };
                    glPushMatrix();
                    glMultMatrixf(T_est_mat);
                    draw_frustum(1.0f, 1.0f, 0.0f, 0.0f);
                    glPopMatrix();
                }
                for (size_t i = 0; i < Rs_gt.size(); ++i) {
                    float T_gt_mat[16] = {
                        float(Rs_gt[i].at<double>(0,0)), float(Rs_gt[i].at<double>(1,0)), float(Rs_gt[i].at<double>(2,0)), 0,
                        float(Rs_gt[i].at<double>(0,1)), float(Rs_gt[i].at<double>(1,1)), float(Rs_gt[i].at<double>(2,1)), 0,
                        float(Rs_gt[i].at<double>(0,2)), float(Rs_gt[i].at<double>(1,2)), float(Rs_gt[i].at<double>(2,2)), 0,
                        float(Ts_gt[i].at<double>(0)), float(Ts_gt[i].at<double>(1)), float(Ts_gt[i].at<double>(2)), 1
                    };
                    glPushMatrix();
                    glMultMatrixf(T_gt_mat);
                    draw_frustum(1.0f, 0.0f, 1.0f, 0.0f);
                    glPopMatrix();
                }
                glPointSize(2.0f);
                glBegin(GL_POINTS);
                glColor3f(0.0f, 1.0f, 0.0f);
                for (const auto& pt : points3D) {
                    glVertex3f(pt.position.x, pt.position.y, pt.position.z);
                }
                glEnd();

                pangolin::FinishFrame();
            }
            std::cout << "Exiting interactive visualization mode.\n";
        }
    }

    void visualize_optical_flow(const cv::Mat& img_current,
                               const std::vector<cv::Point2f>& points_prev,
                               const std::vector<cv::Point2f>& points_current,
                               const cv::Mat& mask, int frame_idx,
                               const std::vector<cv::Point2f>& projected_points) {
        if (img_current.empty()) {
            std::cerr << "Empty image in visualize_optical_flow for frame " << frame_idx << "\n";
            return;
        }
        cv::Mat img_color;
        cv::cvtColor(img_current, img_color, cv::COLOR_GRAY2BGR);
        for (size_t i = 0; i < points_current.size() && i < points_prev.size(); ++i) {
            if (mask.at<uchar>(i)) {
                cv::Point2f pt_current = points_current[i];
                cv::Point2f pt_prev = points_prev[i];
                cv::line(img_color, pt_current, pt_prev, cv::Scalar(0, 255, 0), 1);
                cv::circle(img_color, pt_current, 3, cv::Scalar(0, 0, 255), -1); // Red: observed
            }
        }
        for (const auto& pt_proj : projected_points) {
            cv::circle(img_color, pt_proj, 3, cv::Scalar(255, 0, 0), -1); // Blue: projected
        }
        std::string window_name = "Optical Flow ";
        cv::imshow(window_name, img_color);
        cv::waitKey(0);
        std::cout << "Rendered optical flow for frame " << frame_idx << "\n";
    }
}