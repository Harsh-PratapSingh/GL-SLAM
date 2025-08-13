#include "visualization/visualization.h"
#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>

namespace slam_visualization {
    // void draw_frustum(float scale, float r, float g, float b) {
    //     glLineWidth(2.0f);
    //     glBegin(GL_LINES);
    //     glColor3f(r, g, b);
    //     float z = 0.8f * scale, s = 0.5f * scale;
    //     glVertex3f(-s, -s, z); glVertex3f(s, -s, z);
    //     glVertex3f(s, -s, z);  glVertex3f(s, s, z);
    //     glVertex3f(s, s, z);   glVertex3f(-s, s, z);
    //     glVertex3f(-s, s, z);  glVertex3f(-s, -s, z);
    //     glVertex3f(0, 0, 0);   glVertex3f(-s, -s, z);
    //     glVertex3f(0, 0, 0);   glVertex3f(s, -s, z);
    //     glVertex3f(0, 0, 0);   glVertex3f(s, s, z);
    //     glVertex3f(0, 0, 0);   glVertex3f(-s, s, z);
    //     glEnd();
    // }

    // void visualize_poses(const std::vector<cv::Mat>& Rs_est, const std::vector<cv::Mat>& Ts_est, 
    //                     const std::vector<cv::Mat>& Rs_gt, const std::vector<cv::Mat>& Ts_gt,
    //                     const std::vector<Point3D>& points3D) {
    //     pangolin::CreateWindowAndBind("Pose Visualization", 1024, 768);
    //     glEnable(GL_DEPTH_TEST);
    //     pangolin::OpenGlRenderState s_cam(
    //         pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
    //         pangolin::ModelViewLookAt(0, -10, -20, 0, 0, 0, 0.0, -1.0, 0.0)
    //     );
    //     pangolin::View& d_cam = pangolin::CreateDisplay()
    //         .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
    //         .SetHandler(new pangolin::Handler3D(s_cam));

    //     while (!pangolin::ShouldQuit()) {
    //         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //         d_cam.Activate(s_cam);

    //         for (size_t i = 0; i < Rs_est.size(); ++i) {
    //             float T_est_mat[16] = {
    //                 float(Rs_est[i].at<double>(0,0)), float(Rs_est[i].at<double>(1,0)), float(Rs_est[i].at<double>(2,0)), 0,
    //                 float(Rs_est[i].at<double>(0,1)), float(Rs_est[i].at<double>(1,1)), float(Rs_est[i].at<double>(2,1)), 0,
    //                 float(Rs_est[i].at<double>(0,2)), float(Rs_est[i].at<double>(1,2)), float(Rs_est[i].at<double>(2,2)), 0,
    //                 float(Ts_est[i].at<double>(0)), float(Ts_est[i].at<double>(1)), float(Ts_est[i].at<double>(2)), 1
    //             };
    //             glPushMatrix();
    //             glMultMatrixf(T_est_mat);
    //             draw_frustum(1.0f, 1.0f, 0.0f, 0.0f); // Red
    //             glPopMatrix();
    //         }

    //         for (size_t i = 0; i < Rs_gt.size(); ++i) {
    //             float T_gt_mat[16] = {
    //                 float(Rs_gt[i].at<double>(0,0)), float(Rs_gt[i].at<double>(1,0)), float(Rs_gt[i].at<double>(2,0)), 0,
    //                 float(Rs_gt[i].at<double>(0,1)), float(Rs_gt[i].at<double>(1,1)), float(Rs_gt[i].at<double>(2,1)), 0,
    //                 float(Rs_gt[i].at<double>(0,2)), float(Rs_gt[i].at<double>(1,2)), float(Rs_gt[i].at<double>(2,2)), 0,
    //                 float(Ts_gt[i].at<double>(0)), float(Ts_gt[i].at<double>(1)), float(Ts_gt[i].at<double>(2)), 1
    //             };
    //             glPushMatrix();
    //             glMultMatrixf(T_gt_mat);
    //             draw_frustum(1.0f, 0.0f, 1.0f, 0.0f); // Green
    //             glPopMatrix();
    //         }

    //         glPointSize(2.0f);
    //         glBegin(GL_POINTS);
    //         glColor3f(0.0f, 1.0f, 0.0f); // Green
    //         for (const auto& pt : points3D) {
    //             glVertex3f(pt.position.x, pt.position.y, pt.position.z);
    //         }
    //         glEnd();

    //         pangolin::FinishFrame();
    //     }
    // }

    // void visualize_optical_flow(const cv::Mat& img_current,
    //                            const std::vector<cv::Point2f>& points_prev,
    //                            const std::vector<cv::Point2f>& points_current,
    //                            const cv::Mat& mask, int frame_idx,
    //                            const std::vector<cv::Point2f>& projected_points) {
    //     cv::Mat img_color;
    //     cv::cvtColor(img_current, img_color, cv::COLOR_GRAY2BGR);
    //     for (size_t i = 0; i < points_current.size() && i < points_prev.size(); ++i) {
    //         if (mask.at<uchar>(i)) {
    //             cv::Point2f pt_current = points_current[i];
    //             cv::Point2f pt_prev = points_prev[i];
    //             cv::line(img_color, pt_current, pt_prev, cv::Scalar(0, 255, 0), 1);
    //             cv::circle(img_color, pt_current, 3, cv::Scalar(0, 0, 255), -1); // Red: observed
    //         }
    //     }
    //     for (const auto& pt_proj : projected_points) {
    //         cv::circle(img_color, pt_proj, 3, cv::Scalar(255, 0, 0), -1); // Blue: projected
    //     }
    //     std::string window_name = "Optical Flow ";
    //     cv::imshow(window_name, img_color);
    //     cv::waitKey(1);
    // }
}