#include "core/lightglue.h"
#include "core/superpoint.h"
#include "core/slam_core.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cmath>

static void drawSuperPointKpts(cv::Mat& imgBgr, const std::vector<float>& kptsXY, int N,
                               const cv::Scalar& color = cv::Scalar(0, 255, 0),
                               int radius = 2, int thickness = 1) {
    for (int i = 0; i < N; ++i) {
        float x = kptsXY[size_t(i)*2 + 0];
        float y = kptsXY[size_t(i)*2 + 1];
        cv::circle(imgBgr, cv::Point2f(x, y), radius, color, thickness, cv::LINE_AA);
    }
}

static cv::Mat concatHorizontalSameSize(const cv::Mat& left, const cv::Mat& right) {
    cv::Mat out(left.rows, left.cols + right.cols, left.type());
    left.copyTo(out(cv::Rect(0, 0, left.cols, left.rows)));
    right.copyTo(out(cv::Rect(left.cols, 0, right.cols, right.rows)));
    return out;
}

static void drawLightGlueMatches(cv::Mat& vis,
                                 int img0Width,
                                 const std::vector<float>& kpts0,
                                 const std::vector<float>& kpts1,
                                 const std::vector<int64_t>& matches0,
                                 const std::vector<float>& mscores0,
                                 int N0, int N1,
                                 int maxToDraw = 500,
                                 float minScore = 0.0f) {
    struct M { int i; int j; float s; };
    std::vector<M> mlist;
    mlist.reserve(N0);
    for (int i = 0; i < N0; ++i) {
        int j = static_cast<int>(matches0[i]);
        if (j >= 0 && j < N1) {
            float s = mscores0[i];
            if (s >= minScore) {
                mlist.push_back({i, j, s});
            }
        }
    }
    std::sort(mlist.begin(), mlist.end(), [](const M& a, const M& b){ return a.s > b.s; });
    if ((int)mlist.size() > maxToDraw) mlist.resize(maxToDraw);

    for (const auto& m : mlist) {
        float x0 = kpts0[size_t(m.i)*2 + 0];
        float y0 = kpts0[size_t(m.i)*2 + 1];
        float x1 = kpts1[size_t(m.j)*2 + 0] + img0Width;
        float y1 = kpts1[size_t(m.j)*2 + 1];

        float s = std::max(0.f, std::min(1.f, m.s));
        cv::Scalar color = cv::Scalar(0, 255*s, 255*(1.0f - s));

        cv::circle(vis, cv::Point2f(x0, y0), 2, color, -1, cv::LINE_AA);
        cv::circle(vis, cv::Point2f(x1, y1), 2, color, -1, cv::LINE_AA);
        cv::line(vis, cv::Point2f(x0, y0), cv::Point2f(x1, y1), color, 1, cv::LINE_AA);
    }
}


int main() {
    try {
        SuperPointTRT sp;
        LightGlueTRT lg;
        slam_core::superpoint_lightglue_init(sp, lg);
        const int spH = 376;
        const int spW = 1241;
        cv::Mat img0 = cv::imread("temp3.png", cv::IMREAD_GRAYSCALE);
        cv::Mat img1 = cv::imread("temp4.png", cv::IMREAD_GRAYSCALE);
        if (img0.empty() || img1.empty()) {
            throw std::runtime_error("Failed to load image0.png or image1.png");
        }
        SuperPointTRT::Result spRes0, spRes1;
        spRes0 = sp.runInference(img0, img0.rows, img0.cols);
        spRes1 = sp.runInference(img1, img1.rows, img1.cols);
        std::cout << "Image0: valid keypoints = " << spRes0.numValid << std::endl;
        std::cout << "Image1: valid keypoints = " << spRes1.numValid << std::endl;
        const int maxKpts = 2048;
        const int N0 = std::min(spRes0.numValid, maxKpts);
        const int N1 = std::min(spRes1.numValid, maxKpts);
        
        LightGlueTRT::Result lgRes;
        lgRes = lg.run_Direct_Inference(spRes0, spRes1);
        int nMatches = 0;
        for (int i = 0; i < N0; ++i) {
            if (lgRes.matches0[i] >= 0 && lgRes.mscores0[i] > 0.9) ++nMatches;
        }
        std::cout << "LightGlue matches: " << nMatches << " (of " << N0 << " keypoints in image0)" << std::endl;

        // Visualization without normalization/denormalization
        cv::Mat img0_vis, img1_vis;
        cv::cvtColor(img0, img0_vis, cv::COLOR_GRAY2BGR);
        cv::cvtColor(img1, img1_vis, cv::COLOR_GRAY2BGR);
        std::vector<float> kpts0_vis, kpts1_vis;
        kpts0_vis.resize(size_t(N0) * 2);
        kpts1_vis.resize(size_t(N1) * 2);
        for (int i=0; i < N0; ++i) {
            kpts0_vis[size_t(i)*2 + 0] = static_cast<float>(spRes0.keypoints[size_t(i)*2 + 0]);
            kpts0_vis[size_t(i)*2 + 1] = static_cast<float>(spRes0.keypoints[size_t(i)*2 + 1]);
        }
        for (int i=0; i < N1; ++i) {
            kpts1_vis[size_t(i)*2 + 0] = static_cast<float>(spRes1.keypoints[size_t(i)*2 + 0]);
            kpts1_vis[size_t(i)*2 + 1] = static_cast<float>(spRes1.keypoints[size_t(i)*2 + 1]);
        }
        drawSuperPointKpts(img0_vis, kpts0_vis, N0, cv::Scalar(0, 255, 0), 2, 1);
        drawSuperPointKpts(img1_vis, kpts1_vis, N1, cv::Scalar(0, 255, 0), 2, 1);
        cv::Mat concat = concatHorizontalSameSize(img0_vis, img1_vis);
        drawLightGlueMatches(concat, img0_vis.cols, kpts0_vis, kpts1_vis, lgRes.matches0, lgRes.mscores0, N0, N1, 1000, 0.99f);
        cv::imshow("SuperPoint+LightGlue Matches", concat);
        cv::imwrite("matches_vis.png", concat);
        cv::waitKey(0);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl;
        return -1;
    }
}
