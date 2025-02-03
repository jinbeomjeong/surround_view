#include "surround_view.h"

BirdEye::BirdEye(int img_width, int img_height) {
    const float src_h_ratio = 0.08f;
    const float src_w_ratio_offset = 0.104f;
    const float dst_h_ratio = 0.6f;
    const float dst_w_ratio = 0.0463f;

    cv::Point2f src_pts[4] = {
        {img_width * (0.5f - src_w_ratio_offset), img_height * src_h_ratio},
        {img_width * (0.5f + src_w_ratio_offset), img_height * src_h_ratio},
        {static_cast<float>(img_width), static_cast<float>(img_height)},
        {0.0f, static_cast<float>(img_height)}
    };

    cv::Point2f dst_pts[4] = {
        {img_width * 0.4f, 0.0f},
        {img_width * 0.6f, 0.0f},
        {img_width * (0.5f + dst_w_ratio), img_height * dst_h_ratio},
        {img_width * (0.5f - dst_w_ratio), img_height * dst_h_ratio}
    };

    // Compute transformation matrix
    m = cv::getPerspectiveTransform(src_pts, dst_pts);
    dsize = cv::Size(img_width, cvRound(img_height * dst_h_ratio));
}

cv::Mat BirdEye::transform(const cv::Mat& img) const {
    cv::Mat dst_img;
    cv::warpPerspective(img, dst_img, m, dsize, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return dst_img;
}
