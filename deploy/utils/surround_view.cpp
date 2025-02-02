# include <opencv2/opencv.hpp>


cv::Mat convertBirdEyeView(const cv::Mat &img) {
    int h = img.rows;
    int w = img.cols;

    float src_h_ratio = 0.08f;
    float src_w_ratio_offset = 0.104f;
    float dst_h_ratio = 0.6f;
    float dst_w_ratio = 0.0463f;

    cv::Point2f src_pts[4];
    src_pts[0] = cv::Point2f(w * (0.5f - src_w_ratio_offset), h * src_h_ratio);
    src_pts[1] = cv::Point2f(w * (0.5f + src_w_ratio_offset), h * src_h_ratio);
    src_pts[2] = cv::Point2f(static_cast<float>(w), static_cast<float>(h));
    src_pts[3] = cv::Point2f(0.0f, static_cast<float>(h));

    cv::Point2f dst_pts[4];
    dst_pts[0] = cv::Point2f(w * 0.4f, 0.0f);
    dst_pts[1] = cv::Point2f(w * 0.6f, 0.0f);
    dst_pts[2] = cv::Point2f(w * (0.5f + dst_w_ratio), h * dst_h_ratio);
    dst_pts[3] = cv::Point2f(w * (0.5f - dst_w_ratio), h * dst_h_ratio);

    cv::Mat m = cv::getPerspectiveTransform(src_pts, dst_pts);

    cv::Size dsize(w, static_cast<int>(h * dst_h_ratio));

    cv::Mat dst_img;
    cv::warpPerspective(img, dst_img, m, dsize, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return dst_img;
}
