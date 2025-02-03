#ifndef SURROUND_VIEW_H
#define SURROUND_VIEW_H

# include <opencv2/opencv.hpp>

class BirdEye {
private:
    cv::Mat m;
    cv::Size dsize;

public:
    BirdEye(int img_width, int img_height);
    cv::Mat transform(const cv::Mat& img) const;
};

#endif

