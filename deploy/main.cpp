#include <opencv2/opencv.hpp>
#include <iostream>
#include "utils/surround_view.h"


int main(){
    cv::String const win_name = "windows";
    cv::Mat frame;
    const int desired_frame_time_ms = 33;

    cv::VideoCapture cap("test.mp4");
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);

    while (cap.isOpened()) {
        auto frame_start = std::chrono::steady_clock::now();
        cap >> frame;

        if (frame.empty()) break;

        cv::Mat dst_img_1 = convertBirdEyeView(frame);
        cv::Mat dst_img_2 = convertBirdEyeView(frame);
        cv::Mat dst_img_3 = convertBirdEyeView(frame);
        cv::Mat dst_img_4 = convertBirdEyeView(frame);

        cv::imshow(win_name, frame);

        if (cv::waitKey(1) >= 0) break;

        auto frame_end = std::chrono::steady_clock::now();
        int elapsed_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>
            (frame_end - frame_start).count());
        int delay = desired_frame_time_ms - elapsed_ms;

        if (delay > 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay));

        auto program_end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(program_end - frame_start).count();
        std::cout << "Total execution time: " << duration << " ms" << std::endl;


    }
    return 0;

}
