#include <opencv2/opencv.hpp>
#include <thread>
#include "surround_view.h"


int main() {
    cv::String const win_name = "windows";
    cv::Mat frame;

    cv::VideoCapture cap("test.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file!" << std::endl;
        return -1;
    }

    const int img_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int img_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const int desired_frame_time_ms = 33; // 약 30FPS (1000ms / 30)

    cv::namedWindow(win_name, cv::WINDOW_NORMAL);

    BirdEye transform(img_width, img_height);

    while (true) {
        auto frame_start = std::chrono::steady_clock::now();
        cap >> frame;

        if (frame.empty()) break; // 프레임이 비어있으면 종료

        cv::Mat dst_img_1 = transform.transform(frame);
        cv::Mat dst_img_2 = transform.transform(frame); // 복사
        cv::Mat dst_img_3 = transform.transform(frame);
        cv::Mat dst_img_4 = transform.transform(frame);

        cv::imshow(win_name, frame);

        if (cv::waitKey(1) == 27) break;

        auto frame_end = std::chrono::steady_clock::now();
        int elapsed_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>
            (frame_end - frame_start).count());

        int delay = std::max(desired_frame_time_ms - elapsed_ms, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));

        auto program_end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(program_end - frame_start).count();
        std::cout << "Total execution time: " << duration << " ms" << std::endl;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

