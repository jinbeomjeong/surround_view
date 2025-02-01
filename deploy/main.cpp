#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

int main()
{
    string const imagePath = "img.jpg";
    cv::Mat const image = cv::imread(imagePath, cv::IMREAD_COLOR);

    // 이미지가 제대로 읽혔는지 확인
    if (image.empty()) {
        cerr << "이미지를 불러올 수 없습니다: " << imagePath << endl;
        return -1;
    }

    cv::namedWindow("img", cv::WINDOW_AUTOSIZE);
    cv::imshow("img", image);

    // 키 입력 대기 (키를 누르면 프로그램 종료)
    cv::waitKey(0);

    return 0;
}
