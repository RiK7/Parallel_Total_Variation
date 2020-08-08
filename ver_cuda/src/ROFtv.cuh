#include <opencv2/opencv.hpp>

namespace tv
{
    cv::Mat ROFtv(const cv::Mat& ,
                        int N = 300,
                        float sigma = 0.002,
                        float deltaT = 1e-6,
                        float eps = 1e-5);
}
