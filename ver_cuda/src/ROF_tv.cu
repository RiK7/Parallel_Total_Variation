#include <iostream>
#include <utility>
#include <ctime>
#include <cstdlib>
#include <cstdio>

#include <opencv2/opencv.hpp>

#include "ROFtv.cuh"


using namespace std;
using namespace cv;
using namespace tv;


int main(int argc, char** argv)
{
    
    const string keys = 
    "{help h ? |      | print this message    }"
    "{@imag    |      | imput image           }"
    "{N steps  | 100  | total Time step       integer}"
    "{s sigma2 | 1e-3 | value of sigma square float}"
    "{d deltaT | 1e-6 | time step             float}"
    "{D depth  | 8    | image depth           int}"
    "{e eps    | 1e-4 | epsilon in minmod     float}";

    CommandLineParser parser(argc, argv, keys);
    if(parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    const string fileName = parser.get<string>("@imag");
    int N         = parser.get<int>("N"),
        D         = parser.get<int>("D");
    float sigma2  = parser.get<float>("s");
    float deltaT  = parser.get<float>("d");
    float eps     = parser.get<float>("e");

    //Mat u = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE/*| CV_LOAD_IMAGE_ANYDEPTH*/);
    Mat u = imread(fileName);
    
    if(!parser.check() || u.dims==0)
    {
        parser.printErrors();
        parser.printMessage();
        return -1;
    }

    cout << "file  \t" << fileName   << endl 
         << "N     \t" << N          << endl
         << "sigma2\t" << sigma2     << endl
         << "deltaT\t" << deltaT     << endl
         << "depth \t" << D          << endl
         << "eps   \t" << eps       << endl;

//    vector<int> para;
    vector<int> para;
//    Mat temp, view;
    Mat temp, view;
//    para.push_back(9);
    para.push_back(9);
//
//    Mat result(u.size(), CV_32F);
    Mat YUV;
    cvtColor(u,YUV,COLOR_BGR2YUV);
    vector<Mat> spl;
    split(YUV,spl);

    Mat result(spl[0].size(), CV_32F);
    
//    u.convertTo(temp, CV_32F);
    spl[0].convertTo(temp, CV_32F);
//    temp.copyTo(u);
    temp.copyTo(spl[0]);
//    u/=pow(2,D);
    spl[0] /= pow(2,D);
//
//    result = move(ROFtv(u, N, sigma2, deltaT, eps));
    result = ROFtv(spl[0],N,sigma2,deltaT,eps);
//
//    result.convertTo(temp, CV_8U, 255);
    result.convertTo(spl[0], CV_8U, 255);
    merge(spl,view);
    cvtColor(view,view,COLOR_YUV2BGR);
//
//    char outputFileName[256];
    char outputFileName[256];
//    sprintf(outputFileName, "stepN%d_SIGMA%e_dT%e_EPSILON%e.png", N, sigma2, deltaT, eps);
    sprintf(outputFileName, "stepN%d_SIGMA%e_dT%e_EPSILON%e.png", N, sigma2, deltaT, eps);
//    imwrite(outputFileName,temp, para);
    imwrite(outputFileName,view, para);
    

    return 0;
}


