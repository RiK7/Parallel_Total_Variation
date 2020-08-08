#include <iostream>
#include <utility>
#include <ctime>
#include <cstdlib>
#include <cstdio>

#include <opencv2/opencv.hpp>

#include <ROFtv/ROFtv.hpp>


using namespace std;
using namespace cv;
using namespace tv;


int main(int argc, char** argv)
{
    
    const string keys = 
    "{help h ? |      | print this message    }"
    "{@video   |      | imput video           }"
    "{N steps  | 100  | total Time step       integer}"
    "{s sigma2 | 1e-3 | value of sigma square float}"
    "{d deltaT | 1e-6 | time step             float}"
    "{L limit  | -1   | limited max frame number}"
    "{D depth  | 8    | image depth           int}"
    "{e eps    | 1e-4 | epsilon in minmod     float}"
    "{o outName| out  | output name           }"
    "{f format | mp4  | output format         }";

    CommandLineParser parser(argc, argv, keys);
    if(parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    const string fileName = parser.get<string>("@video");
    const string outName  = parser.get<string>("o")+"."+parser.get<string>("f");

    int N         = parser.get<int>("N"),
        D         = parser.get<int>("D"),
        L         = parser.get<int>("L");
    float sigma2  = parser.get<float>("s");
    float deltaT  = parser.get<float>("d");
    float eps     = parser.get<float>("e");

    VideoCapture inputVideo(fileName);
    int ex = static_cast<int>(inputVideo.get(CAP_PROP_FOURCC));
    Size S = Size((int) inputVideo.get(CAP_PROP_FRAME_WIDTH),
                  (int) inputVideo.get(CAP_PROP_FRAME_HEIGHT));

    VideoWriter  outputVideo;
    outputVideo.open(outName, ex, inputVideo.get(CAP_PROP_FPS), S);

    if(!parser.check() || !inputVideo.isOpened())
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

    Mat Image,temp;
    int i = 0;
    while(inputVideo.read(Image)){
      cvtColor(Image,Image,COLOR_BGR2YUV); 
      vector<Mat> spl;
      split(Image,spl);
      cout << "FRAME:\t" << ++i << "/" << inputVideo.get(CAP_PROP_FRAME_COUNT) << endl;
      if( L > 0 && i > L) break;
      Mat result(spl[0].size(),CV_32F);
      spl[0].convertTo(temp,CV_32F);
      temp.copyTo(spl[0]);

      spl[0]/=pow(2,D);
      result = ROFtv(spl[0], N, sigma2, deltaT, eps);

      result.convertTo(spl[0], CV_8U, 255);

      merge(spl,temp);
      
      cvtColor(temp,temp,COLOR_YUV2BGR);
      outputVideo << temp;      
    }
     


    return 0;
}


