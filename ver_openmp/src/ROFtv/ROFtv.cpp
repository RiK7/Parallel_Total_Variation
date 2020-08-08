#include "ROFtv.hpp"

#include "finiteDifference.hpp"

#include <cmath>
#include <memory>

using namespace std;
using namespace cv;

namespace tv
{

    void calculateSecondDiff(InputArray U0p_, InputArray U0n_,
            InputArray U1p_, InputArray U1n_,
            OutputArray result_,
            float h,
            float eps)
    {
        Mat U0p = U0p_.getMat(),
        U1p = U1p_.getMat(),
        U0n = U0n_.getMat(),
        U1n = U1n_.getMat();
        Size s = U0p.size();
        result_.create(s, U0p.type());
        Mat result = result_.getMat();

        float *data_U0p = (float*)U0p.data,
              *data_U1p = (float*)U1p.data,
              *data_U0n = (float*)U0n.data,
              *data_U1n = (float*)U1n.data,
              *data_    = (float*)result.data;

        int index = 0;

        #pragma omp parallel for 
        for(int i=0; i<s.height; ++i)
            #pragma omp parallel for 
            for(int j=0; j<s.width; ++j)
            {
                index = i*s.width+j;
                data_[index] = data_U0p[index] /(sqrt(data_U0p[index]*data_U0p[index] +pow(minmod(data_U1n[index], data_U1p[index]),2))+eps)/h;
            }
        return;
    }

    Mat ROFtv(const Mat& u0,
            int  N,
            float sigma,
            float deltaT,
            float eps)
    {
        int index = 0;
        float l=-0, normValue;
        Size s = u0.size();
        float h = 1./s.width;
        Mat U0x(s, CV_32F),
            U0y(s, CV_32F),
            Ux(s, CV_32F), 
            Uy(s, CV_32F),
            xU(s, CV_32F),
            yU(s, CV_32F),
            pool(s, CV_32F),
            X(s, CV_32F),
            Y(s, CV_32F),
            u(s, CV_32F),
            temp,view;


        float *data_Ux = (float*) Ux.data,
              *data_Uy = (float*) Uy.data,
              *data_xU = (float*) xU.data,
              *data_yU = (float*) yU.data,
              *data_X  = (float*) X.data,
              *data_Y  = (float*) Y.data,
              *data_u  = (float*) u.data,
              *data_temp;

        u0.copyTo(u);

        for(int k=0; k<N; ++k)
        {
            #pragma omp parallel sections 
            {
                #pragma omp section
                deltaY_p(u,Uy);
                #pragma omp section
                deltaX_p(u,Ux);
            }

            if (k == 0) {
                Ux.copyTo(U0x); 
                Uy.copyTo(U0y);
            }

            #pragma omp parallel sections 
            {
                #pragma omp section
                deltaX_n(u, xU);
                #pragma omp section
                deltaY_n(u, yU);
            }

            calculateSecondDiff(Uy,yU,Ux,xU,Y,h,eps);

            pool = u - u0;
            if(k > 0)
                lambda(U0x, U0y, Ux, Uy, -0.5*h/sigma, l);

            calculateSecondDiff(Ux, xU, Uy, yU, X, h, eps);

            temp = -l*pool;
            deltaX_n(X, pool);
            temp += pool;
            deltaY_n(Y, pool);
            temp += pool;
            temp *= deltaT;
            normValue=0;
            data_temp = (float*)temp.data;

            #pragma omp parallel for
            for(int i=1; i<s.height-1; ++i)
                #pragma omp parallel for reduction(+:normValue)
                for(int j=1; j<s.width; ++j)
                    normValue += abs(data_temp[i*s.width+j]);
            normValue*=(h*h);

            u += temp;

            #pragma omp parallel sections
            {
                #pragma omp section
                #pragma omp parallel for 
                for(int i=0; i<s.width; ++i)
                    data_u[i] = data_u[i+s.width];

                #pragma omp section
                #pragma omp parallel for 
                    for(int i=0; i<s.width; ++i)
                        data_u[(s.height-1)*s.width+i] = data_u[(s.height-2)*s.width+i];

                #pragma omp section
                #pragma omp parallel for 
                for(int i=0; i<s.height; ++i)
                {
                    data_u[i*s.width] = data_u[(i+1)*s.width-2];
                    data_u[(i+1)*s.width-1] = data_u[(i+1)*s.width-2];
                }
            }

        }
        return u;
    }

}


