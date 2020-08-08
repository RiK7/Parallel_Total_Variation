#include <opencv2/opencv.hpp>
#include <cmath> 

#include <iostream>
using namespace std;

void deltaX_p(const cv::Mat& u, cv::Mat& result)
{
    cv::Size s = u.size();
    int type   = u.type(); 

    result.create(s, type);
    float* data_u = (float*)u.data;
    float* data_r = (float*)result.data;

    for(int i=0; i<s.height; ++i)
    {
        for(int j=0; j<s.width-1; ++j)
        {
            data_r[i*s.width+j] = -data_u[i*s.width+j]  + data_u[i*s.width+j+1];
        }
        data_r[i*s.width+s.width-1] = 0;
    }
}

__global__ void d_deltaX_p(const int height,const int width, float* data_u, float* data_r){
    int row,col;
    row = blockIdx.y*blockDim.y + threadIdx.y;
    col = blockIdx.x*blockDim.x + threadIdx.x;

    if(row < height){
        if(col < width-1)
            data_r[row*width+col] = -data_u[row*width+col] + data_u[row*width+col+1];
        else if(col == width-1)
            data_r[row*width+col] = 0;
    }
}

void deltaX_n(const cv::Mat& u, cv::Mat& result)
{
    cv::Size s = u.size();
    int type   = u.type(); 

    result.create(s, type);
    float* data_u = (float*)u.data;
    float* data_r = (float*)result.data;

    for(int i=0; i<s.height; ++i)
    {
        data_r[i*s.width] = 0;
        for(int j=1; j<s.width; ++j)
        {
            data_r[i*s.width+j] = - data_u[i*s.width + j-1] + data_u[i*s.width+j];
        }
    }
}

__global__ void d_deltaX_n(const int height,const int width, float* data_u, float* data_r){
    int row,col;
    row = blockIdx.y*blockDim.y + threadIdx.y;
    col = blockIdx.x*blockDim.x + threadIdx.x;

    if(row < height){
        if(col == 0)
            data_r[width*row] = 0;
        else if(col < width)
            data_r[width*row + col] = -data_u[row*width+col-1] + data_u[row*width+col];
    }
}

void deltaY_p(const cv::Mat& u, cv::Mat& result)
{
    cv::Size s = u.size();
    int type = u.type();
    
    result.create(s, type);
    float* data_u = (float*)u.data;
    float* data_r = (float*)result.data;

    for(int j = 0; j<s.width; ++j)
    {
        for(int i = 0; i<s.height-1; ++i)
        {
            data_r[i*s.width+j] = data_u[(i+1)*s.width+j] - data_u[i*s.width+j];
        }
        data_r[(s.height-1)*s.width+j] = 0;
    }

}

__global__ void d_deltaY_p(const int height,const int width, float* data_u, float* data_r){
    int row,col;
    row = blockIdx.y*blockDim.y + threadIdx.y;
    col = blockIdx.x*blockDim.x + threadIdx.x;

    if(col < width){
        if(row < height-1)
            data_r[row*width+col] = data_u[(row+1)*width+col] - data_u[row*width+col];
        else if(row == height-1)
            data_r[(height-1)*width+col] = 0;
    }
}

void deltaY_n(const cv::Mat& u, cv::Mat& result)
{
    cv::Size s = u.size();
    int type = u.type();
    
    result.create(s, type);
    float* data_u = (float*)u.data;
    float* data_r = (float*)result.data;

    for(int j = 0; j<s.width; ++j)
    {
        for(int i = 1; i<s.height; ++i)
        {
            data_r[i*s.width+j] = - data_u[(i-1)*s.width+j] + data_u[i*s.width+j];
        }
        data_r[j] = 0;
    }

}

__global__ void d_deltaY_n(const int height,const int width, float* data_u, float* data_r){
    int row,col;
    row = blockIdx.y*blockDim.y + threadIdx.y;
    col = blockIdx.x*blockDim.x + threadIdx.x;

    if(col < width){
        if(row == 0)
            data_r[width*row] = 0;
        else if(row < height)
            data_r[width*row + col] = -data_u[(row-1)*width+col] + data_u[row*width+col];
    }
}

float minmod(const float a, const float b)
{
    int sgn_a = (a>0) - (a<0),
        sgn_b = (b>0) - (b<0);
    
    return 0.5*static_cast<float>(sgn_a+sgn_b)*std::min(abs(a),abs(b));    
}

void lambda(const cv::Mat& u0x, const cv::Mat& u0y, 
            const cv::Mat& ux,  const cv::Mat& uy, 
            float coeff, float& lam)
{
    lam = 0;
    int index;
    float temp, elem;
    cv::Size s = u0x.size();

    float *data_u0x = (float*) u0x.data,
          *data_u0y = (float*) u0y.data,
          *data_ux  = (float*) ux.data,
          *data_uy  = (float*) uy.data;

    for(int i=0,j; i<s.height; ++i)
    {
        for(j=0; j<s.width; ++j)
        {
            index = i*s.width+j;
            temp = data_ux[index]*data_ux[index] 
                  +data_uy[index]*data_uy[index];
            
            elem =  temp 
                  - data_ux[index]*data_u0x[index]
                  - data_uy[index]*data_u0y[index];

            elem /= (sqrt(temp) + 0.0001);

            lam += elem;
        }
    }

    lam *= coeff;
}


__global__ void d_lambda(int height, int width,
                        float* data_u0x, float* data_u0y, float* data_ux, float* data_uy,
                        float coeff, float *lam) //lam is a matrix of all element
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < height*width){
        float temp,elem;
        temp = data_ux[idx]*data_ux[idx] 
              +data_uy[idx]*data_uy[idx];
        elem = temp
             - data_ux[idx]*data_u0x[idx]
             - data_uy[idx]*data_u0y[idx];
        elem /= (sqrt(temp)+ 0.0001);
        lam[idx] = coeff*elem;
    }
}


__global__ void d_matrixMinus(float* A, float* B, float* out, int N){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < N){
        out[idx] = A[idx] - B[idx];
    }
}

__global__ void d_assignWithShift(float* in, int N, int shift){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < N)
        in[idx] = in[idx+shift];
}

__global__ void d_assignWithMagic(float* in, int N, int shift, int NEXT){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int position = idx*NEXT;
    in[position] = in[position+shift];
}

