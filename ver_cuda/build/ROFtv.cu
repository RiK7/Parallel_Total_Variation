#include "ROFtv.cuh"

#include "finiteDifference.cu"
#include "CU_tool.cu"


#include <cmath>
//#include <thread>
//#include <memory>

using namespace std;
using namespace cv;

namespace tv
{
/*
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

    for(int i=0; i<s.height; ++i)
        for(int j=0; j<s.width; ++j)
        {
                index = i*s.width+j;
                data_[index] = data_U0p[index]
                                /(sqrt(data_U0p[index]*data_U0p[index]
                                      +pow(minmod(data_U1n[index], data_U1p[index]),2))+eps)/h;
        }
    return;
}
*/
__global__ void d_calculateSecondDiff(int height, int width,
                                 float* U0p, float* U0n,
                                 float* U1p, float* U1n,
                                 float* result,
                                 float h,
                                 float eps){
    int row,col;
    row = blockIdx.y*blockDim.y + threadIdx.y;
    col = blockIdx.x*blockDim.x + threadIdx.x;
    if( row < height && col < width ){
        int idx = row*width+col;
        //minmod
        //0.5*((a>0)-(a<0)+(b>0)-(b<0))*fminf(fabsf(a),fabsf(b));
        float temp = 0.5*((U1n[idx]>0)-(U1n[idx]<0)+(U1p[idx]>0)-(U1p[idx]<0))*fminf(fabsf(U1n[idx]),fabsf(U1p[idx]));
        result[idx] = U0p[idx]/(sqrtf(U0p[idx]*U0p[idx] + 
                                powf(temp,2.))+eps)/h;
    }    

}

Mat ROFtv(const Mat& u0,
                   int  N,
                   float sigma,
                   float deltaT,
                   float eps)
{
 //   int index = 0;
    float l=-0;
//    float normValue;
    Size s = u0.size();
    float h = 1./s.width;
/*
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
        temp(s,CV_32F),view;
*/
    Mat u;
    
/*
    float *data_Ux = (float*) Ux.data,
          *data_Uy = (float*) Uy.data,
          *data_xU = (float*) xU.data,
          *data_yU = (float*) yU.data,
          *data_X  = (float*) X.data,
          *data_Y  = (float*) Y.data,
          *data_u  = (float*) u.data,
          *data_temp;
*/

    u0.copyTo(u);

    //thread* lambda_thread;

    dim3 block(32,32),
         grid((s.width-1)/32+1,(s.height-1)/32+1);


    int length = s.width*s.height;
    int size = length*sizeof(float);
    
    float *d_u0,
          *d_Ux,
          *d_Uy,
          *d_xU,
          *d_yU,
          *d_X,
          *d_Y,
          *d_u,
          *d_pool,
          *d_U0x,
          *d_U0y,
          *d_temp,
          *d_temp2;
    float *d_l;

    cudaMalloc(&d_u0,size);
    cudaMalloc(&d_u,size);
    cudaMalloc(&d_xU,size);
    cudaMalloc(&d_yU,size);
    cudaMalloc(&d_Ux,size);
    cudaMalloc(&d_Uy,size);
    cudaMalloc(&d_X,size);
    cudaMalloc(&d_Y,size);
    cudaMalloc(&d_pool,size);
    cudaMalloc(&d_U0x,size);
    cudaMalloc(&d_U0y,size);
    cudaMalloc(&d_temp,size);
    cudaMalloc(&d_temp2,size);

    cudaMalloc(&d_l,size);

    cudaMemcpy(d_u,u.data,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_u0,u0.data,size,cudaMemcpyHostToDevice);

    for(int k=0;k<N ; ++k)
    {
        //TODO para
//        thread delta_x_n(deltaX_n, ref(u), ref(xU)),
//               delta_y_n(deltaY_n, ref(u), ref(yU)),
//               delta_x_p(deltaX_p, ref(u), ref(Ux));

//        deltaY_p(u,Uy);
        d_deltaY_p<<<grid,block>>>(s.height,s.width,d_u,d_Uy);
//*        cudaMemcpy(data_Uy,d_Uy,size,cudaMemcpyDeviceToHost);
//        if(delta_x_p.joinable()) delta_x_p.join();
//        deltaX_p(u,Ux);
        d_deltaX_p<<<grid,block>>>(s.height,s.width,d_u,d_Ux);
//*        cudaMemcpy(data_Ux,d_Ux,size,cudaMemcpyDeviceToHost);

        if(k == 0 ){
            //Ux.copyTo(U0x); Uy.copyTo(U0y);
//*            cudaMemcpy((float*)U0x.data,d_Ux,size,cudaMemcpyDeviceToHost);
            cudaMemcpy(d_U0x,d_Ux,size,cudaMemcpyDeviceToDevice);
            
//*            cudaMemcpy((float*)U0y.data,d_Uy,size,cudaMemcpyDeviceToHost);
            cudaMemcpy(d_U0y,d_Uy,size,cudaMemcpyDeviceToDevice);
        }
//        else
//            lambda_thread = new thread(lambda, ref(U0x), ref(U0y), 
//                                          ref(Ux), ref(Uy), 
//                                          -0.5*h/sigma, ref(l));

//        if(delta_x_n.joinable()) delta_x_n.join();
////        deltaX_n(u,xU);
        d_deltaX_n<<<grid,block>>>(s.height,s.width,d_u,d_xU);
//*        cudaMemcpy(data_xU,d_xU,size,cudaMemcpyDeviceToHost);
//        if(delta_y_n.joinable()) delta_y_n.join();
////        deltaY_n(u,yU);
        d_deltaY_n<<<grid,block>>>(s.height,s.width,d_u,d_yU);
//*        cudaMemcpy(data_yU,d_yU,size,cudaMemcpyDeviceToHost);
        

//        thread secDiffthread(calculateSecondDiff, Ux, xU, 
//                                                  Uy, yU, 
//                                                  X,
//                                                  h,eps);
//        calculateSecondDiff(Uy,yU,Ux,xU,Y,h,eps);
        d_calculateSecondDiff<<<grid,block>>>(s.height,s.width,d_Uy,d_yU,d_Ux,d_xU,d_Y,h,eps);
//*        cudaMemcpy(data_Y,d_Y,size,cudaMemcpyDeviceToHost);
        
////        pool = u-u0;
        d_matrixMinus<<<(length-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(d_u,d_u0,d_pool,length);
//*        cudaMemcpy((float*)pool.data, d_pool,size , cudaMemcpyDeviceToHost);
        
        if(k>0)
        {
//            if(lambda_thread->joinable())
//                lambda_thread->join();
//            delete lambda_thread;
////            lambda(U0x,U0y,Ux,Uy,-0.5*h/sigma,l);
        
        d_lambda<<<(length-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(s.height,s.width,
                   d_U0x, d_U0y, d_Ux, d_Uy,
                   -0.5*h/sigma, d_l);
        l = sumOfArray_with_D(d_l,length); 

        }
        
//        if(secDiffthread.joinable()) secDiffthread.join();
////        calculateSecondDiff(Ux,xU,Uy,yU,X,h,eps);
        d_calculateSecondDiff<<<grid,block>>>(s.height,s.width,d_Ux,d_xU,d_Uy,d_yU,d_X,h,eps);
//*        cudaMemcpy(data_X,d_X,size,cudaMemcpyDeviceToHost);

////        temp = -l*pool;
        d_mul<<<(length-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(d_pool,d_temp,-l,length);
//*        cudaMemcpy(temp.data,d_temp,size,cudaMemcpyDeviceToHost);
        
////        deltaX_n(X, pool); //TODO para
        d_deltaX_n<<<grid,block>>>(s.height,s.width,d_X,d_pool);
//*        cudaMemcpy(pool.data,d_pool,size,cudaMemcpyDeviceToHost);
        
////        temp += pool;
        d_add_equal<<<(length-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(d_pool,d_temp,length);
//*        cudaMemcpy(temp.data,d_temp,size,cudaMemcpyDeviceToHost);

////        deltaY_n(Y, pool); //TODO para
        d_deltaY_n<<<grid,block>>>(s.height,s.width,d_Y,d_pool);
//*        cudaMemcpy(pool.data,d_pool,size,cudaMemcpyDeviceToHost);

////        temp += pool;
        d_add_equal<<<(length-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(d_pool,d_temp,length);
//*        cudaMemcpy(temp.data,d_temp,size,cudaMemcpyDeviceToHost);

////        temp *= deltaT;
        d_mul<<<(length-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(d_temp,d_temp,deltaT,length);
//*        cudaMemcpy(temp.data,d_temp,size,cudaMemcpyDeviceToHost);
        

/*below didn't use
        normValue=0;
        data_temp = (float*)temp.data;
        for(int i=1, j; i<s.height-1; ++i)
            for(j=1; j<s.width; ++j)
                normValue += abs(data_temp[i*s.width+j]);
        normValue*=(h*h);
*/

////        u += temp;
        d_add_equal<<<(length-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(d_temp,d_u,length);
//*        cudaMemcpy(data_u,d_u,size,cudaMemcpyDeviceToHost);

        //TODO para
////        for(int i=0; i<s.width; ++i)
////            data_u[i] = data_u[i+s.width];
        d_assignWithShift<<<(s.width-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(d_u, s.width, s.width);
//*        cudaMemcpy(data_u,d_u,size,cudaMemcpyDeviceToHost);

//        for(int i=0; i<s.width; ++i)
//            data_u[(s.height-1)*s.width+i] = data_u[(s.height-2)*s.width+i];
        d_assignWithShift<<<(s.width-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(d_u+(s.height-1)*s.width, s.width, -s.width);
//*        cudaMemcpy(data_u,d_u,size,cudaMemcpyDeviceToHost);

////        for(int i=0; i<s.height; ++i)
////        {
////            data_u[i*s.width] = data_u[(i+1)*s.width-2];
////            data_u[(i+1)*s.width-1] = data_u[(i+1)*s.width-2];
////        }
        d_assignWithMagic<<<(s.height-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(d_u,s.height,s.width-1,s.width);
        d_assignWithMagic<<<(s.height-1)/BLOCK_SIZE+1,BLOCK_SIZE>>>(d_u+s.width-1,s.height,-1,s.width);
        
//*        cudaMemcpy(d_u,data_u,size,cudaMemcpyHostToDevice); 
    }
    cudaMemcpy(u.data,d_u,size,cudaMemcpyDeviceToHost); 

    cudaFree(d_u0);
    cudaFree(d_u);
    cudaFree(d_xU);
    cudaFree(d_yU);
    cudaFree(d_Ux);
    cudaFree(d_Uy);
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_pool);
    cudaFree(d_U0x);
    cudaFree(d_U0y);
    cudaFree(d_temp);
    cudaFree(d_temp2);

    cudaFree(d_l);

    return u;
}

}


