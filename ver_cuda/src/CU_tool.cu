#ifndef BLOCK_SIZE
    #define BLOCK_SIZE 512
#endif

template <class DataType>
__global__ void reduce(DataType *g_idata, DataType *g_odata,int N) {
    //extern __shared__ int sdata[];
    __shared__ DataType sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i]*(i<N);
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s]; 
        }
        __syncthreads();
    }   
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template<class DataType>
DataType sumOfArray_with_D(DataType *d_in, int N){
    
    DataType *d_0;
    DataType *d_1;

    cudaMalloc(&d_0,sizeof(DataType)*N);
    cudaMalloc(&d_1,sizeof(DataType)*N);

    cudaMemcpy(d_0,d_in,N*sizeof(DataType),cudaMemcpyDeviceToDevice);

    bool flag = true;
    for(; N > 1 ; N = (N+BLOCK_SIZE-1)/BLOCK_SIZE){
        if(flag)
            reduce<<< (N-1)/BLOCK_SIZE+1 , BLOCK_SIZE >>>(d_0,d_1,N);
        else
            reduce<<< (N-1)/BLOCK_SIZE+1 , BLOCK_SIZE >>>(d_1,d_0,N);
        flag = !flag;
    }

    DataType out;
    if(flag)
        cudaMemcpy(&out,d_0,1*sizeof(DataType),cudaMemcpyDeviceToHost);
    else
        cudaMemcpy(&out,d_1,1*sizeof(DataType),cudaMemcpyDeviceToHost); 
    
    cudaFree(d_0);
    cudaFree(d_1);

    return out;
}

template<class DataType>
DataType sumOfArray(DataType *in, int N){
    DataType *d_in;
    cudaMalloc(&d_in,N*sizeof(DataType));
    cudaMemcpy(d_in, in, N*sizeof(DataType), cudaMemcpyHostToDevice);

    DataType out = sumOfArray_with_D(d_in, N);

    cudaFree(d_in);
    return out;
}

template<class DataType>
__global__ void d_mul(DataType *in, DataType* out, DataType scale, int N){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < N)
        out[idx] = in[idx]*scale;
}

template<class DataType>
__global__ void d_add_equal(DataType *in, DataType* out,int N){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < N)
        out[idx] += in[idx];
}

template<class DataType>
__global__ void d_abs(DataType *in, DataType *out, int N){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    DataType temp = in[idx];
    if(idx < N)
        out[idx] = temp*((temp > 0) - (temp < 0));
}
