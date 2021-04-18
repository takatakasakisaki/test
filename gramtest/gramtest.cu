
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>
#include "cublas_v2.h"
#include "openblas/cblas.h"

using namespace std;

typedef uint32_t arrya1_t[1024];
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
int srchmax(uint32_t(*buf)[1024], int sz)
{
    uint32_t maxv = 0;
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < 1024; j++) {
            maxv = std::max(maxv, buf[i][j]);
        }
    }
    return maxv;
}

uint32_t (*magbuf)[1024];
// Helper function for using CUDA to add vectors in parallel.
cudaError_t transfer(uint32_t *src[])
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;
    arrya1_t* p1;
    cudaStatus = cudaMallocHost((void**)&p1, 1024 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    p1[0][0] = 1;


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
#if 0
    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
#endif 
Error:
    return cudaStatus;
}
#define LINESIZE 2432
#define LINENUM 640
#define IMGHEIGHT 104
#define IMGWIDTH (LINESIZE*LINENUM)
uint16_t (*indata)[IMGWIDTH];
float scale[IMGHEIGHT];
float (*indata_f)[IMGHEIGHT] ;
typedef float (*indataf_t)[LINESIZE*LINENUM] ;
float (*gram)[IMGHEIGHT ] ;
void mulmatrix(float A[IMGHEIGHT][IMGWIDTH], float B[IMGWIDTH][IMGHEIGHT])
{
    for (int row = 0; row < IMGHEIGHT; row++) {
        for (int col = 0; col < IMGHEIGHT; col++) {
            float s = 0;
            for (int i = 0; i < IMGWIDTH; i++) {
                s += A[row][i] * B[i][col];
            }
            gram[row][col] = s;
        }
    }

}

__global__ void kernelgramm(float A[], 
    float C[],
    int nwidth, int nheight
)

{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row >= nheight) {
        return;
    }
    if (col >= nheight) {
        return;
    }
    float sum = 0;
    for (int i = 0; i < IMGWIDTH; i++) {
        sum += A[row*nwidth+i] * A[col*nwidth+i];
    }
    C[row*nheight+col] = sum;

}
#define UNIMEM 1
int  makegram_cblas()
{
    //int m = IMGWIDTH;
    int m = IMGHEIGHT;
    int n = IMGHEIGHT;
    int k = IMGWIDTH;
    float* A=(float*)indata_f;
    float* B=(float*)indata_f;
    float* C=(float*)gram;
    int lda=IMGWIDTH;
    int ldb=IMGWIDTH;
    int ldc=IMGHEIGHT;
    float alpha = 1.0;
    float beta = 0;

    memset(C, 0, IMGHEIGHT * IMGHEIGHT * sizeof(float));
    auto ts0 = chrono::system_clock::now();
    cblas_sgemm(CblasColMajor, CblasTrans,CblasNoTrans, 
        m, n ,k, alpha, A, lda, B, ldb, beta, C, ldc);
    auto ts1 = chrono::system_clock::now();
    chrono::duration<double> diff = ts1 - ts0;
    cout << "makegram_cblas " << diff.count() << ",stat," << endl;
    return 0;
}
int  makegram_cublas()
{
    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
#if 0
    cublasStatus_t cublasSgemm(cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const float* alpha,
        const float* A, int lda,
        const float* B, int ldb,
        const float* beta,
        float* C, int ldc);
#endif
    float alpha = 1.0;
    cublasOperation_t transa=CUBLAS_OP_T;
    cublasOperation_t transb=CUBLAS_OP_N;
    //int m = IMGWIDTH;
    int m = IMGHEIGHT;
    int n = IMGHEIGHT;
    int k = IMGWIDTH;
    float* A;
    float* B;
    float* C;
    int lda=IMGWIDTH;
    int ldb=IMGWIDTH;
    int ldc=IMGHEIGHT;
    float beta = 0;
    cudaError_t cudaStatus;

	cudaStatus = cudaMallocManaged((void**)&A, IMGHEIGHT * IMGWIDTH * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 1;
	}
	cudaStatus = cudaMallocManaged((void**)&C, IMGHEIGHT * IMGWIDTH * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 1;
	}

        //cudaStatus = cudaMemcpy(A, indata_f, IMGWIDTH * IMGHEIGHT * sizeof(float), cudaMemcpyHostToDevice);
        cudaStatus = cudaMemcpy(A, indata_f, IMGWIDTH * IMGHEIGHT * sizeof(float), cudaMemcpyHostToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
        }

	B = A;

    cudaMemset(C, 0, IMGHEIGHT * IMGHEIGHT * sizeof(float));
    auto ts0 = chrono::system_clock::now();
    stat = cublasSgemm(handle, transa, transb, m, n ,k, &alpha, A, lda, B, ldb, &beta, C, ldc);
        cudaDeviceSynchronize();
    auto ts1 = chrono::system_clock::now();
        chrono::duration<double> diff = ts1 - ts0;
        cout << "makegram_cublas " << diff.count() << ",stat," << stat << endl;
    cublasDestroy(handle);
    cudaFree(A);
    cudaFree(C);
    return 0;
}
void makegram_gpu()
{
    cudaError_t cudaStatus;
    float* matA;
    float* matC;
    if(UNIMEM)
    {
        auto ts0 = chrono::system_clock::now();
        cudaStatus = cudaMallocManaged((void**)&matA, IMGHEIGHT * IMGWIDTH * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "matA cudaMalloc failed!");
            return;
        }
        cudaStatus = cudaMallocManaged((void**)&matC, IMGHEIGHT * IMGWIDTH * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            return;
        }
        auto ts1 = chrono::system_clock::now();
        chrono::duration<double> diff = ts1 - ts0;
        cout << "makegram_kernel alloc " << diff.count() << endl;
    }
    else {
        auto ts0 = chrono::system_clock::now();
        cudaStatus = cudaMalloc((void**)&matA, IMGHEIGHT * IMGWIDTH * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            return;
        }
        cudaStatus = cudaMalloc((void**)&matC, IMGHEIGHT * IMGWIDTH * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            return;
        }
        auto ts1 = chrono::system_clock::now();
        chrono::duration<double> diff = ts1 - ts0;
        cout << "makegram_kernel alloc " << diff.count() << endl;
    }

    auto ts0 = chrono::system_clock::now();
    if (UNIMEM) {
        //float (*indata_f)[IMGHEIGHT] ;
        //typedef float (*indataf_t)[LINESIZE*LINENUM] ;
        //float (*gram)[IMGHEIGHT ] ;
            // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(matA, indata_f, IMGWIDTH * IMGHEIGHT * sizeof(float), cudaMemcpyHostToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
        }
    }
    else {
        cudaStatus = cudaMemcpy(matA, indata_f, IMGWIDTH*IMGHEIGHT* sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
        }
    }

    auto ts0a = chrono::system_clock::now();

    int thnum = 128;
    dim3 blkdim(128, 1, 1);
    dim3 grid( (IMGHEIGHT+thnum-1)/ thnum, (IMGHEIGHT + thnum-1)/thnum, 1);
        //makecontrast_kernel<<< grid, blkdim >>> ((uchar4*)dev_dest, (uchar4*)dev_in, height, max_val, min_val);
   //     makecontrast1_kernel<<< grid, blkdim >>> (dev_dest, dev_in, height, max_val, min_val);
    kernelgramm <<< grid, blkdim >>> (matA, matC, IMGWIDTH, IMGHEIGHT);
    cudaDeviceSynchronize();



    auto ts1a = chrono::system_clock::now();
    cudaStatus = cudaMemcpy(matC, gram, IMGHEIGHT*IMGHEIGHT* sizeof(float), cudaMemcpyDeviceToHost);
    auto ts1b = chrono::system_clock::now();
    {
        chrono::duration<double> diff = ts0a - ts0;
        cout << "makegram_kernel pre copy " << diff.count() << endl;
    }
    {
        chrono::duration<double> diff = ts1a - ts0a;
        cout << "makegram_kernel kernel " << diff.count() << endl;
    }
    {
        chrono::duration<double> diff = ts1b - ts1a;
        cout << "makegram_kernel after copy " << diff.count() << endl;
    }

    {
        chrono::duration<double> diff = ts1b - ts0;
        cout << "makegram_kernel total" << diff.count() << endl;
    }
    cudaFree(matA);
    cudaFree(matC);
}

void transpose(float indata[IMGHEIGHT][IMGWIDTH], float outdata[IMGWIDTH][IMGHEIGHT])
{
    for (int row = 0; row < IMGHEIGHT; row++) {
        for (int col = 0; col < IMGWIDTH; col++) {
            outdata[col][row] = indata[row][col];
        }
    }
}
void makegram()
{
    cout << "makegram\n";
    //gram = new float[IMGHEIGHT][IMGHEIGHT];
    auto ts0 = chrono::system_clock::now();
    int col, i;
#pragma omp parallel for private(col, i)
    for (int row = 0; row < IMGHEIGHT; row++) {
        for (col = 0; col < IMGHEIGHT; col++) {
            float s = 0;
            for (i = 0; i < IMGWIDTH; i++) {
                s += indata_f[row][i] * indata_f[col][i];
            }
            gram[row][col] = s;
        }
    }
    auto ts1 = chrono::system_clock::now();
    chrono::duration<double> diff = ts1 - ts0;
    cout << "makegram done " << diff.count() << endl;
}
void makedata()
{
    cout << "makedata\n";
    auto ts0 = chrono::system_clock::now();
    for (int row = 0; row < IMGHEIGHT; row++) {
        for (int col = 0; col < IMGWIDTH; col++) {
            indata_f[row][col] *= scale[row];
        }
    }
    auto ts1 = chrono::system_clock::now();
    chrono::duration<double> diff = ts1 - ts0;
    cout << "makedata done " << diff.count() << endl;

}
void loaddata()
{
    indata = new uint16_t[IMGHEIGHT][IMGWIDTH];
    for(int i = 0; i < IMGHEIGHT ; i++){ 
        scale[i] = 1.0;
    }
    std::random_device rnd;

//float (*indata_f)[IMGHEIGHT] ;
    indata_f = new float[IMGWIDTH][IMGHEIGHT];
    for (int row = 0; row < IMGWIDTH; row++) {
        for (int col = 0; col < IMGHEIGHT; col++) {
            indata_f[row][col] = int (rnd())&0x3fff;
        }
    }

    gram = new float[IMGHEIGHT][IMGHEIGHT];

}


int main()
{
    printf("%s %s\n", __DATE__, __TIME__);
    loaddata();
    printf("%s %s\n", __DATE__, __TIME__);
    makedata();
    printf("%s %s\n", __DATE__, __TIME__);
    makegram();
    printf("%s %s\n", __DATE__, __TIME__);

    makegram_gpu();
    makegram_cublas();
    makegram_cblas();
#if 0
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    cudaError_t cudaStatus;
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    loaddata();
#endif
    return 0;
}
