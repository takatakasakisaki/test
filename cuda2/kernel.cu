
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <chrono>
#include <valarray>
#include <iostream>
#include <omp.h>

void makecontrast_cpu(uint8_t* dest, const uint8_t* input, const int height, uint8_t max_val, uint8_t min_val)
{
    float range = max_val - min_val;
    if (range <= 0) {
        range = 1.0;
    }
    float ratio = 255.0 / range;
#pragma omp parallel for
    for (int row = 0; row < height; row++) {
        uint8_t* line_out = dest + 2440 * row;
        const uint8_t* in = input + 2440 * row;
        for (int col = 0; col < 2440; col++) {
            line_out[col] = (in[col] - min_val) * ratio;
        }
    }

}

__global__ void 
makecontrast_kernel(uchar4* dest, const uchar4* input, const int height, uint8_t max_val, uint8_t min_val)
{
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    if (col < 2440/4) {
        float range = max_val - min_val;
        if (range <= 0) {
            range = 1.0;
        }
        float ratio = 255.0 / range;
        for (int row = 0; row < height; row++) {
            const uchar4* in = input + row * (2440 / 4) + col;
            uchar4* out = dest + row * (2440 / 4) + col;
            uchar4 odat;
            odat.x = (in->x - min_val) * ratio;
            odat.y = (in->y - min_val) * ratio;
            odat.z = (in->z - min_val) * ratio;
            odat.w = (in->w - min_val) * ratio;
            *out = odat;
        }
    }

}
__global__ void 
makecontrast1_kernel(uint8_t* dest, const uint8_t* input, const int height, uint8_t max_val, uint8_t min_val)
{
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    if(col < 2440){
		float range = max_val - min_val;
        if (range <= 0) {
            range = 1.0;
        }
		float ratio = 255.0 / range;
        for (int row = 0; row < height; row++) {
            const uint8_t* in = input + row * (2440) + col;
            uint8_t* out = dest + row * (2440) + col;
            *out = (*in - min_val) * ratio;
        }
	}

}
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
__global__ void verticalave_kernel(uint8_t* dest, const uint8_t* input, const int height, uint8_t *max_val, uint8_t *min_val)
{
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    if (col < 2440) {
        uint8_t max0 = 0;
        uint8_t min0 = 255;
        int pitch = 2440;
        uint32_t sum = 0x00;
        int pos = col;
        //max_val[col] = 0x55;
        //min_val[col] = 0xaa;
        for (int row = 0; row < height; row++) {
            uint8_t p =input[pos]; 
            sum += p;
            pos += pitch;
        }
        pos = col;
        for (int row = 0; row < height; row++) {
            uint8_t p =input[pos]; 
#if 0
            max0 = max(p, max0);
            min0 = min(p, min0);
#else
            if (p > max0) {
                max0 = p;
            }
            if (p < min0) {
                min0 = p;
            }
#endif
            pos += pitch;
        }
        //printf("col=%x,%x,%x\n", col, max0, min0);
        sum /= height;
        dest[col] = sum;
        max_val[col] = max0;
        min_val[col] = min0;
        //max_val[col] = col;
        //min_val[col] = ~col;
        //printf("g:bx=%x,tx=%x,col=%x, dest=%02x,h=%d\n", blockIdx.x, threadIdx.x,col, dest[col],height);
    }

}
void verticalave_host(uint8_t* dest, const uint8_t* input, const int height, uint8_t* max_val, uint8_t* min_val)
{
	uint8_t max0[2440];
    uint8_t min0[2440];
    memset(max0, 0, 2440);
    memset(min0, 255, 2440);
	int pitch = 2440;
	//uint32_t sum = 0x00;
    uint32_t sum[2440];
    memset(sum, 0, 2440);
    const uint8_t* linep = input;
#pragma omp parallel for
	for (int row = 0; row < height; row++) {
        for (int col = 0; col < 2440; col++) {
            uint8_t p = linep[col];
            sum[col] += p;
			if (p > max0[col]) {
				max0[col] = p;
			}
			if (p < min0[col]) {
				min0[col] = p;
			}
        }
        linep += pitch;
	}
	for (int col = 0; col < pitch; col++) {
		uint32_t p =sum[col]; 
        dest[col] = p / height;
	}
	//printf("col=%x,%x,%x\n", col, max0, min0);
}

uint8_t *dev_in = nullptr;
uint8_t *dev_dest = nullptr;
uint8_t *dev_maxval = nullptr;
uint8_t *dev_minval = nullptr;
void kernel_init()
{
    cudaError_t cudaStatus;
    int pitch = 2440;
    int height = 220;
#if 0
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_dest, pitch * sizeof(uint8_t)*height);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_maxval, pitch * sizeof(uint8_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&dev_minval, pitch * sizeof(uint8_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&dev_in, pitch * sizeof(uint8_t) * height);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
#else
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMallocManaged((void**)&dev_dest, pitch * sizeof(uint8_t)*height);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMallocManaged((void**)&dev_maxval, pitch * sizeof(uint8_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMallocManaged((void**)&dev_minval, pitch * sizeof(uint8_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMallocManaged((void**)&dev_in, pitch * sizeof(uint8_t) * height);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
#endif
}
// convert contrast
int convcontrast_cpu(uint8_t *dest, const uint8_t *in, unsigned int height, uint8_t max_val, uint8_t min_val)
{
    int cudaStatus=0;

    do {

        auto t0 = std::chrono::system_clock::now();
        // Launch a kernel on the GPU with one thread for each element.
        makecontrast_cpu(dest, in, height, max_val, min_val);

        auto t1 = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = t1 - t0;
        printf("convcontrast_cpu t=%f,\n", diff.count() * 1000);
        std::fflush(stdout);
    } while (0);

//Error:
    return cudaStatus;
}
// convert contrast
cudaError_t convcontrast_gpu(uint8_t *dest, const uint8_t *in, unsigned int height, uint8_t max_val, uint8_t min_val)
{
    cudaError_t cudaStatus;

    int pitch = 2440;

    do {


        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(dev_in, in, pitch * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
			break;
        }

        auto t0 = std::chrono::system_clock::now();
        // Launch a kernel on the GPU with one thread for each element.
        //dev_c dest
        // c = a + b
        //int thnum = 128;
        int thnum = 256;
        dim3 blkdim(thnum, 1, 1);
        dim3 grid((2440 + thnum - 1) / thnum, 1, 1);
        //makecontrast_kernel<<< grid, blkdim >>> ((uchar4*)dev_dest, (uchar4*)dev_in, height, max_val, min_val);
        makecontrast1_kernel<<< grid, blkdim >>> (dev_dest, dev_in, height, max_val, min_val);
        //__global__ void makecontrast(uchar4* dest, const uchar4* input, const int height, uint8_t max_val, uint8_t min_val)


        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			break;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			break;
        }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(dest, dev_dest, pitch * sizeof(uint8_t)*height, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            break;
        }
        auto t1 = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = t1 - t0;
        printf("convcontrast gpu t=%f\n", diff.count() * 1000);
        std::fflush(stdout);
    } while (0);

//Error:
    return cudaStatus;
}
// Helper function for using CUDA to add vectors in parallel.
cudaError_t verticalave_gpu(uint8_t *dest, const uint8_t *in, unsigned int height)
{
    cudaError_t cudaStatus;

    int pitch = 2440;

    do {


        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(dev_in, in, pitch * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
			break;
        }

        auto t0 = std::chrono::system_clock::now();
        // Launch a kernel on the GPU with one thread for each element.
        //dev_c dest
        // c = a + b
        //int thnum = 128;
        int thnum = 256;
        dim3 blkdim(thnum, 1, 1);
        dim3 grid((2440 + thnum - 1) / thnum, 1, 1);
        //verticalave_kernel <<<grid, blkdim >>> (dev_dest, dev_in, height, dev_maxval, dev_minval);
        verticalave_kernel<<<grid, blkdim >>> (dev_dest, dev_in, height, dev_maxval, dev_minval);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			break;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			break;
        }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(dest, dev_dest, pitch * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            break;
        }
        //std::valarray<uint8_t> h_min(pitch);
        static uint8_t h_min[2440];
        memset(&h_min[0], 0x55, pitch);
        cudaStatus = cudaMemcpy(&h_min[0], dev_minval, pitch * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            break;
        }
        static uint8_t h_max[2440];
        memset(&h_max[0], 0xaa, pitch);
        cudaStatus = cudaMemcpy(&h_max[0], dev_maxval, pitch * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            break;
        }
        auto t1 = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = t1 - t0;
        std::valarray<uint8_t> h_maxa(h_max,2440);
        std::valarray<uint8_t> h_mina(h_min,2440);
        printf("verticalave_gpu t=%f,%d,%d\n", diff.count()*1000, h_maxa.max(), h_mina.min());
#undef max
#undef min
#if 0
        for (int i = 0; i < 2440; i++) {
            printf("%x, %02x,%02x\n", i, h_max[i], h_min[i]);
        }
#endif
        std::fflush(stdout);
    } while (0);

//Error:
    return cudaStatus;
}
cudaError_t verticalave_cpu(uint8_t *dest, const uint8_t *in, unsigned int height)
{
    cudaError_t cudaStatus= cudaSuccess;

    int pitch = 2440;
    uint8_t max_val[2440];
    uint8_t min_val[2440];
    do {

        auto t0 = std::chrono::system_clock::now();
        verticalave_host (dest, in, height, max_val, min_val);
        auto t1 = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = t1 - t0;
        printf("verticalave_cpu t=%f\n", diff.count() * 1000);
    } while (0);

//Error:
    return cudaStatus;
}


__global__ void addKernel(int *dest, const int *a, const int *b)
{
    int i = threadIdx.x;
    int a0 = a[i];
    int b0 = b[i];
    int c0 = a0 + b0;
    //c[i] = a[i] + b[i];
    dest[i] = c0;

    printf("a=%d,b=%d,c=%d\n", a0, b0, c0);

}

__global__ void convvalue_kernel(uint8_t* dest, const uint8_t* input, const int height, const int bpl, float value)
{
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    if (col < 1920) {
        //int pos = col;
        for (int row = 0; row < height; row++) {
            int pos = row * bpl + col * 3;
            uint8_t c0 =input[pos+0]; 
            uint8_t c1 =input[pos+1]; 
            uint8_t c2 =input[pos+2]; 
            float o0 = c0 * value;
            float o1 = c1 * value;
            float o2 = c2 * value;
            if (o0 > 255) {
                o0 = 255;
            }
            if (o1 > 255) {
                o1 = 255;
            }
            if (o2 > 255) {
                o2 = 255;
            }
            dest[pos+0] = o0;
            dest[pos+1] = o1;
            dest[pos+2] = o2;
        }
    }
}
uint8_t *dev_in_value = nullptr;
uint8_t *dev_out_value = nullptr;
void initconv_value()
{
    cudaError_t cudaStatus;
	//cudaStatus = cudaMallocManaged((void**)&dev_in_value, 1920*1200 * 3 );
	cudaStatus = cudaMalloc((void**)&dev_in_value, 1920*1200 * 3 );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	//cudaStatus = cudaMallocManaged((void**)&dev_out_value, 1920*1200 * 3 );
	cudaStatus = cudaMalloc((void**)&dev_out_value, 1920*1200 * 3 );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
}

cudaError_t convvalue_gpu(uint8_t in[], uint8_t out[], unsigned int height, int bpl, float ratio, bool incopy, bool outcopy)
{
    cudaError_t cudaStatus;
    printf("%d,%d,%f", height, bpl, ratio);

    do {

        if (incopy) {
            // Copy input vectors from host memory to GPU buffers.
            cudaStatus = cudaMemcpy(dev_in_value, in, height * bpl, cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
                break;
            }
        }
        auto t0 = std::chrono::system_clock::now();
        // Launch a kernel on the GPU with one thread for each element.
        //dev_c dest
        // c = a + b
        int thnum = 128;
        //int thnum = 256;
        dim3 blkdim(thnum, 1, 1);
        dim3 grid((1920 + thnum - 1) / thnum, 1, 1);
        //makecontrast_kernel<<< grid, blkdim >>> ((uchar4*)dev_dest, (uchar4*)dev_in, height, max_val, min_val);
        convvalue_kernel<<< grid, blkdim >>> (dev_out_value, dev_in_value, height, bpl, ratio);
        //__global__ void makecontrast(uchar4* dest, const uchar4* input, const int height, uint8_t max_val, uint8_t min_val)


        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			break;
        }
        if (outcopy) {
            // Copy output vector from GPU buffer to host memory.
            cudaStatus = cudaMemcpy(out, dev_out_value, height * bpl, cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!");
                break;
            }
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			break;
        }
        auto t1 = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = t1 - t0;
        printf("convvalue gpu t=%f\n", diff.count() * 1000);
        std::fflush(stdout);
    } while (0);

//Error:
    return cudaStatus;
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int array_size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, array_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, array_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, array_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, array_size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, array_size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //dev_c dest
    // c = a + b
    addKernel<<<1, array_size>>>(dev_c, dev_a, dev_b);

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
    cudaStatus = cudaMemcpy(c, dev_c, array_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
void mymemcpy(uint64_t* dst, uint64_t* src, int len) 
{
    int i;
    int row;
    int col;
    int stride = 1920 * 3 / 8;
    uint64_t* d, * s;
#pragma omp parallel for num_threads(4) private(col, d, s)
    for (row = 0; row < 1200; row++){
        d = dst + stride * row;
        s = src + stride * row;
        for (col = 0; col < (1920*3)/8; col++) {
            d[col] = src[col];
        }
    }
}
//Š¿Žš
int main()
{
    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

    cudaError_t cudaStatus;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 2;
    }
    kernel_init();
#if 0
    int pitch = 2440;
    uint8_t *avebuf = new uint8_t[pitch];
    int height = 220;
    uint8_t *contrastbuf = new uint8_t[pitch*height];
    uint8_t *inbuf = new uint8_t[pitch*height];
#if 1
    puts("\n\ninbuf");
    for(int row = 0; row < height; row++){
        for (int col = 0; col < pitch; col++) {
            int idx = row * pitch + col;
            inbuf[idx] = col;
            //printf("%05x=%02x ", idx, inbuf[idx]);
            //if ((idx & 0xf) == 0xf) {
                //puts("");
            //}
        }
    }
    puts("");
    puts("");
    fflush(stdout);
//    Sleep(1);
#endif
#endif
#if 0
    // Add vectors in parallel.
    cudaStatus = verticalave_gpu(avebuf, inbuf, height);
    cudaStatus = verticalave_gpu(avebuf, inbuf, height);
    cudaStatus = verticalave_gpu(avebuf, inbuf, height);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    verticalave_cpu(avebuf, inbuf, height);
    verticalave_cpu(avebuf, inbuf, height);

    cudaStatus = convcontrast_gpu(contrastbuf, inbuf, height, 128, 1);
    cudaStatus = convcontrast_gpu(contrastbuf, inbuf, height, 128, 1);
    cudaStatus = convcontrast_gpu(contrastbuf, inbuf, height, 128, 1);
    convcontrast_cpu(contrastbuf, inbuf, height, 128, 1);
    convcontrast_cpu(contrastbuf, inbuf, height, 128, 1);
    convcontrast_cpu(contrastbuf, inbuf, height, 128, 1);
#endif
#if 0
    fflush(stdout);
    puts("\n\nave=");
	for (int col = 0; col < pitch; col++) {
		printf("%04x=%02x ", col, avebuf [col]);
        if ((col & 0xf) == 0xf) {
            puts("");
        }
	}
#endif

    initconv_value();
    uint8_t* inbufh;
    cudaMallocHost(&inbufh, 1920 * 1200 * 3);
    uint8_t *outbufm;
	cudaMallocManaged((void**)&outbufm, 1920*1200 * 3 );
    uint8_t *outbufh;
    cudaMallocHost(&outbufh, 1920 * 1200 * 3);
    static uint8_t inbuf[1920 * 1200 * 3];
    static uint8_t outbuf[1920 * 1200 * 3];
    memset(outbuf, 0, 1920 * 1200 * 3);
    uint8_t *outbuf_dev;
	cudaMalloc((void**)&outbuf_dev, 1920*1200 * 3 );
    uint8_t* inbuf_devm;
	cudaMallocManaged((void**)&inbuf_devm, 1920*1200 * 3 );
#if 1
    for (int i = 0; i < 1920*1200*3; i++) {
        inbuf[i] = i;
        inbufh[i] = i;
    }
#endif
    ///convvalue_gpu(1200, 1920*3, 10.1);
    bool incopy = true;
    bool outcopy = true;
    convvalue_gpu(inbuf, outbufm, 1200, 1920*3, 10.1, incopy, outcopy);
    convvalue_gpu(inbuf, outbuf, 1200, 1920*3, 10.1, incopy, outcopy);
    convvalue_gpu(inbufh, outbufh, 1200, 1920*3, 10.1, incopy, outcopy);
    incopy = false; 
    outcopy = false;
    dev_in_value = inbuf_devm;
    dev_out_value = outbufm;
    convvalue_gpu(dev_in_value, outbufm, 1200, 1920*3, 10.1, incopy, outcopy);
    {
        auto ts = omp_get_wtime();
        cudaMemcpy(outbufh, outbufm, 1920 * 1200 * 3, cudaMemcpyHostToHost);
        auto te = omp_get_wtime();
        std::cout << te - ts <<std::endl;
    }
    {
        auto ts = omp_get_wtime();
        mymemcpy((uint64_t*)outbufh, (uint64_t*)outbufm, 1920 * 1200 * 3);
        auto te = omp_get_wtime();
        std::cout << te - ts << "outbufh" << outbufh << "outbufm" << outbufm <<std::endl;
    }

    cudaFree(dev_dest);
    cudaFree(dev_in);


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    getchar();
    return 0;
}

