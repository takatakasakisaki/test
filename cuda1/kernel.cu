
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <chrono>
#include <valarray>
#include <iostream>

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
            max0 = max(p, max0);
            min0 = min(p, min0);
#if 0
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

uint8_t *dev_in = nullptr;
uint8_t *dev_dest = nullptr;
uint8_t *dev_maxval = nullptr;
uint8_t *dev_minval = nullptr;
void kernel_init()
{
    cudaError_t cudaStatus;
    int pitch = 2440;
    int height = 220;
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_dest, pitch * sizeof(uint8_t));
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
}
// Helper function for using CUDA to add vectors in parallel.
cudaError_t verticalave(uint8_t *dest, const uint8_t *in, unsigned int height)
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
        int thnum = 128;
        dim3 blkdim(thnum, 1, 1);
        dim3 grid((2440 + thnum - 1) / thnum, 1, 1);
        verticalave_kernel <<<grid, blkdim >>> (dev_dest, dev_in, height, dev_maxval, dev_minval);

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
        printf("t=%f,%d,%d\n", diff.count()*1000, h_maxa.max(), h_mina.min());
#undef max
#undef min
        for (int i = 0; i < 2440; i++) {
            printf("%x, %02x,%02x\n", i, h_max[i], h_min[i]);
        }
        std::fflush(stdout);
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
    int pitch = 2440;
    uint8_t *avebuf = new uint8_t[pitch];
    int height = 220;
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
    // Add vectors in parallel.
    cudaStatus = verticalave(avebuf, inbuf, height);
    cudaStatus = verticalave(avebuf, inbuf, height);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
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
    cudaFree(dev_dest);
    cudaFree(dev_in);


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

