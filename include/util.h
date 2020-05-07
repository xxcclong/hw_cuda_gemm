#ifndef UTIL_H
#define UTIL_H

#include <cusparse.h>
#include <curand.h>
#include <cublas_v2.h>
#include <iostream>
#include <cuda_profiler_api.h>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <vector>
#include <chrono>
#include <assert.h>
#include <sstream>
#include <assert.h>
#include <pthread.h>
#include <omp.h>
#include <string>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <random>


#include "args.hxx"

using namespace std;

#define timestamp(__var__) auto __var__ = std::chrono::system_clock::now();

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
	_where << __FILE__ << ':' << __LINE__;                             \
	_message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
	std::cerr << _message.str() << "\nAborting...\n";                  \
	cudaDeviceReset();                                                 \
	exit(1);                                                           \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

int M = 100, N = 100, K = 100;
int ITER = 10;

template<class T>
__global__ void validate_kernel(T* ref, T* ans, int num, int* numdiff)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < num)
    {
        if(abs((ref[tid] - ans[tid]) / ref[tid]) > 1e-1)
        {
            atomicAdd(numdiff, 1);
        }
    }
}
__managed__ int diff_num;
template<class T>
int validate(T* d1, T* d2, size_t size)
{
    diff_num = 0;
    validate_kernel<<< (size + 255)/256, 256 >>>(d1, d2, size, &diff_num);
    checkCudaErrors(cudaDeviceSynchronize());
    return diff_num;
}

inline double getDuration(std::chrono::time_point<std::chrono::system_clock> a,
                std::chrono::time_point<std::chrono::system_clock> b) {
    return  std::chrono::duration<double>(b - a).count();
}


#endif
