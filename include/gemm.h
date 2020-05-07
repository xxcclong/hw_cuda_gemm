// studentid: 2016123456
#include "util.h"

template<class T>
double myGEMM(T* A, T* B, T* C, T alpha, T beta)
{
	printf("perform your gemm here on m=%d n=%d k=%d\n", M, N, K);
	bool preprocess = false;
	if(preprocess)
	{
		// your preprocess
		timestamp(t0);
		// your gemm

		checkCudaErrors(cudaDeviceSynchronize());
		timestamp(t1);
		return getDuration(t0, t1);
	}
	else
	{
		// your gemm

		checkCudaErrors(cudaDeviceSynchronize());
		return 0.f;	
	}
	
}
