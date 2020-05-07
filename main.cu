#include<iostream>
#include<sys/time.h>
#include<stdlib.h>
#include<stdio.h>
#include<cuda.h>
#include "util.h"
#include "gemm.h"
#include "args.hxx"

#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8
using namespace std;

template<class T>
__global__ void gemm(T *A, T *B, T *C, int m, int n, int k, T alpha, T beta) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // printf("col %d row %d\n", col, row);
    if( (col < n) && (row < m) )
    {
        T tmp = beta * C[row * n + col];
        for(int i = 0; i < k; ++i)
        {
            tmp += alpha * A[row * k + i] * B[col + i * n];
        }
        C[row * n + col] = tmp;
    }
}
#define GFLOP (((1e-9)*M*N*K*2))

cublasHandle_t cublasH;

void argParse(int argc, char ** argv)
{
    args::ArgumentParser parser("GEMM parameters", "");
    args::ValueFlag<int> arg_run_iter(parser, "run iter", "", {"iter"});
    args::ValueFlag<int> arg_m(parser, "matrix m", "", {"m"});
    args::ValueFlag<int> arg_n(parser, "matrix n", "", {"n"});
    args::ValueFlag<int> arg_k(parser, "matrix k", "", {"k"});

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help)
    {
        std::cout << parser;
        exit(0);
    }
    catch (args::ParseError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        exit(1);
    }
    catch (args::ValidationError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        exit(1);
    }
    if(bool{arg_run_iter})
    {
        ITER = args::get(arg_run_iter);
    }
    if(bool{arg_m})
    {
        M = args::get(arg_m);
    }
    if(bool{arg_n})
    {
        N = args::get(arg_n);
    }
    if(bool{arg_k})
    {
        K = args::get(arg_k);
    }
    fprintf(stderr, 
        "*****************************************\n"
        "iterations: %d\n"
        "m: %d\n"
        "n: %d\n"
        "k: %d\n"
        "*****************************************\n"
        ,ITER, M, N, K);
}

template <typename T>
typename std::enable_if<(std::is_same<T, float>::value)>::type
cublasGemm(T* A, T* B, T* C, T alpha, T beta)
{
    checkCudaErrors(cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_T,
        M, N, K, 
        &alpha,
        A, K,
        B, N,
        &beta, 
        C, M));       
}

template <typename T>
typename std::enable_if<(std::is_same<T, double>::value)>::type
cublasGemm(T* A, T* B, T* C, T alpha, T beta)
{
    checkCudaErrors(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_T,
        M, N, K, 
        &alpha,
        A, K,
        B, N,
        &beta, 
        C, M));
}


template <typename T>
typename std::enable_if<(std::is_same<T, float>::value)>::type
transpose(T* in, T* out)
{
    T alpha = 1;
    checkCudaErrors(cublasSgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 
        N, M, 
        &alpha, 
        in, M, 
        NULL, 
        NULL, N, 
        out, N));
}

template <typename T>
typename std::enable_if<(std::is_same<T, double>::value)>::type
transpose(T* in, T* out)
{
    T alpha = 1;
    checkCudaErrors(cublasDgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 
        N, M, 
        &alpha, 
        in, M, 
        NULL, 
        NULL, N, 
        out, N));
}

template <typename T>
typename std::enable_if<(std::is_same<T, float>::value)>::type
curandGen(curandGenerator_t curand, T* p, size_t size)
{
    curandGenerateUniform(curand, p, size);
}

template <typename T>
typename std::enable_if<(std::is_same<T, double>::value)>::type
curandGen(curandGenerator_t curand, T* p, size_t size)
{
    curandGenerateUniformDouble(curand, p, size);
}



template<class T>
void run()
{
    // rand init
    curandGenerator_t curand;
    curandCreateGenerator(&curand, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand, 123ULL);
    T *A, *B, *C, *C_gold, *tmp;
    checkCudaErrors(cudaMallocManaged(&A,      M * K * sizeof(T)));
    checkCudaErrors(cudaMallocManaged(&B,      K * N * sizeof(T)));
    checkCudaErrors(cudaMallocManaged(&C,      M * N * sizeof(T)));
    checkCudaErrors(cudaMallocManaged(&C_gold, M * N * sizeof(T)));
    checkCudaErrors(cudaMallocManaged(&tmp,    M * N * sizeof(T)));
    curandGen(curand, A, M * K);
    curandGen(curand, B, K * N);

    T alpha = 1.5;
    T beta = 0.5;


    curandGen(curand, C, M * N);
    checkCudaErrors(cudaMemcpy(tmp, C, M * N * sizeof(T), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(C_gold, 0, sizeof(T) * M * N));

    // correctness validation
    cublasGemm(A, B, tmp, alpha, beta);
    transpose(tmp, C_gold);
    checkCudaErrors(cudaDeviceSynchronize());

    myGEMM(A, B, C, alpha, beta);
    checkCudaErrors(cudaDeviceSynchronize());
    
    // baseline implementation which has right answer
    // dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    // dim3 grid( (N + block.x - 1) / block.x, (M + block.y - 1) / block.y );
    // gemm <<<grid, block>>>
    //     (A, B, C, M, N, K, alpha, beta);

    
    checkCudaErrors(cudaDeviceSynchronize());
    int errornum = validate(C_gold, C, M * N);
    checkCudaErrors(cudaDeviceSynchronize());
    if( errornum != 0)
    {
        printf("%d errors compared with correct answer\n", errornum);
        // exit(0);
    }
    else
    {
        printf("Validation PASS\n");
    }



    // baseline
    double t_base = 0;
    for(int i = 0; i < ITER; ++i)
    {
        dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
        dim3 grid( (N + block.x - 1) / block.x, (M + block.y - 1) / block.y );
        timestamp(t0);
        gemm <<<grid, block>>>
            (A, B, C, M, N, K, alpha, beta);
        checkCudaErrors(cudaDeviceSynchronize());
        timestamp(t1);
        t_base += getDuration(t0, t1);
    }
    printf("t_baseline=%f GFLOPS_baseline=%f\n", t_base / ITER, GFLOP / (t_base / ITER));


    double t_cublas = 0.f;
    for(int i = 0; i < ITER; ++i)
    {
        timestamp(t0);
        cublasGemm(A, B, tmp, alpha, beta);
        checkCudaErrors(cudaDeviceSynchronize());
        timestamp(t1);
        t_cublas += getDuration(t0, t1);
    }
    printf("t_cublas=%f GFLOPS_cublas=%f\n", t_cublas / ITER, GFLOP / (t_cublas / ITER));
    assert(GFLOP > 0.f);
    assert((t_cublas / ITER) > 0.f);
    assert(GFLOP / (t_cublas / ITER) > 0.f);

    double t_hw = 0;
    bool preprocessed = false;
    for(int i = 0; i < ITER; ++i)
    {
        timestamp(t0);
        double t = myGEMM(A, B, C, alpha, beta);
        checkCudaErrors(cudaDeviceSynchronize());
        timestamp(t1);
        if(t == 0.f)
        {
            t_hw += getDuration(t0, t1);
        }
        else
        {
            t_hw += t;
            preprocessed = true;
        }
    }
    printf("t_hw=%f GFLOPS_hw=%f\n", t_hw / ITER, GFLOP / (t_hw / ITER));
    if(preprocessed)
    {
        printf("preprocessed\n");
    }


     
}

int main(int argc, char ** argv)
{
    checkCudaErrors(cublasCreate(&cublasH));
    argParse(argc, argv);
    printf("\nEval on float\n");
    run<float>();
    printf("\nEval on double\n");
    run<double>();

    return 0;
}
