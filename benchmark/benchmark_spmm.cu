// file: benchmark_spmm.cu
//
// Using cusparse API and ge-spmm kernels to test SpMM performance.
//  author: guyue huang
//  date  : 2021/06/29
// compile: nvcc version >=11.0


#include <cuda_runtime_api.h>   // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>           // cusparseSpMM (>= v11.0) or cusparseScsrmm
#include <stdio.h>
#include <stdlib.h>             
#include <cstdlib>              // std::rand(), RAND_MAX
#include <vector>     
#include "common_util.h"      // read_mtx
#include "cusparse_helper.h"
#include "../ge-spmm/dispatch.h"

int main(int argc, const char** argv) {

    /// check command-line argument

    if (argc < 2) {
        printf("Require command-line argument: name of the sparse matrix file in .mtx format.\n");
        return EXIT_FAILURE;
    }
    
    //
    // Load sparse matrix
    //

    int M;   // number of A-rows
    int K;   // number of A-columns
    int nnz; // number of non-zeros in A
    std::vector<int> csr_indptr_buffer;     // buffer for indptr array in CSR format
    std::vector<int> csr_indices_buffer;    // buffer for indices (column-ids) array in CSR format
    // load sparse matrix from mtx file
    read_mtx_file(argv[1], 
                M, 
                K, 
                nnz, 
                csr_indptr_buffer, 
                csr_indices_buffer);
    printf("Finish reading matrix %d rows, %d columns, %d nnz. \nIgnore original values and use randomly generated values.\n", M, K, nnz);
    
    // Create GPU arrays
    int maxN = 128;   // number of B-columns
    float *B_h = NULL, *C_h = NULL, *csr_values_h = NULL, *C_ref = NULL;
    float *B_d = NULL, *C_d = NULL, *csr_values_d = NULL;
    int   *csr_indptr_d = NULL, *csr_indices_d = NULL;


    B_h   = (float*) malloc(sizeof(float) * K * maxN);
    C_h   = (float*) malloc(sizeof(float) * M * maxN);
    C_ref = (float*) malloc(sizeof(float) * M * maxN);
    csr_values_h = (float*) malloc(sizeof(float) * nnz);
    if (!B_h || !C_h || !C_ref || !csr_values_h) {
        printf("Host allocation failed.\n");
        return EXIT_FAILURE;
    } 

    fill_random(csr_values_h, nnz);
    fill_random(B_h, K * maxN);

    CUDA_CHECK( cudaMalloc((void**)&B_d, sizeof(float) * K * maxN) );
    CUDA_CHECK( cudaMalloc((void**)&C_d, sizeof(float) * M * maxN) );
    CUDA_CHECK( cudaMalloc((void**)&csr_values_d, sizeof(float) * nnz) );
    CUDA_CHECK( cudaMalloc((void**)&csr_indptr_d, sizeof(int) * (M + 1)) );
    CUDA_CHECK( cudaMalloc((void**)&csr_indices_d, sizeof(int) * nnz) );

    CUDA_CHECK( cudaMemcpy(B_d, B_h, sizeof(float) * K * maxN, cudaMemcpyHostToDevice));
    CUDA_CHECK( cudaMemset(C_d, 0x0, sizeof(float) * M * maxN));
    CUDA_CHECK( cudaMemcpy(csr_values_d, csr_values_h, sizeof(float)*nnz, cudaMemcpyHostToDevice));
    CUDA_CHECK( cudaMemcpy(csr_indptr_d, csr_indptr_buffer.data(), sizeof(int) * (M+1), cudaMemcpyHostToDevice));
    CUDA_CHECK( cudaMemcpy(csr_indices_d, csr_indices_buffer.data(), sizeof(int)*nnz, cudaMemcpyHostToDevice));

    //
    // Run SpMM and check result
    //

    CusparseCsrSpMMProblem problem(M,
                                K,
                                nnz,
                                maxN,
                                csr_indptr_d,
                                csr_indices_d,
                                csr_values_d,
                                B_d,
                                C_d
                            );

    cusparseSpMMAlg_t all_cusparse_algs[2] = { CUSPARSE_SPMM_ALG_DEFAULT, CUSPARSE_SPMM_CSR_ALG2};

    gespmmAlg_t all_gespmm_algs[6] = { 
                                    GESPMM_ALG_PARREDUCE_ROWBALANCE,
                                    GESPMM_ALG_PARREDUCE_NNZBALANCE,
                                    GESPMM_ALG_SEQREDUCE_ROWBALANCE,
                                    GESPMM_ALG_SEQREDUCE_NNZBALANCE,
                                    GESPMM_ALG_ROWCACHING_ROWBALANCE,
                                    GESPMM_ALG_ROWCACHING_NNZBALANCE };


    bool all_passed = true;


    for (cusparseSpMMAlg_t alg : all_cusparse_algs) {
        for (int N = 1; N <= maxN; N = N * 2) {
            
            CUDA_CHECK( cudaMemset(C_d, 0x0, sizeof(float)* M * maxN));

            problem.run(alg, N);

            CUDA_CHECK( cudaMemcpy(C_h, C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost) );

            spmm_reference_host<int, float>(M, N, K, csr_indptr_buffer.data(), csr_indices_buffer.data(), csr_values_h, B_h, C_ref);

            bool correct = check_result<float>(M, N, C_h, C_ref);
            std::string test_result = correct ? "Pass" : "Fail";
            std::cout << "Cusparse Algorithm " << alg << " N " << N << " Verify Correct: " << test_result << std::endl;
            
            all_passed = correct && all_passed;
        }
    }

    SpMatCsrDescr_t matA = {M, K, nnz, csr_indptr_d, csr_indices_d, csr_values_d};

    for (gespmmAlg_t alg : all_gespmm_algs) {
        for (int N = 1; N <= maxN; N = N * 2) {
            
            CUDA_CHECK( cudaMemset(C_d, 0x0, sizeof(float)* M * maxN));

            gespmmCsrSpMM(matA, B_d, N, C_d, alg);

            cudaDeviceSynchronize();
            CUDA_CHECK( cudaGetLastError());

            CUDA_CHECK( cudaMemcpy(C_h, C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost) );

            spmm_reference_host<int, float>(M, N, K, csr_indptr_buffer.data(), csr_indices_buffer.data(), csr_values_h, B_h, C_ref);

            bool correct = check_result<float>(M, N, C_h, C_ref);
            std::string test_result = correct ? "Pass" : "Fail";
            std::cout << "Ge-spmm Algorithm " << alg << " N " << N << " Verify Correct: " << test_result << std::endl;
            
            all_passed = correct && all_passed;
        }        
    }


    
    //
    // Benchmark SpMM performance
    //
    if (all_passed) {
        printf("\nBenchmark:\n");
        
        for (cusparseSpMMAlg_t alg : all_cusparse_algs) {
            for (int N = 1; N <= maxN; N = N * 2) {

                float gflops = problem.benchmark(alg, N);

                std::cout << "Cusparse Algorithm " << alg << " N " << N << " Throughput " << gflops << "(gflops) " << std::endl;
            }
        }

        for (gespmmAlg_t alg : all_gespmm_algs) {
            for (int N = 1; N <= maxN; N = N * 2) {
                GpuTimer gpu_timer;
        
                int warmup_iter = 5;
                int repeat_iter = 50;
                for (int iter = 0; iter < warmup_iter + repeat_iter; iter++) {
                    if (iter == warmup_iter) {
                        gpu_timer.start();
                    }
        
                    gespmmCsrSpMM(matA, B_d, N, C_d, alg);
        
                }
                gpu_timer.stop();
        
                float kernel_dur_msecs = gpu_timer.elapsed_msecs() / repeat_iter;
        
                float MFlop_count = (float)nnz / 1e6 * N * 2;
        
                float gflops = MFlop_count / kernel_dur_msecs;
        
                std::cout << "Ge-spmm Algorithm " << alg << " N " << N << " Throughput " << gflops << "(gflops) " << std::endl;
        }}
    }
       
    /// free memory

    if (B_h)            free(B_h);
    if (C_h)            free(C_h);
    if (C_ref)          free(C_ref);
    if (csr_values_h)   free(csr_values_h);
    if (B_d)            CUDA_CHECK( cudaFree(B_d) );
    if (C_d)            CUDA_CHECK( cudaFree(C_d) );
    if (csr_values_d)   CUDA_CHECK( cudaFree(csr_values_d) );
    if (csr_indptr_d)   CUDA_CHECK( cudaFree(csr_indptr_d) );
    if (csr_indices_d)  CUDA_CHECK( cudaFree(csr_indices_d) );
    
    return EXIT_SUCCESS;
}
