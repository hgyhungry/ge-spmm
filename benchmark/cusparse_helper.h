// file: common_util.hpp
//
// Encapsule cusparse csr-spmm APIs
//  require cuda version >= 11.0
//  author: guyue huang
//  date  : 2021/06/29

#include "common_util.h"
#include "cusparse.h"

#define CUSPARSE_CHECK(func)    \
{                                                                           \
    cusparseStatus_t status = ( func );                                     \
    if (status != CUSPARSE_STATUS_SUCCESS)                                  \
    {                                                                       \
        std::cerr << "Got cusparse error"                                   \
                << " at line: " << __LINE__                                 \
                << " of file " << __FILE__ << std::endl;                    \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
}

struct CusparseCsrSpMMProblem {
    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;


    float alpha;
    float beta;
    int nr; // number of sparse matrix rows
    int nc; // number of sparse matrix columns
    int nnz; // number of sparse matrix non-zeros
    int maxNv; // maximum row dimension of dense matrices
    int *rowPtr; // pointer to csr row_offset array
    int *colIdx; // pointer to csr row_indices array
    float *values; // pointer to csr values array 
    float *dnInput; // pointer to the dense input matrix of nc*maxNv
    float *dnOutput; // pointer to the dense output matrix of size nr*maxNv

    CusparseCsrSpMMProblem(int nr,
                            int nc,
                            int nnz, 
                            int maxNv,
                            int *rowPtr,
                            int *colIdx,
                            float *values,
                            float *dnInput,
                            float *dnOutput) :
        nr(nr), nc(nc), nnz(nnz), maxNv(maxNv), 
        rowPtr(rowPtr), colIdx(colIdx), values(values), 
        dnInput(dnInput), dnOutput(dnOutput), 
        alpha(1.0f), beta(0.0f)
    {
        CUSPARSE_CHECK( cusparseCreate(&handle));
        CUSPARSE_CHECK( cusparseCreateCsr(&matA,
                            nr, nc, nnz, rowPtr, colIdx, values,
                            CUSPARSE_INDEX_32I, // index 32-integer for indptr
                            CUSPARSE_INDEX_32I, // index 32-integer for indices
                            CUSPARSE_INDEX_BASE_ZERO,
                            CUDA_R_32F          // datatype: 32-bit float real number
                        ));
    }

    void run(cusparseSpMMAlg_t alg, int N) {
        assert(N <= maxNv);
        CUSPARSE_CHECK(cusparseCreateDnMat(&matB,
                                            nc,
                                            N,
                                            N,
                                            dnInput,
                                            CUDA_R_32F,
                                            CUSPARSE_ORDER_ROW
        ));
        CUSPARSE_CHECK(cusparseCreateDnMat(&matC,
                                            nr,
                                            N,
                                            N,
                                            dnOutput,
                                            CUDA_R_32F,
                                            CUSPARSE_ORDER_ROW
        ));        

        void *workspace;
        size_t workspace_size;

        CUSPARSE_CHECK( cusparseSpMM_bufferSize(handle, 
                                                CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha,
                                                matA,
                                                matB,
                                                &beta,
                                                matC,
                                                CUDA_R_32F,
                                                alg,
                                                &workspace_size
                                                ));           
        CUDA_CHECK( cudaMalloc(&workspace, workspace_size));

        // run SpMM
        CUSPARSE_CHECK( cusparseSpMM(handle, 
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                                    &alpha, 
                                    matA,
                                    matB, 
                                    &beta,
                                    matC,
                                    CUDA_R_32F,
                                    alg,
                                    workspace) );        

        CUDA_CHECK( cudaFree(workspace));
    }

    float benchmark (cusparseSpMMAlg_t alg, int N, int warmup = 5, int repeat = 50){
        assert(N <= maxNv);
        CUSPARSE_CHECK(cusparseCreateDnMat(&matB,
                                            nc,
                                            N,
                                            N,
                                            dnInput,
                                            CUDA_R_32F,
                                            CUSPARSE_ORDER_ROW
        ));
        CUSPARSE_CHECK(cusparseCreateDnMat(&matC,
                                            nr,
                                            N,
                                            N,
                                            dnOutput,
                                            CUDA_R_32F,
                                            CUSPARSE_ORDER_ROW
        ));        

        void *workspace;
        size_t workspace_size;

        CUSPARSE_CHECK( cusparseSpMM_bufferSize(handle, 
                                                CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha,
                                                matA,
                                                matB,
                                                &beta,
                                                matC,
                                                CUDA_R_32F,
                                                alg,
                                                &workspace_size
                                                ));           
        CUDA_CHECK( cudaMalloc(&workspace, workspace_size));

        GpuTimer timer;

        for (int i = 0; i < warmup + repeat; i++) {
            if (i == warmup) {
                timer.start();
            }
            cusparseSpMM(handle, 
                        CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                        CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                        &alpha, 
                        matA,
                        matB, 
                        &beta,
                        matC,
                        CUDA_R_32F,
                        alg,
                        workspace);
        }
        timer.stop();

        float dur = timer.elapsed_msecs() / repeat;

        float MFlop_count = (float)nnz / 1e6 * N * 2;

        float gflops = MFlop_count / dur;      

        CUDA_CHECK( cudaFree(workspace));

        return gflops;
    }
};

