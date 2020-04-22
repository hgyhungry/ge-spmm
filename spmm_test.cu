#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <sys/time.h>
#include <stdexcept>
#include <fstream>
#include <ctime>

#include <cuda_runtime.h>
#include "cusparse.h"
// #include "cublas_v2.h"

#include "./util/mmio.hpp"
#include "./util/util.hpp"

using namespace std;

// #define VALIDATE

#define checkCudaError( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    CLEANUP("Exit.");   \
    exit(EXIT_FAILURE); \
    } \
} while(0)

#define checkCuSparseError( a ) do { \
    if (CUSPARSE_STATUS_SUCCESS != (a)) { \
    fprintf(stderr, "CuSparse runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    CLEANUP("Exit.");   \
    exit(EXIT_FAILURE); \
    } \
} while (0)

#define CLEANUP(s)                      \
do {                                    \
    printf("%s", s);                  \
    if (A_data) free(A_data);   \
    if (A_indptr) free(A_indptr);   \
    if (A_indices) free(A_indices); \
    if (B)  free(B);    \
    if (C)  free(C);    \
    if (golden) free(golden);   \
    if (A_data_dev) cudaFree(A_data_dev);     \
    if (A_indptr_dev) cudaFree(A_indptr_dev);         \
    if (A_indices_dev) cudaFree(A_indices_dev);     \
    if (B_dev) cudaFree(B_dev); \
    if (C_dev) cudaFree(C_dev); \
    if (start)      cudaEventDestroy(start);    \
    if (stop)       cudaEventDestroy(stop);     \
    if (descr)      cusparseDestroyMatDescr(descr);     \
    if (cusp_handle)     cusparseDestroy(cusp_handle);    \
    fclose(fpo);\
    cudaDeviceReset();                  \
    fflush(stdout);                     \
} while (0)

__global__ void warmup(){}

template<typename T>
__global__ void spmm_test0(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, T* A_csrVal,
    T* B_dnVal, T* C_dnVal
)
{
    int rid = blockDim.y*blockIdx.x+threadIdx.y;
    if (rid<A_nrows) {
    int cid = (blockIdx.y<<5)+threadIdx.x;
    int lb = A_csrRowPtr[rid];
    int hb = A_csrRowPtr[(rid+1)];
    int offset = 0;
    T acc=0;
    if (blockIdx.y!=gridDim.y-1){
        for (int ptr = lb; ptr<hb; ptr++) {
            offset = A_csrColInd[ptr]*B_ncols+cid;
            acc += A_csrVal[ptr]*B_dnVal[offset];
        }
        C_dnVal[(rid*B_ncols+cid)] = acc;
    }
    else {
        for (int ptr = lb; ptr<hb; ptr++) {
            if (cid<B_ncols) {
            offset = A_csrColInd[ptr]*B_ncols+cid;}
            acc += A_csrVal[ptr]*B_dnVal[offset];
        }
        if (cid<B_ncols) {
        C_dnVal[(rid*B_ncols+cid)] = acc;}
    }
    }
}

template<typename T>
__global__ void spmm_test1(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, T* A_csrVal,
    T* B_dnVal, T* C_dnVal
)
{
    extern __shared__ int sh[];
    int *colInd_sh = sh;
    T *val_sh = (T *)&sh[(blockDim.y<<5)];
    int shmem_offset = (threadIdx.y<<5);
    int thread_idx = shmem_offset+threadIdx.x;

    int rid = blockDim.y*blockIdx.x+threadIdx.y;
    
    if (rid<A_nrows) {
        int cid = (blockIdx.y<<5)+threadIdx.x;
        int lb = A_csrRowPtr[rid];
        int hb = A_csrRowPtr[(rid+1)];
        int ptr = lb+threadIdx.x;
        int offset;
        T acc=0;

        if (blockIdx.y != gridDim.y-1) {
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = A_csrVal[ptr];
                    colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    acc += val_sh[(shmem_offset+kk)]*B_dnVal[offset];
                }
                __syncwarp();
            }
            C_dnVal[(rid*B_ncols+cid)] = acc;
        }
        else {
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = A_csrVal[ptr];
                    colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    if (cid<B_ncols) {
                    acc += val_sh[(shmem_offset+kk)]*B_dnVal[offset];
                    }
                }
                __syncwarp();
            }
            if (cid<B_ncols) {
            C_dnVal[(rid*B_ncols+cid)] = acc;
            }
        }
    }
}

template<typename T>
__global__ void spmm_test2(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, T* A_csrVal,
    T* B_dnVal, T* C_dnVal
)
{
    extern __shared__ int sh[];
    int *colInd_sh = sh;
    T *val_sh = (T *)&sh[(blockDim.y<<5)];
    int shmem_offset = (threadIdx.y<<5);
    int thread_idx = shmem_offset+threadIdx.x;

    int rid = blockDim.y*blockIdx.x+threadIdx.y;
    
    if (rid<A_nrows) {
        int cid = (blockIdx.y<<6)+threadIdx.x;
        int lb = A_csrRowPtr[rid];
        int hb = A_csrRowPtr[(rid+1)];
        int ptr = lb+threadIdx.x;
        int offset;
        T acc1=0, acc2=0, val;

        if (blockIdx.y != gridDim.y-1) {
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = A_csrVal[ptr];
                    colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    val = val_sh[(shmem_offset+kk)];
                    acc1 += val*B_dnVal[offset];
                    acc2 += val*B_dnVal[offset+32];
                }
                __syncwarp();
            }
            offset = rid*B_ncols+cid;
            C_dnVal[offset] = acc1;
            C_dnVal[offset+32] = acc2;
        }
        else {
            int nout = (B_ncols-cid+31)/32;
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = A_csrVal[ptr];
                    colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    val = val_sh[(shmem_offset+kk)];
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    if (nout>0) {
                    acc1 += val*B_dnVal[offset];
                    }
                    if (nout>1) {
                    acc2 += val*B_dnVal[offset+32];  
                    }
                }
                __syncwarp();
            }
            offset = rid*B_ncols+cid;
            if (nout>0) {
            C_dnVal[offset] = acc1;
            }
            if (nout>1) {
            C_dnVal[(offset+32)] = acc2;
            }
        }
    }
}

template<typename T>
__global__ void spmm_test3(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, T* A_csrVal,
    T* B_dnVal, T* C_dnVal
)
{
    extern __shared__ int sh[];
    int *colInd_sh = sh;
    T *val_sh = (T *)&sh[(blockDim.y<<5)];
    int shmem_offset = (threadIdx.y<<5);
    int thread_idx = shmem_offset+threadIdx.x;

    int rid = blockDim.y*blockIdx.x+threadIdx.y;
    
    if (rid<A_nrows) {
        int cid = (blockIdx.y<<7)+threadIdx.x;
        int lb = A_csrRowPtr[rid];
        int hb = A_csrRowPtr[(rid+1)];
        int ptr = lb+threadIdx.x;
        int offset;
        T acc1=0, acc2=0, acc3=0, acc4=0, val;

        if (blockIdx.y != gridDim.y-1) {
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = A_csrVal[ptr];
                    colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    val = val_sh[(shmem_offset+kk)];
                    acc1 += val*B_dnVal[offset];
                    acc2 += val*B_dnVal[offset+32];
                    acc3 += val*B_dnVal[offset+64];
                    acc4 += val*B_dnVal[offset+96];
                }
                __syncwarp();
            }
            offset = rid*B_ncols+cid;
            C_dnVal[offset] = acc1;
            C_dnVal[offset+32] = acc2;
            C_dnVal[offset+64] = acc3;
            C_dnVal[offset+96] = acc4;
        }
        else {
            int nout = (B_ncols-cid+31)/32;
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = A_csrVal[ptr];
                    colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    val = val_sh[(shmem_offset+kk)];
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    if (nout>0) {
                    acc1 += val*B_dnVal[offset];
                    }
                    if (nout>1) {
                    acc2 += val*B_dnVal[offset+32];  
                    }
                    if (nout>2) {
                    acc3 += val*B_dnVal[offset+64];
                    }
                    if (nout>3) {
                    acc4 += val*B_dnVal[offset+96];  
                    }
                }
                __syncwarp();
            }
            offset = rid*B_ncols+cid;
            if (nout>0) {
            C_dnVal[offset] = acc1;
            }
            if (nout>1) {
            C_dnVal[(offset+32)] = acc2;
            }
            if (nout>2) {
            C_dnVal[(offset+64)] = acc3;
            }
            if (nout>3) {
            C_dnVal[(offset+96)] = acc4;
            }
        }
    }
}

template<typename T>
__global__ void spmm_test4(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, T* A_csrVal,
    T* B_dnVal, T* C_dnVal
)
{
    extern __shared__ int sh[];
    int *colInd_sh = sh;
    T *val_sh = (T *)&sh[(blockDim.y<<5)];
    int shmem_offset = (threadIdx.y<<5);
    int thread_idx = shmem_offset+threadIdx.x;

    int rid = blockDim.y*blockIdx.x+threadIdx.y;
    
    if (rid<A_nrows) {
        int cid = (blockIdx.y<<8)+threadIdx.x;
        int lb = A_csrRowPtr[rid];
        int hb = A_csrRowPtr[(rid+1)];
        int ptr = lb+threadIdx.x;
        int offset;
        T acc1=0, acc2=0, acc3=0, acc4=0, acc5=0,acc6=0,acc7=0,acc8=0,val;

        if (blockIdx.y != gridDim.y-1) {
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = A_csrVal[ptr];
                    colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    val = val_sh[(shmem_offset+kk)];
                    acc1 += val*B_dnVal[offset];
                    acc2 += val*B_dnVal[offset+32];
                    acc3 += val*B_dnVal[offset+64];
                    acc4 += val*B_dnVal[offset+96];
                    acc5 += val*B_dnVal[offset+128];
                    acc6 += val*B_dnVal[offset+160];
                    acc7 += val*B_dnVal[offset+192];
                    acc8 += val*B_dnVal[offset+224];
                }
                __syncwarp();
            }
            offset = rid*B_ncols+cid;
            C_dnVal[offset] = acc1;
            C_dnVal[offset+32] = acc2;
            C_dnVal[offset+64] = acc3;
            C_dnVal[offset+96] = acc4;
            C_dnVal[offset+128] = acc5;
            C_dnVal[offset+160] = acc6;
            C_dnVal[offset+192] = acc7;
            C_dnVal[offset+224] = acc8;
        }
        else {
            int nout = (B_ncols-cid+31)/32;
            for (int jj=lb; jj<hb; jj+=32) {
                if (ptr<hb) {
                    val_sh[thread_idx] = A_csrVal[ptr];
                    colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
                }
                __syncwarp();
                ptr += 32;

                for (int kk=0; kk<32&&jj+kk<hb; kk++) {
                    val = val_sh[(shmem_offset+kk)];
                    offset = colInd_sh[(shmem_offset+kk)] + cid;
                    if (nout>0) {
                    acc1 += val*B_dnVal[offset];
                    }
                    if (nout>1) {
                    acc2 += val*B_dnVal[offset+32];  
                    }
                    if (nout>2) {
                    acc3 += val*B_dnVal[offset+64];
                    }
                    if (nout>3) {
                    acc4 += val*B_dnVal[offset+96];  
                    }
                    if (nout>4) {
                        acc5 += val*B_dnVal[offset+128];  
                    }
                    if (nout>5) {
                        acc6 += val*B_dnVal[offset+160];  
                    }
                    if (nout>6) {
                        acc7 += val*B_dnVal[offset+192];  
                    }
                    if (nout>7) {
                        acc8 += val*B_dnVal[offset+224];  
                    }
                }
                __syncwarp();
            }
            offset = rid*B_ncols+cid;
            if (nout>0) {
            C_dnVal[offset] = acc1;
            }
            if (nout>1) {
            C_dnVal[(offset+32)] = acc2;
            }
            if (nout>2) {
            C_dnVal[(offset+64)] = acc3;
            }
            if (nout>3) {
            C_dnVal[(offset+96)] = acc4;
            }
            if (nout>4) {
                C_dnVal[(offset+128)] = acc5;
            }
            if (nout>5) {
                C_dnVal[(offset+160)] = acc6;
            }
            if (nout>6) {
                C_dnVal[(offset+192)] = acc7;
            }
            if (nout>7) {
                C_dnVal[(offset+224)] = acc8;
            }
        }
    }
}

void spmmWrapper(int method, int tile_row, int A_nrows, int B_ncols, int *A_rowPtr, int *A_colInd, float *A_val, float *B, float *C) {
    switch(method) {
        case 0:
        if (B_ncols>32) {
            spmm_test0<float><<<dim3((A_nrows+tile_row-1)/tile_row, (B_ncols+31)/32, 1), dim3(32, tile_row, 1),0,0>>>(
                A_nrows, B_ncols, A_rowPtr, A_colInd, A_val, B, C
            );
        }
        else {
            spmm_test0<float><<<dim3((A_nrows+tile_row-1)/tile_row, 1, 1), dim3(B_ncols, tile_row, 1),0,0>>>(
                A_nrows, B_ncols, A_rowPtr, A_colInd, A_val, B, C
            );
        }
        break;
        case 1:
        spmm_test1<float><<<dim3((A_nrows+tile_row-1)/tile_row, (B_ncols+31)/32, 1), dim3(32, tile_row, 1), 32*tile_row*(sizeof(int)+sizeof(float)),0>>> (
            A_nrows, B_ncols, A_rowPtr, A_colInd, A_val, B, C
        );
        break;
        case 2:
        spmm_test2<float><<<dim3((A_nrows+tile_row-1)/tile_row, (B_ncols+63)/64, 1), dim3(32, tile_row, 1), 32*tile_row*(sizeof(int)+sizeof(float)),0>>> (
            A_nrows, B_ncols, A_rowPtr, A_colInd, A_val, B, C
        );
        break;
        case 3:
        spmm_test3<float><<<dim3((A_nrows+tile_row-1)/tile_row, (B_ncols+127)/128, 1), dim3(32, tile_row, 1), 32*tile_row*(sizeof(int)+sizeof(float)),0>>> (
            A_nrows, B_ncols, A_rowPtr, A_colInd, A_val, B, C
        );
        break;
        case 4:
        spmm_test4<float><<<dim3((A_nrows+tile_row-1)/tile_row, (B_ncols+255)/256, 1), dim3(32, tile_row, 1), 32*tile_row*(sizeof(int)+sizeof(float)),0>>> (
            A_nrows, B_ncols, A_rowPtr, A_colInd, A_val, B, C
        );
        break;

    }
}


int main(int argc, char** argv) {

    int A_nrows, A_ncols, nnz, B_ncols, max_ncols=512;
    int DEV_ID;
    if (argc>2) {
        DEV_ID = atoi(argv[2]);
    }
    else {
        DEV_ID = 0;
    }

    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<float> values;

    // Host allocate
    int* A_indptr = 0;
    int* A_indices = 0;
    float* A_data = 0;
    float* B = 0;
    float* C = 0;
    float* golden = 0;
    float* A_data_dev = 0;
    int* A_indices_dev = 0;
    int* A_indptr_dev = 0;
    float* B_dev = 0;
    float* C_dev = 0;
    // float* C_tran_dev = 0;
    float rt, rt2;
    float one=1, zero=0;
    double gflop;
    

    cudaEvent_t start, stop;
    cusparseHandle_t cusp_handle=0;
    // cublasHandle_t cubl_handle=0;
    cusparseMatDescr_t descr=0;
    cusparseStatus_t cusp_stat;
    // cublasStatus_t cubl_stat;

    FILE *fpo = fopen("spmm_test_out.out","a");
    printf("reading file ...\n");
    readMtx<float>(argv[1], row_indices, col_indices, values, A_nrows, A_ncols, nnz);

    A_data = (float *)malloc(nnz*sizeof(A_data[0]));
    A_indptr = (int *)malloc((A_nrows+1)*sizeof(A_indptr[0]));
    A_indices = (int *)malloc(nnz*sizeof(A_indices[0]));
    B = (float *)malloc((max_ncols*A_ncols)*sizeof(B[0]));
    
#ifdef VALIDATE
    C = (float *)malloc((A_nrows*max_ncols)*sizeof(C[0]));
    golden = (float *)malloc((A_nrows*max_ncols)*sizeof(golden[0]));
    if (!C || !golden) {
        CLEANUP("Host malloc failed\n");
        return 1;
    }
#endif
    if ( !A_data || !A_indices || !A_indptr || !B ) {
        CLEANUP("Host malloc failed\n");
        return 1;
    }

    /* format conversation COO -> CSR */
    for (int i=0; i<A_nrows+1; i++) {
        A_indptr[i] = 0;
    }
    for (int n=0; n<nnz; n++) {
        int row = row_indices[n];
        if (row>=A_nrows) fprintf(stderr, "out of bound row\n");
        A_indptr[row+1]++;
    }
    for (int n=1; n<A_nrows+1; n++) {
        A_indptr[n] += A_indptr[n-1];
    }
    for (int n=0; n<nnz; n++) {
        int ptr = A_indptr[row_indices[n]];
        if (col_indices[n]>A_ncols) fprintf(stderr, "out of bound column\n");
        A_indices[ptr] = col_indices[n];
        // A_data[ptr] = values[n];
        A_data[ptr] = 1;
        ptr++;
        A_indptr[row_indices[n]]=ptr;
    }
    for (int n=A_nrows-1; n>0; n--) {
        A_indptr[n] = A_indptr[n-1];
    }
    A_indptr[0] = 0; // COO->CSR finish
    
    printf("read file ok. N=%d nnz=%d\n", A_nrows, nnz);

    /* random assign */
    unsigned seed;
    seed = time(0);
    srand(seed);
    // for (int i=0; i<nnz; i++) {
    //     A_data[i] = float(rand() %10000 - 5000)/10000;
    // }
    for (int i=0; i<max_ncols*A_ncols; i++) {
        B[i] = float(rand() %100 - 50)/100;
    }
#ifdef VALIDATE
    for (int i=0; i<A_nrows; i++) {
        for (int k=0; k<max_ncols; k++) {
            float acc = 0.0;
            for (int ptr=A_indptr[i]; ptr<A_indptr[i+1]; ptr++) {
                acc += A_data[ptr]*B[(max_ncols*A_indices[ptr]+k)];
            }
            golden[(max_ncols*i+k)] = acc;
        }
    }
#endif


    // allocate device memory
    cudaDeviceReset();
    cudaSetDevice(DEV_ID);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties( &deviceProp, DEV_ID );
    int max_threads_per_block = deviceProp.sharedMemPerBlock/(sizeof(int)+sizeof(float));
    if (max_threads_per_block > deviceProp.maxThreadsPerBlock) max_threads_per_block = deviceProp.maxThreadsPerBlock;
    // int max_threads_per_block = 1024;

    cudaError_t cudaStat1, cudaStat2, cudaStat3, cudaStat4, cudaStat5, cudaStat6;
    while (true) {
        cudaStat1 = cudaMalloc((void**)&A_indptr_dev, (A_nrows+1)*sizeof(A_indptr_dev[0]));
        cudaStat2 = cudaMalloc((void**)&A_indices_dev, nnz*sizeof(A_indices_dev[0]));
        cudaStat3 = cudaMalloc((void**)&A_data_dev, nnz*sizeof(A_data_dev[0]));
        cudaStat4 = cudaMalloc((void**)&B_dev, max_ncols*A_ncols*sizeof(B_dev[0]));
        cudaStat5 = cudaMalloc((void**)&C_dev, A_nrows*max_ncols*sizeof(C_dev[0]));
        // cudaStat6 = cudaMalloc((void**)&C_tran_dev, A_nrows*max_ncols*sizeof(C_tran_dev[0]));
        if ((cudaStat1 == cudaSuccess) && (cudaStat2 == cudaSuccess) && (cudaStat3 == cudaSuccess) &&
        (cudaStat4 == cudaSuccess) && (cudaStat5 == cudaSuccess)) {
        //  && (cudaStat6 == cudaSuccess)) {
            break;
        }
        cudaDeviceReset();
        cudaSetDevice(DEV_ID);
        max_ncols /= 2;
    }
    printf("max_ncols = %d\n", max_ncols);
    
    checkCudaError(cudaMemcpy(A_indptr_dev, A_indptr, (A_nrows+1)*sizeof(A_indptr_dev[0]), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(A_indices_dev, A_indices, nnz*sizeof(A_indices_dev[0]), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(A_data_dev, A_data, nnz*sizeof(A_data_dev[0]), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(B_dev, B, max_ncols*A_ncols*sizeof(B_dev[0]), cudaMemcpyHostToDevice));
    
    cudaError_t cudaStat = cudaGetLastError();
    // device warm up
    warmup<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaStat = cudaGetLastError();
    if (cudaStat != cudaSuccess) 
    {
        fprintf(stderr, "Warm-up failed: %s\t", cudaGetErrorString(cudaStat));
    }

    // init cusparse params
    // cublasCreate(&cubl_handle);
    // cublasSetPointerMode(cubl_handle, CUBLAS_POINTER_MODE_HOST);
    cusparseCreate(&cusp_handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    cusp_stat = cusparseScsrmm2(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, A_nrows, max_ncols, A_ncols, nnz, &one, descr, A_data_dev, A_indptr_dev, A_indices_dev, B_dev, max_ncols, &zero, C_dev, A_nrows);
    if (cusp_stat != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("csrmm2 failed");
        return 1;
    }
    // cubl_stat = cublasSgeam(cubl_handle, CUBLAS_OP_T, CUBLAS_OP_N, max_ncols, A_nrows, &one, C_dev, A_nrows, &zero, nullptr, max_ncols, C_tran_dev, max_ncols);
    // if (cubl_stat != CUBLAS_STATUS_SUCCESS) {
    //     CLEANUP("Sgeam failed");
    //     return 1;
    // }

#ifdef VALIDATE
    B_ncols=max_ncols;
    checkCudaError(cudaMemcpy(C, C_dev, A_nrows*B_ncols*sizeof(C[0]), cudaMemcpyDeviceToHost));
    for (int i=0; i<A_nrows; i++) 
        for (int j=0; j<B_ncols; j++) 
            if ( fabs((C[(i+j*A_nrows)] - golden[(i*B_ncols+j)])) > 1e-2 ) {
                std::cout << "csrmm2 WA: C[" << i << ", " << j << "] = " << C[(i+j*A_nrows)] <<", golden = " << golden[(i*B_ncols+j)] << '\n';
                break;
            }
    // checkCudaError(cudaMemcpy(C, C_tran_dev, A_nrows*B_ncols*sizeof(C[0]), cudaMemcpyDeviceToHost));
    // for (int i=0; i<A_nrows; i++) 
    //     for (int j=0; j<B_ncols; j++) 
    //         if ( fabs((C[(i*B_ncols+j)] - golden[(i*B_ncols+j)])) > 1e-2 ) {
    //             std::cout << "csrmm2sgeam WA: C[" << i << ", " << j << "] = " << C[(i*B_ncols+j)] <<", golden = " << golden[(i*B_ncols+j)] << '\n';
    //             break;
    //         }
    
    for (int method=0; method<5; method++) {
        checkCudaError(cudaMemset((void*)C_dev, 0, A_nrows*B_ncols*sizeof(C_dev[0])));
        spmmWrapper(method, 4,  A_nrows, B_ncols, A_indptr_dev, A_indices_dev, A_data_dev, B_dev, C_dev);
        checkCudaError(cudaMemcpy(C, C_dev, A_nrows*B_ncols*sizeof(C[0]), cudaMemcpyDeviceToHost));
        for (int i=0; i<A_nrows; i++) 
            for (int j=0; j<B_ncols; j++) 
                if ( fabs((C[(i*B_ncols+j)] - golden[(i*B_ncols+j)])) > 1e-2 ) {
                    std::cout << "kernel" << method << " WA: C[" << i << ", " << j << "] = " << C[(i*B_ncols+j)] <<", golden = " << golden[(i*B_ncols+j)] << '\n';
                    break;
                }
    }

    // for (int method=0; method<3; method++) {
    //     checkCudaError(cudaMemset((void*)C_dev, 0, A_nrows*B_ncols*sizeof(C_dev[0])));
    //     spmmWrapper(method, 1, A_nrows, B_ncols, A_indptr_dev, A_indices_dev, A_data_dev, B_dev, C_dev);
    //     checkCudaError(cudaMemcpy(C, C_dev, A_nrows*B_ncols*sizeof(C[0]), cudaMemcpyDeviceToHost));
    //     for (int i=0; i<A_nrows; i++) 
    //         for (int j=0; j<B_ncols; j++) 
    //             if ( fabs((C[(i*B_ncols+j)] - golden[(i*B_ncols+j)])) > 1e-2 ) {
    //                 std::cout << "kernel" << method << " WA: C[" << i << ", " << j << "] = " << C[(i*B_ncols+j)] <<", golden = " << golden[(i*B_ncols+j)] << '\n';
    //                 break;
    //             }
    // }
    
#endif // VALIDATE

#define ITER 200

    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));
    
    for (int i=0; i<ITER; i++) {
        warmup<<<1,1>>>();
    }
    printf("running tests...\n");

    int tile_row = 8;
    
    for (B_ncols=128; B_ncols<=max_ncols; B_ncols *= 2) {

        gflop = (double)nnz*2/1000000*B_ncols;

        cudaEventRecord(start, 0);
        for (int i=0; i<ITER; i++) {
            cusp_stat = cusparseScsrmm2(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, A_nrows, B_ncols, A_ncols, nnz, &one, descr, A_data_dev, A_indptr_dev, A_indices_dev, B_dev, B_ncols, &zero, C_dev, A_nrows);
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&rt, start, stop);
        // fprintf(fpo, "%f,%f,", rt/ITER, gflop/(rt/ITER));
        fprintf(fpo, "%f,", gflop/(rt/ITER));
        // cudaEventRecord(start, 0);
        // for (int i=0; i<ITER; i++) {
        //     cubl_stat = cublasSgeam(cubl_handle, CUBLAS_OP_T, CUBLAS_OP_N, B_ncols, A_nrows, &one, C_dev, A_nrows, &zero, nullptr, B_ncols, C_tran_dev, B_ncols);
        // }
        // cudaEventRecord(stop, 0);
        // cudaEventSynchronize(stop);
        // cudaEventElapsedTime(&rt2, start, stop);
        // rt += rt2;
        // // fprintf(fpo, "%f,%f,", rt/ITER, gflop/(rt/ITER));
        // fprintf(fpo, "%f,", gflop/(rt/ITER));

        // if (B_ncols<128) tile_row = 2;
        // else 
        tile_row = 8;
        // int method = 2;
        cudaEventRecord(start, 0);
        for (int i=0; i<ITER; i++) {
            spmmWrapper(2, tile_row, A_nrows, B_ncols, A_indptr_dev, A_indices_dev, A_data_dev, B_dev, C_dev);
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&rt, start, stop);
        // fprintf(fpo, "%f,%f,", rt/ITER, gflop/(rt/ITER));
        fprintf(fpo, "%f,", gflop/(rt/ITER));

        // if (B_ncols>=256) {
        //     // GE-SPMM (cf=2)
        //     // for (int method=0; method<5; method++) {
        //         cudaEventRecord(start, 0);
        //         for (int i=0; i<ITER; i++) {
        //             spmmWrapper(method, tile_row, A_nrows, B_ncols, A_indptr_dev, A_indices_dev, A_data_dev, B_dev, C_dev);
        //         }
        //         cudaEventRecord(stop, 0);
        //         cudaEventSynchronize(stop);
        //         cudaEventElapsedTime(&rt, start, stop);
        //         // fprintf(fpo, "%f,%f,", rt/ITER, gflop/(rt/ITER));
        //         fprintf(fpo, "%f,", gflop/(rt/ITER));
        //     // }
        // }
        // else if (B_ncols==128) {
        //     // GE-SPMM (cf=2)
        //     // for (int method=0; method<4; method++) {
        //         cudaEventRecord(start, 0);
        //         for (int i=0; i<ITER; i++) {
        //             spmmWrapper(method, tile_row, A_nrows, B_ncols, A_indptr_dev, A_indices_dev, A_data_dev, B_dev, C_dev);
        //         }
        //         cudaEventRecord(stop, 0);
        //         cudaEventSynchronize(stop);
        //         cudaEventElapsedTime(&rt, start, stop);
        //         // fprintf(fpo, "%f,%f,", rt/ITER, gflop/(rt/ITER));
        
        //         fprintf(fpo, "%f,", gflop/(rt/ITER));
        //     // }
        // }
        // else if (B_ncols==64) {
        //     for (int method=0; method<3; method++) {
        //         cudaEventRecord(start, 0);
        //         for (int i=0; i<ITER; i++) {
        //             spmmWrapper(method, tile_row, A_nrows, B_ncols, A_indptr_dev, A_indices_dev, A_data_dev, B_dev, C_dev);
        //         }
        //         cudaEventRecord(stop, 0);
        //         cudaEventSynchronize(stop);
        //         cudaEventElapsedTime(&rt, start, stop);
        //         // fprintf(fpo, "%f,%f,", rt/ITER, gflop/(rt/ITER));
        
        //         fprintf(fpo, "%f,", gflop/(rt/ITER));
        //     }
        // }
        // else {
        //     for (int method=0; method<2; method++) {
        //         cudaEventRecord(start, 0);
        //         for (int i=0; i<ITER; i++) {
        //             spmmWrapper(method, tile_row, A_nrows, B_ncols, A_indptr_dev, A_indices_dev, A_data_dev, B_dev, C_dev);
        //         }
        //         cudaEventRecord(stop, 0);
        //         cudaEventSynchronize(stop);
        //         cudaEventElapsedTime(&rt, start, stop);
        //         // fprintf(fpo, "%f,%f,", rt/ITER, gflop/(rt/ITER));
        
        //         fprintf(fpo, "%f,", gflop/(rt/ITER));
        //     }
        // }
    }
    
    CLEANUP("");

    return 0;
}
