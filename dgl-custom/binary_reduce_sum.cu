/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/binary_reduce_sum.cu
 * \brief CUDA kernels for binary reduce sum
 */
#include <dgl/runtime/device_api.h>

#include "../../runtime/cuda/cuda_common.h"
#include "./binary_reduce_impl.cuh"
#include "./backward_binary_reduce_impl.cuh"
#include "../utils.h"
#include "../csr_interface.h"

using minigun::advance::RuntimeConfig;

namespace dgl {
namespace kernel {
namespace cuda {
// specialization for cusparse

template <typename DType>
cusparseStatus_t Xcsrmm2(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const DType* alpha, const cusparseMatDescr_t descrA,
    const DType* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const DType* B, int ldb, const DType* beta, DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUSPARSE_STATUS_EXECUTION_FAILED;
}

template <>
cusparseStatus_t Xcsrmm2<float>(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const float* alpha, const cusparseMatDescr_t descrA,
    const float* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const float* B, int ldb, const float* beta, float* C, int ldc) {
  return cusparseScsrmm2(handle, transA, transB, m, n, k, nnz,
      alpha, descrA, csrValA, csrRowPtrA, csrColIndA,
      B, ldb, beta, C, ldc);
}

template <>
cusparseStatus_t Xcsrmm2<double>(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const double* alpha, const cusparseMatDescr_t descrA,
    const double* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const double* B, int ldb, const double* beta, double* C, int ldc) {
  return cusparseDcsrmm2(handle, transA, transB, m, n, k, nnz,
      alpha, descrA, csrValA, csrRowPtrA, csrColIndA,
      B, ldb, beta, C, ldc);
}

template <typename DType>
cublasStatus_t Xgeam(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const DType* alpha, const DType* A, int lda,
    const DType* beta, const DType* B, int ldb,
    DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}

template <>
cublasStatus_t Xgeam<float>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const float* alpha, const float* A, int lda,
    const float* beta, const float* B, int ldb,
    float* C, int ldc) {
  return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda,
      beta, B, ldb, C, ldc);
}

template <>
cublasStatus_t Xgeam<double>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const double* alpha, const double* A, int lda,
    const double* beta, const double* B, int ldb,
    double* C, int ldc) {
  return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda,
      beta, B, ldb, C, ldc);
}

template <typename DType>
void CusparseCsrmm2(
    const RuntimeConfig& rtcfg,
    const aten::CSRMatrix& csr,
    const DType* B_data, DType* C_data,
    int x_length) {
  // We use csrmm2 to perform following operation:
  // C = A x B, where A is a sparse matrix in csr format, B is the dense matrix for node
  // feature tensor. However, since cusparse only supports column-major, while our tensor
  // is stored in row-major, the actual computation is:
  // C = trans(A x trans(B)).
  // Currently, we use cublasXgeam to implement transposition and allocate intermediate
  // workspace memory for this.
  const int m = csr.num_rows;
  const int n = x_length;
  const int k = csr.num_cols;
  const int nnz = csr.indices->shape[0];
  const DType alpha = 1.0;
  const DType beta = 0.0;
  // device
  auto device = runtime::DeviceAPI::Get(rtcfg.ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, rtcfg.stream));
  // allocate matrix for temporary transposed output
  DType* trans_out = static_cast<DType*>(device->AllocWorkspace(rtcfg.ctx, m * n * sizeof(DType)));
  // all one data array
  DType* valptr = static_cast<DType*>(device->AllocWorkspace(rtcfg.ctx, nnz * sizeof(DType)));
  utils::Fill<kDLGPU>(rtcfg.ctx, valptr, nnz, static_cast<DType>(1.));
  cusparseMatDescr_t descr;
  CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
  CUSPARSE_CALL(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CALL(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
  CUSPARSE_CALL(Xcsrmm2<DType>(
      thr_entry->cusparse_handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE,
      m, n, k, nnz, &alpha,
      descr, valptr,
      static_cast<int32_t*>(csr.indptr->data),
      static_cast<int32_t*>(csr.indices->data),
      B_data, n, &beta, trans_out, m));
  device->FreeWorkspace(rtcfg.ctx, valptr);
  // transpose the output matrix
  if (!thr_entry->cublas_handle) {
    CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
  }
  CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle, rtcfg.stream));
  CUBLAS_CALL(Xgeam<DType>(
      thr_entry->cublas_handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n, m,
      &alpha, trans_out, m,
      &beta, nullptr, n,
      C_data, n));
  device->FreeWorkspace(rtcfg.ctx, trans_out);
}


__device__ __forceinline__ float sum_reduce(float acc, float x) {
  return acc + x;
}

__device__ __forceinline__ float sum_init() {
  return 0;
}

__global__ void topoCacheCoarsenSPMMKernel(
  int m, int k, const int* A_indptr, const int* A_indices, const float* B, float* C
) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y<<5);
  int thread_idx = sm_offset+threadIdx.x;

  int rid = blockDim.y*blockIdx.x+threadIdx.y;
  if (rid<m) {

    int cid = (blockIdx.y<<6)+threadIdx.x;
    int lb = A_indptr[rid];
    int hb = A_indptr[rid+1];
    int ptr = lb+threadIdx.x;
    int offset;
    float acc1 = sum_init();
    float acc2 = sum_init();
    if (blockIdx.y != gridDim.y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[(sm_offset+kk)] + cid;
          acc1 = sum_reduce(acc1, B[offset]);
          acc2 = sum_reduce(acc2, B[(offset+32)]);
          // acc1 = sum_reduce(acc1, __ldg(B+offset));
          // acc2 = sum_reduce(acc2, __ldg(B+offset+32));
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      C[offset] = acc1;
      C[offset+32] = acc2;
    }
    else { // threadIdx.y==blockDim.y-1
      int nout = (k-cid+31)/32;
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[(sm_offset+kk)] + cid;
          if (nout>0) {
          acc1 = sum_reduce(acc1, B[offset]);}
          // acc1 = sum_reduce(acc1, __ldg(B+offset)); }
          if (nout>1) {
          acc2 = sum_reduce(acc2, B[(offset+32)]);}
          // acc2 = sum_reduce(acc2, __ldg(B+offset+32));}
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      if (nout>0) {
      C[offset] = acc1;}
      if (nout>1) {
      C[offset+32] = acc2;}
    }
  }
} 

__global__ void topoCacheSPMMKernel(
  int m, int k, const int* A_indptr, const int* A_indices, const float* B, float* C 
) {
  extern __shared__ int sh[];
  int sm_offset = (threadIdx.y<<5);
  int thread_idx = sm_offset + threadIdx.x;
  
  int cid = (blockIdx.y<<5)+threadIdx.x;
  int rid = blockDim.y*blockIdx.x+threadIdx.y;
    
  if (rid<m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    int offset;
    int ptr = lb+threadIdx.x;
    float acc1 = sum_init();
    if (blockIdx.y != gridDim.y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[sm_offset+kk]+cid;
          acc1 = sum_reduce(acc1, B[offset]);
          // acc1 = sum_reduce(acc1, __ldg(B+offset));
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      C[offset] = acc1;
    }
    else { // threadIdx.y==blockDim.y-1
      int nout = (k-cid+31)/32;
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          sh[thread_idx] = A_indices[ptr]*k;
          // sh[thread_idx] = __ldg(A_indices+ptr)*k;
        }
        __syncwarp();
        ptr += 32;
        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = sh[(sm_offset+kk)] + cid;
          if (nout>0) {
          acc1 = sum_reduce(acc1, B[offset]);}
          // acc1 = sum_reduce(acc1, __ldg(B+offset)); }
        }
        __syncwarp();
      }
      offset = rid*k+cid;
      if (nout>0) {
      C[offset] = acc1;}
    }
  }
}

__global__ void topoSimpleSPMMKernel(
  int m, int k, const int* A_indptr, const int* A_indices, const float* B, float* C 
) {
  int rid = blockDim.y*blockIdx.x+threadIdx.y;
  if (rid<m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    float acc1 = sum_init();
    int offset;
    for (int ptr=lb; ptr<hb; ptr++) {
      // offset = __ldg(A_indices+ptr)*k+threadIdx.x;
      // acc1 = sum_reduce(acc1, __ldg(B+offset));
      offset = A_indices[ptr]*k+threadIdx.x;
      acc1 = sum_reduce(acc1, B[offset]);
    }
    C[(rid*k+threadIdx.x)] = acc1;
  }
}


template <typename DType>
int XTopoCsrmm(const RuntimeConfig& rtcfg,
  int m, int n,
  const int* A_indptr,
  const int* A_indices,
  const DType* B, DType* C) {
LOG(INFO) << "Not supported by custom spmm";
return -1;
}

template <>
int XTopoCsrmm<float>(const RuntimeConfig& rtcfg,
  int m, int n,
  const int* A_indptr,
  const int* A_indices,
  const float* B, float* C) {

  // LOG(INFO) << "Using custom spmm";
  if (n<32) {
    const int row_per_block = 128/n;
    const int n_block = (m+row_per_block-1)/row_per_block;
    topoSimpleSPMMKernel<<< dim3(n_block,1,1),dim3(n, row_per_block, 1), 0,rtcfg.stream>>>(m,n,A_indptr,A_indices,B,C);
    return 0;
  }
  if (n<64) {
    const int tile_k = (n+31)/32;
    const int n_block = (m+3)/4;
    topoCacheSPMMKernel<<< dim3(n_block,tile_k,1), dim3(32,4,1), 128*sizeof(int), rtcfg.stream>>>(m,n,A_indptr,A_indices,B,C);
    return 0;
  }
  else {
    const int tile_k = (n+63)/64;
    const int n_block = (m+8-1)/8;
    topoCacheCoarsenSPMMKernel<<< dim3(n_block,tile_k,1), dim3(32,8,1), 8*32*sizeof(int), rtcfg.stream>>>(m,n,A_indptr,A_indices,B,C);
    return 0;
  }
}

template <typename DType>
void CustomCsrmm(
  const RuntimeConfig& rtcfg,
  const aten::CSRMatrix& csr,
  const DType* B_data, DType* C_data,
  int x_length) {

  const int m = csr.num_rows;
  const int n = x_length;
  typedef int32_t Idx;

  int ret = XTopoCsrmm<DType> ( rtcfg,
    m, n, 
    static_cast<Idx*>(csr.indptr->data),
    static_cast<Idx*>(csr.indices->data),
    B_data, C_data
  );
  if (ret == -1) {
    CusparseCsrmm2(rtcfg, csr, B_data, C_data, x_length);
  }

  cudaStreamSynchronize(rtcfg.stream);
  CUDA_CALL(cudaGetLastError());
}

// forward

template <typename DType>
void FallbackCallBinaryReduce(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    GData<int32_t, DType>* gdata) {
  constexpr int XPU = kDLGPU;
  typedef int32_t Idx;
  typedef SelectSrc LeftSelector;
  typedef SelectNone RightSelector;
  typedef BinaryUseLhs<DType> BinaryOp;
  typedef ReduceSum<kDLGPU, DType> Reducer;
  typedef cuda::FunctorsTempl<Idx, DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer>
          Functors;
  typedef cuda::BinaryReduce<Idx, DType, Functors> UDF;
  // csr
  auto outcsr = graph.GetOutCSRMatrix();
  minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(outcsr.indptr, outcsr.indices);
  // If the user-given mapping is none and the target is edge data, we need to
  // replace the mapping by the edge ids in the csr graph so that the edge
  // data is correctly read/written.
  if (LeftSelector::target == binary_op::kEdge && gdata->lhs_mapping == nullptr) {
    gdata->lhs_mapping = static_cast<Idx*>(outcsr.data->data);
  }
  if (RightSelector::target == binary_op::kEdge && gdata->rhs_mapping == nullptr) {
    gdata->rhs_mapping = static_cast<Idx*>(outcsr.data->data);
  }
  if (OutSelector<Reducer>::Type::target == binary_op::kEdge
      && gdata->out_mapping == nullptr) {
    gdata->out_mapping = static_cast<Idx*>(outcsr.data->data);
  }
  // TODO(minjie): allocator
  minigun::advance::Advance<XPU, Idx, cuda::AdvanceConfig, GData<Idx, DType>, UDF>(
        rtcfg, csr, gdata, minigun::IntArray1D<Idx>());
}

template <typename DType>
void FallbackCallBackwardBinaryReduce(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, DType>* gdata) {
  constexpr int XPU = kDLGPU;
  constexpr int Mode = binary_op::kGradLhs;
  typedef int32_t Idx;
  typedef SelectSrc LeftSelector;
  typedef SelectNone RightSelector;
  typedef BinaryUseLhs<DType> BinaryOp;
  typedef ReduceSum<kDLGPU, DType> Reducer;
  // For backward computation, we use reverse csr and switch dst and src.
  // This benefits the most common src_op_edge or copy_src case, because the
  // gradients of src are now aggregated into destination buffer to reduce
  // competition of atomic add.
  auto incsr = graph.GetInCSRMatrix();
  minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr.indptr, incsr.indices);
  typedef cuda::BackwardFunctorsTempl<Idx, DType,
          typename SwitchSrcDst<LeftSelector>::Type,
          typename SwitchSrcDst<RightSelector>::Type,
          BinaryOp, Reducer> Functors;
  typedef cuda::BackwardBinaryReduce<Mode, Idx, DType, Functors> UDF;
  // If the user-given mapping is none and the target is edge data, we need to
  // replace the mapping by the edge ids in the csr graph so that the edge
  // data is correctly read/written.
  if (LeftSelector::target == binary_op::kEdge
      && gdata->lhs_mapping == nullptr) {
    gdata->lhs_mapping = static_cast<Idx*>(incsr.data->data);
  }
  if (RightSelector::target == binary_op::kEdge
      && gdata->rhs_mapping == nullptr) {
    gdata->rhs_mapping = static_cast<Idx*>(incsr.data->data);
  }
  if (OutSelector<Reducer>::Type::target == binary_op::kEdge
      && gdata->out_mapping == nullptr) {
    gdata->out_mapping = static_cast<Idx*>(incsr.data->data);
  }
  // TODO(minjie): allocator
  minigun::advance::Advance<XPU, Idx, cuda::AdvanceConfig, BackwardGData<Idx, DType>, UDF>(
        rtcfg, csr, gdata, minigun::IntArray1D<Idx>());
}

}  // namespace cuda

template <>
void CallBinaryReduce<kDLGPU, int32_t, float, SelectSrc, SelectNone,
                      BinaryUseLhs<float>, ReduceSum<kDLGPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    GData<int32_t, float>* gdata) {
  if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBinaryReduce<float>(rtcfg, graph, gdata);
  } else {
    // cusparse use rev csr for csrmm
    auto csr = graph.GetInCSRMatrix();
    // cuda::CusparseCsrmm2(rtcfg, csr, gdata->lhs_data, gdata->out_data,
        // gdata->x_length);
    cuda::CustomCsrmm(rtcfg, csr, gdata->lhs_data, gdata->out_data,
        gdata->x_length);
  }
}

template <>
void CallBinaryReduce<kDLGPU, int32_t, double, SelectSrc, SelectNone,
                      BinaryUseLhs<double>, ReduceSum<kDLGPU, double>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    GData<int32_t, double>* gdata) {
  if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBinaryReduce<double>(rtcfg, graph, gdata);
  } else {
    // cusparse use rev csr for csrmm
    auto csr = graph.GetInCSRMatrix();
    cuda::CusparseCsrmm2(rtcfg, csr, gdata->lhs_data, gdata->out_data,
        gdata->x_length);
  }
}

// backward

template <>
void CallBackwardBinaryReduce<kDLGPU, binary_op::kGradLhs, int32_t, float,
                              SelectSrc, SelectNone,
                              BinaryUseLhs<float>, ReduceSum<kDLGPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, float>* gdata) {
  if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBackwardBinaryReduce<float>(rtcfg, graph, gdata);
  } else {
    auto csr = graph.GetOutCSRMatrix();
    // cuda::CusparseCsrmm2(rtcfg, csr, gdata->grad_out_data, gdata->grad_lhs_data,
        // gdata->x_length);
    cuda::CustomCsrmm(rtcfg, csr, gdata->grad_out_data, gdata->grad_lhs_data,
        gdata->x_length);
  }
}

template <>
void CallBackwardBinaryReduce<kDLGPU, binary_op::kGradLhs, int32_t, double,
                              SelectSrc, SelectNone,
                              BinaryUseLhs<double>, ReduceSum<kDLGPU, double>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, double>* gdata) {
  if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBackwardBinaryReduce<double>(rtcfg, graph, gdata);
  } else {
    auto csr = graph.GetOutCSRMatrix();
    cuda::CusparseCsrmm2(rtcfg, csr, gdata->grad_out_data, gdata->grad_lhs_data,
        gdata->x_length);
  }
}

// generate definitions

#define REDUCER ReduceSum
#define XPU kDLGPU
#define IDX int32_t

EVAL(GEN_DTYPE, GEN_OP_TARGET, GEN_DEFINE);
EVAL(GEN_BACKWARD_MODE, GEN_DTYPE, GEN_OP_TARGET, GEN_BACKWARD_DEFINE);

}  // namespace kernel
}  // namespace dgl
