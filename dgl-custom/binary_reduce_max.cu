/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/binary_reduce_max.cu
 * \brief CUDA kernels for binary reduce max
 */
#include <dgl/runtime/device_api.h>


#include "./binary_reduce_impl.cuh"
#include "./backward_binary_reduce_impl.cuh"
#include "../csr_interface.h"

using minigun::advance::RuntimeConfig;

namespace dgl {
namespace kernel {
namespace cuda {
  __device__ __forceinline__ float max_reduce(float acc, float x) {
    return acc>x? acc:x;
  }
  
  __device__ __forceinline__ float max_init() {
    return -10000; //TODO: a better minus-inf
  }
  
  __global__ void topoCacheCoarsenSPMMMaxKernel(
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
    float acc1 = max_init();
    float acc2 = max_init();
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
          acc1 = max_reduce(acc1, B[offset]);
          acc2 = max_reduce(acc2, B[(offset+32)]);
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
          acc1 = max_reduce(acc1, B[offset]);}
          // acc1 = sum_reduce(acc1, __ldg(B+offset)); }
          if (nout>1) {
          acc2 = max_reduce(acc2, B[(offset+32)]);}
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
  
  __global__ void topoCacheSPMMMaxKernel(
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
    float acc1 = max_init();
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
          acc1 = max_reduce(acc1, B[offset]);
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
          acc1 = max_reduce(acc1, B[offset]);}
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
  
__global__ void topoSimpleSPMMMaxKernel(
  int m, int k, const int* A_indptr, const int* A_indices, const float* B, float* C 
) {
  int rid = blockDim.y*blockIdx.x+threadIdx.y;
  if (rid<m) {
    int lb = A_indptr[rid];
    int hb = A_indptr[(rid+1)];
    float acc1 = max_init();
    int offset;
    for (int ptr=lb; ptr<hb; ptr++) {
      // offset = __ldg(A_indices+ptr)*k+threadIdx.x;
      // acc1 = sum_reduce(acc1, __ldg(B+offset));
      offset = A_indices[ptr]*k+threadIdx.x;
      acc1 = max_reduce(acc1, B[offset]);
    }
    C[(rid*k+threadIdx.x)] = acc1;
  }
}
  
  
template <typename DType>
int XTopoCsrmmmax(const RuntimeConfig& rtcfg,
  int m, int n,
  const int* A_indptr,
  const int* A_indices,
  const DType* B, DType* C) {
LOG(INFO) << "Not supported by custom spmm";
return -1;
}

template <>
int XTopoCsrmmmax<float>(const RuntimeConfig& rtcfg,
  int m, int n,
  const int* A_indptr,
  const int* A_indices,
  const float* B, float* C) {

  // LOG(INFO) << "Using custom spmm";
  if (n<32) {
    const int row_per_block = 128/n;
    const int n_block = (m+row_per_block-1)/row_per_block;
    topoSimpleSPMMMaxKernel<<< dim3(n_block,1,1),dim3(n, row_per_block, 1), 0,rtcfg.stream>>>(m,n,A_indptr,A_indices,B,C);
    return 0;
  }
  if (n<64) {
    const int tile_k = (n+31)/32;
    const int n_block = (m+3)/4;
    topoCacheSPMMMaxKernel<<< dim3(n_block,tile_k,1), dim3(32,4,1), 128*sizeof(int), rtcfg.stream>>>(m,n,A_indptr,A_indices,B,C);
    return 0;
  }
  else {
    const int tile_k = (n+63)/64;
    const int n_block = (m+8-1)/8;
    topoCacheCoarsenSPMMMaxKernel<<< dim3(n_block,tile_k,1), dim3(32,8,1), 8*32*sizeof(int), rtcfg.stream>>>(m,n,A_indptr,A_indices,B,C);
    return 0;
  }
}

template <typename DType>
void CustomCsrmmmax(
  const RuntimeConfig& rtcfg,
  const aten::CSRMatrix& csr,
  const DType* B_data, DType* C_data,
  int x_length) {

  const int m = csr.num_rows;
  const int n = x_length;
  typedef int32_t Idx;

  int ret = XTopoCsrmmmax<DType> ( rtcfg,
    m, n, 
    static_cast<Idx*>(csr.indptr->data),
    static_cast<Idx*>(csr.indices->data),
    B_data, C_data
  );

  cudaStreamSynchronize(rtcfg.stream);
  CUDA_CALL(cudaGetLastError());
}

template <typename DType>
void FallbackCallBinaryReduceMax(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    GData<int32_t, DType>* gdata) {
  constexpr int XPU = kDLGPU;
  typedef int32_t Idx;
  typedef SelectSrc LeftSelector;
  typedef SelectNone RightSelector;
  typedef BinaryUseLhs<DType> BinaryOp;
  typedef ReduceMax<kDLGPU, DType> Reducer;
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
} // namespace cuda

template <>
void CallBinaryReduce<kDLGPU, int32_t, float, SelectSrc, SelectNone,
                      BinaryUseLhs<float>, ReduceMax<kDLGPU, float>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    GData<int32_t, float>* gdata) {
  if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBinaryReduceMax<float>(rtcfg, graph, gdata);
  } else {
    // cusparse use rev csr for csrmm
    auto csr = graph.GetInCSRMatrix();
    cuda::CustomCsrmmmax(rtcfg, csr, gdata->lhs_data, gdata->out_data,
        gdata->x_length);
  }
}

#define REDUCER ReduceMax
#define XPU kDLGPU
#define IDX int32_t
EVAL(GEN_DTYPE, GEN_OP_TARGET, GEN_DEFINE)
EVAL(GEN_BACKWARD_MODE, GEN_DTYPE, GEN_OP_TARGET, GEN_BACKWARD_DEFINE)

}  // namespace kernel
}  // namespace dgl
