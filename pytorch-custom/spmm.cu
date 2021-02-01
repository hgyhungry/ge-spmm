#include <cuda.h>
#include <torch/types.h>


__global__ void spmm_test0(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, float* A_csrVal,
    float* B_dnVal, float* C_dnVal
)
{
    int rid = blockDim.y*blockIdx.x+threadIdx.y;
    if (rid<A_nrows) {
    int cid = (blockIdx.y<<5)+threadIdx.x;
    int lb = A_csrRowPtr[rid];
    int hb = A_csrRowPtr[(rid+1)];
    int offset = 0;
    float acc=0;
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

__global__ void spmm_test1(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, float* A_csrVal,
    float* B_dnVal, float* C_dnVal
)
{
    extern __shared__ int sh[];
    int *colInd_sh = sh;
    float *val_sh = (float *)&sh[(blockDim.y<<5)];
    int shmem_offset = (threadIdx.y<<5);
    int thread_idx = shmem_offset+threadIdx.x;

    int rid = blockDim.y*blockIdx.x+threadIdx.y;
    
    if (rid<A_nrows) {
        int cid = (blockIdx.y<<5)+threadIdx.x;
        int lb = A_csrRowPtr[rid];
        int hb = A_csrRowPtr[(rid+1)];
        int ptr = lb+threadIdx.x;
        int offset;
        float acc=0;

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

__global__ void spmm_test2(
    int A_nrows, int B_ncols,
    int* A_csrRowPtr, int* A_csrColInd, float* A_csrVal,
    float* B_dnVal, float* C_dnVal
)
{
    extern __shared__ int sh[];
    int *colInd_sh = sh;
    float *val_sh = (float *)&sh[(blockDim.y<<5)];
    int shmem_offset = (threadIdx.y<<5);
    int thread_idx = shmem_offset+threadIdx.x;

    int rid = blockDim.y*blockIdx.x+threadIdx.y;
    
    if (rid<A_nrows) {
        int cid = (blockIdx.y<<6)+threadIdx.x;
        int lb = A_csrRowPtr[rid];
        int hb = A_csrRowPtr[(rid+1)];
        int ptr = lb+threadIdx.x;
        int offset;
        float acc1=0, acc2=0, val;

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

torch::Tensor spmm_cuda(
    torch::Tensor rowptr,
    torch::Tensor colind,
    torch::Tensor edge_weight_csr,
    torch::Tensor grad_out
) {
    const auto m = rowptr.size(0)-1;
    const auto k = grad_out.size(1);
    auto devid = grad_out.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto out = torch::empty({m,k}, options);
    
    if (k<32) {
        const int row_per_block = 128/k;
        const int n_block = (m+row_per_block-1)/row_per_block;
        spmm_test0<<<dim3(n_block,1,1),dim3(k, row_per_block, 1)>>>(
            m, k, rowptr.data<int>(), colind.data<int>(), edge_weight_csr.data<float>(), grad_out.data<float>(), out.data<float>());
        return out;
    }
    if (k<64) {
        const int tile_k = (k+31)/32;
        const int n_block = (m+4-1)/4;
        spmm_test1<<<dim3(n_block, tile_k, 1), dim3(32, 4, 1), 32*4*(sizeof(int)+sizeof(float))>>> (
            m, k, rowptr.data<int>(), colind.data<int>(), edge_weight_csr.data<float>(), grad_out.data<float>(), out.data<float>());
        return out;
    }
    else {
        const int tile_k = (k+63)/64;
        const int n_block = (m+8-1)/8;
        spmm_test2<<<dim3(n_block, tile_k, 1), dim3(32, 8, 1), 32*8*(sizeof(int)+sizeof(float))>>> (
            m, k, rowptr.data<int>(), colind.data<int>(), edge_weight_csr.data<float>(), grad_out.data<float>(), out.data<float>());        
        return out;
    }
}
