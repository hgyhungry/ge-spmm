#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>

torch::Tensor spmm_cuda(
    torch::Tensor rowptr,
    torch::Tensor colind,
    torch::Tensor edge_weight_csr,
    torch::Tensor grad_out
);

// #define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a cuda tensor")
// #define CHECK_CONIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONIGUOUS(x)

torch::Tensor csr_spmm(
    torch::Tensor A_rowptr,
    torch::Tensor A_colind,
    torch::Tensor A_csrVal,
    torch::Tensor B
) {
    assert(A_rowptr.device().type()==torch::kCUDA);
    assert(A_colind.device().type()==torch::kCUDA);
    assert(A_csrVal.device().type()==torch::kCUDA);
    assert(B.device().type()==torch::kCUDA);
    assert(A_rowptr.is_contiguous());
    assert(A_colind.is_contiguous());
    assert(A_csrVal.is_contiguous());
    assert(B.is_contiguous());
    assert(A_rowptr.dtype()==torch::kInt32);
    assert(A_colind.dtype()==torch::kInt32);
    assert(A_csrVal.dtype()==torch::kFloat32);
    assert(B.dtype()==torch::kFloat32);
    return spmm_cuda(A_rowptr, A_colind, A_csrVal, B);
}

PYBIND11_MODULE(spmm, m) {
    m.def("csr_spmm", &csr_spmm, "CSR SPMM");
}