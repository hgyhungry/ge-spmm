// file: gespmm.hpp
//      Kernel dispatcher.

#include "cuda.h"

struct SpMatCsrDescr_t {
    int nrow;
    int ncol;
    int nnz;
    int *indptr;
    int *indices;
    float *data=nullptr;
};

enum gespmmAlg_t {
    GESPMM_ALG_PARREDUCE_ROWBALANCE,
    GESPMM_ALG_PARREDUCE_NNZBALANCE,
    GESPMM_ALG_SEQREDUCE_ROWBALANCE,
    GESPMM_ALG_SEQREDUCE_NNZBALANCE,
    GESPMM_ALG_ROWCACHING_ROWBALANCE,
    GESPMM_ALG_ROWCACHING_NNZBALANCE,
    GESPMM_ALG_DEFAULT
};

//
// top-level functin
//
void gespmmCsrSpMM( const SpMatCsrDescr_t spmatA,
                    const float *B,
                    const int   N,
                    float       *C,
                    gespmmAlg_t alg
                    );