import torch
from torch.utils.cpp_extension import load
from torch.nn import Parameter, init
import torch.nn.functional as F 

spmm = load(name='spmm', sources=['spmm.cpp', 'spmm_kernel.cu'], verbose=True)

class SPMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rowptr, colind, colptr, rowind, feat):
        out = spmm.csr_spmm(rowptr, colind, feat)

        ctx.backward_csc = (colptr, rowind)
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        colptr, rowind = ctx.backward_csc
        grad_feat = spmm.csr_spmm(colptr, rowind, grad_out)
        return None, None, None, None, grad_feat


# import numpy as np 
# import scipy.sparse as scpsp

# def proc():
#     adj = scpsp.random(1000,1000,density=0.01, dtype=np.float32)
#     adj = adj.tocsr()
#     rowptr = torch.tensor(adj.indptr)
#     colind = torch.tensor(adj.indices)
#     adj = adj.tocsc()
#     colptr = torch.tensor(adj.indptr)
#     rowind = torch.tensor(adj.indices)
#     g = {}
#     g['rowptr'] = rowptr
#     g['colind'] = colind
#     g['colptr'] = colptr
#     g['rowind'] = rowind
#     return g
# g = proc()

# class TestGCN(torch.nn.Module):
#     def __init__(self, nh):
#         super(TestGCN, self).__init__()
#         self.n_hidden = nh
#         self.weight = Parameter(torch.Tensor(nh, nh))
#         self.bias = Parameter(torch.Tensor(nh))
#         self.reset_parameters()
#     def reset_parameters(self):
#         init.xavier_uniform_(self.weight)
#         init.zeros_(self.bias)
    
#     def forward(self, x):
#         x = torch.matmul(x, self.weight)
#         x = SPMMFunction.apply(g['rowptr'], g['colind'], g['colptr'], g['rowind'], x)
#         x = x + self.bias
#         return F.log_softmax(x, dim=1)

from torch_geometric.nn.inits import glorot, zeros  

class GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, normalize=True, **kwargs):
        super(GCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None
    
    @staticmethod
    def in_deg_sqrt(indptr):
        return (1 / torch.sqrt((indptr[1:]-indptr[:-1]).float())).unsqueeze(dim=1)
    
    @staticmethod
    def out_deg_sqrt(indptr):
        return (1 / torch.sqrt((indptr[1:]-indptr[:-1]).float())).unsqueeze(dim=1)

    def forward(self, x, rowptr, colind, colptr, rowind):
        """"""
        x = torch.matmul(x, self.weight)
        # if self.cached and self.cached_result is not None:
        #     if edge_index.size(1) != self.cached_num_edges:
        #         raise RuntimeError(
        #             'Cached {} number of edges, but found {}. Please '
        #             'disable the caching behavior of this layer by removing '
        #             'the `cached=True` argument in its constructor.'.format(
        #                 self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            # self.cached_num_edges = edge_index.size(1)
            # self.cached_num_edges = colind.size(0)
            if self.normalize:
                # edge_index, norm = self.norm(edge_index, x.size(
                    # self.node_dim), edge_weight, self.improved, x.dtype)
                in_deg_norm = self.in_deg_sqrt(rowptr)
                out_deg_norm = self.out_deg_sqrt(colptr)

            else:
                # norm = edge_weight
                in_deg_norm = torch.ones(rowptr.shape(0)-1, dtype=x.dtype, device=x.device)
                out_deg_norm = torch.ones(colptr.shape(0)-1, dtype=x.dtype, device=x.device)
                
            # self.cached_result = edge_index, norm
            self.cached_result = in_deg_norm, out_deg_norm

        # edge_index, norm = self.cached_result
        in_deg_norm, out_deg_norm = self.cached_result
        if self.normalize:
            x = x*out_deg_norm
        aggr_out = SPMMFunction.apply(rowptr, colind, colptr, rowind, x)
        if self.normalize:
            aggr_out = aggr_out*in_deg_norm
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
    
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
