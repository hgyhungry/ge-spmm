import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
# from torch_geometric.nn import GCNConv, ChebConv  # noqa
from op import GCNConv

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
parser.set_defaults(use_gdc=False)
parser.add_argument('--n-hidden', type=int, default=64,help="number of hidden features")
args = parser.parse_args()


dataset = 'PubMed'
path = osp.join(osp.dirname(osp.realpath('.')), '..', 'data', dataset)
# dataset = Planetoid(path, dataset, T.NormalizeFeatures())
dataset=Planetoid("/home/henrychang/ge-spmm_test/ge-spmm/data/PubMed",dataset,transform=T.NormalizeFeatures())
data = dataset[0]

import scipy.sparse as scpsp
import numpy as np

device = torch.device('cuda' )#if torch.cuda.is_available() else 'cpu')

def proc(add_self_loop=True):
    edge_index = data.edge_index.numpy().astype(np.int32)
    n_v = data.x.shape[0]
    if add_self_loop:
        edge_index = np.concatenate((edge_index, np.array([np.arange(n_v).astype(np.int32)]*2)), axis=1) # self-loop
    n_e = edge_index.shape[1]
    import scipy.sparse as scpsp
    adj = scpsp.coo_matrix((np.ones(n_e), (edge_index[0],edge_index[1])), shape=(n_v,n_v))

    g = {}
    adj = adj.tocsr()
    g['colptr'] = torch.tensor(adj.indptr).to(device)
    g['rowind'] = torch.tensor(adj.indices).to(device)
    g['value_csr']=torch.tensor(adj.data).to(device).float()
    adj = adj.tocsc()
    g['rowptr'] = torch.tensor(adj.indptr).to(device)
    g['colind'] = torch.tensor(adj.indices).to(device)
    g['value_csc']=torch.tensor(adj.data).to(device).float()
    
    
    return g 

g = proc()
 

# if args.use_gdc:
#     gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
#                 normalization_out='col',
#                 diffusion_kwargs=dict(method='ppr', alpha=0.05),
#                 sparsification_kwargs=dict(method='topk', k=128,
#                                            dim=0), exact=True)
#     data = gdc(data)
data = data.to(device)
    
class Net(torch.nn.Module):
    def __init__(self, n_hidden):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, n_hidden, cached=True,
                             normalize=True)
        self.conv2 = GCNConv(n_hidden, dataset.num_classes, cached=True,
                             normalize=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self):
        x, rowptr, colind, colptr, rowind, edge_weight_csr, edge_weight_csc= data.x, g['rowptr'], g['colind'], g['colptr'], g['rowind'], g['value_csr'], g["value_csc"]
        x = F.relu(self.conv1(x, rowptr, colind, colptr, rowind, edge_weight_csr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, rowptr, colind, colptr, rowind, edge_weight_csr)
        return F.log_softmax(x, dim=1)
        

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
#                              normalize=not args.use_gdc)
#         self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
#                              normalize=not args.use_gdc)
#         # self.conv1 = ChebConv(data.num_features, 16, K=2)
#         # self.conv2 = ChebConv(16, data.num_features, K=2)

#         self.reg_params = self.conv1.parameters()
#         self.non_reg_params = self.conv2.parameters()

#     def forward(self):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#         x = F.relu(self.conv1(x, edge_index, edge_weight))
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index, edge_weight)
#         return F.log_softmax(x, dim=1)

n_hidden = args.n_hidden
model = Net(n_hidden).to(device)
# model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
#warmup 
train()

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for epoch in range(1, 201):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))
print(prof.key_averages().table(sort_by="cuda_time_total"))