import torch
import scipy
import numpy as np
from scipy.io import mmread
from torch.utils.cpp_extension import load
from torch.nn import Parameter, init
import torch.nn.functional as F

sparsePath = '/home/yzm18/myMatrix/test.mtx'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sparsecsr = mmread(sparsePath).tocsr().astype('float32')
sparsecsc = sparsecsr.tocsc()

print(sparsecsr.shape)
filename = 'spmm'

spmm = load(name=filename, sources = [filename + '_kernel.cpp', filename + '.cu'], verbose=True)
rowptr = torch.from_numpy(sparsecsr.indptr).to(device).int()
colind = torch.from_numpy(sparsecsr.indices).to(device).int()
csr_data = torch.from_numpy(sparsecsr.data).to(device).float()

colptr = torch.from_numpy(sparsecsc.indptr).to(device).int()
rowind = torch.from_numpy(sparsecsc.indices).to(device).int()

cscval = spmm.csr2csc(rowptr, colind, colptr, rowind, csr_data)

print('cscval:')
print(cscval)

