# GE-SpMM: General-purposed Sparse Matrix-Matrix Multiplication on GPUs for Graph Neural Networks

Guyue Huang\*, Guohao Dai, Yu Wang and Huazhong Yang

Tsinghua University

\*hguyue1@outlook.com

Accepeted to SC2020.

## Abstract
**GE-SpMM** is a fast CSR-based CUDA kernel of sparse-dense matrix multiplication (SpMM), designed to accelerate GNN applications.

## Get started
```
git clone --recursive https://github.com/hgyhungry/ge-spmm.git
```

## Kernel performance

### Prerequisites
CUDA toolkit 10.1

### Compilation
```
source compile.sh
```
The script should also build the baseline implementation in ./merge-spmm. 

### Download dataset
```
cd data
source download_SNAP.sh
```

### run tests
```
source run_test.sh
```

## GunRock baseline
When cloning this repo, pass --recursive flag to automatically pull GunRock submodule.
```
cd gunrock-test
cp -r app/spmm ./gunrock/gunrock/app/
cp -r examples/spmm ./gunrock/examples
cp CMakeList.txt ./gunrock/examples

mkdir build && cd build
cmake .. && make spmm -j8
```

Run tests
``` 
cd $(this-repo)/gunrock-test/gunrock/
cp examples/spmm/test.sh .
source test.sh
```
Results are written to gr_test.txt

## DGL integration

**Prerequisites** CUDA toolkit 10.1  PyTorch 1.4

GE-SpMM can be integrated to [DGL](dgl.ai). When cloning this repo, pass --recursive flag to automatically pull DGL repo. First build DGL from source. Instructions are also in [this tutorial](https://docs.dgl.ai/install/index.html#install-from-source).
```
cd $(this-repo)/dgl-custom/dgl
mkdir build
cd build
cmake -DUSE_CUDA=ON ..
make -j8
cd ../python
python setup.py install --user
```
Run example code.
```
cd $(this-repo)/dgl-custom/benchmark
cd gcn
python gcn_dgl.py --gpu=0 --dataset=pubmed --n-hidden=128 --n-layers=1 
cd ../sage
python sage_dgl.py --gpu=0 --dataset=pubmed --n-hidden=32 --n-layers=2 --aggregator-type=pool
```

Integrate DGL with GE-SpMM
```
cd $(this-repo)/dgl-custom/
cp *.cu ./dgl/src/kernel/cuda/
# rebuild dgl
cd build
make -j8
cd ../python
python setup.py install --user
```
Then you can run the same tests again to see differences of pytorch profiling report.

## PyTorch extension

**Prerequisites** CUDA toolkit 10.1  PyTorch 1.4

We also wrap GE-SpMM to be a pytorch custom op. The operator is compiled in a JIT way and can be called in python code. We use this to substitute MessagePassing propogate step provided in [pyg](https://github.com/rusty1s/pytorch_geometric) and test performance gain.
### Build PyG baseline
```
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
cd $(this-repo)/pytorch-custom/pytorch_geometric
python setup.py install --user
```
### Run tests
```
cd $(this-repo)/pytorch-custom

# first time running gcn_custom the cuda source will be compiled to lib file
# next time the compilation is not repeated and pytorch can directly load the built lib
python gcn_custom.py --n-hidden=32

python gcn_pyg.py --n-hidden=32
python gcn_custom_2layers.py --n-hidden=32
python gcn_pyg_2layers.py --n-hidden=32
```
