nvcc -std=c++11 -lcusparse -O3 -o spmm_test spmm_test.cu

# compile baseline
# cp gbspmm.cu merge-spmm/test/
cd merge-spmm
make clean
# rm -rf CMakeFiles
# rm CMakeCache.txt cmake_install.cmake
cd ext/merge-spmv
make gpu_spmv sm=350
cd ../..
cmake .
make gbspmm
cd ..