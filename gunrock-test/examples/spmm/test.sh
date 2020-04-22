for grh in cora citeseer pubmed
do
for k in 32 64 128
do
build/bin/spmm market ./../../data/misc/${grh}.mtx --num-runs=200 --feature-len=$k &>> gr_test.txt
done
done