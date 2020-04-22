device=0
rm spmm_test_out.out grb_test_out.out
echo "data,K=128-cusparse-gflops,K=128-gespmm-gflops,K=256-cusparse-gflops,K=256-gespmm-gflops,K=512-cusparse-gflops,K=512-gespmm-gflops," >> spmm_test_out.out

for i in ./data/snap/*/
do
    ii=$(basename $i)
    echo -n "$ii," >> spmm_test_out.out
    ./spmm_test ./data/snap/${ii}/${ii}.mtx
    echo >> spmm_test_out.out

    merge-spmm/bin/gbspmm --max_ncols=512 --device=$device ./data/snap/${ii}/${ii}.mtx &>> grb_test_out.out
    merge-spmm/bin/gbspmm --max_ncols=256 --device=$device ./data/snap/${ii}/${ii}.mtx &>> grb_test_out.out
    merge-spmm/bin/gbspmm --max_ncols=128 --device=$device ./data/snap/${ii}/${ii}.mtx &>> grb_test_out.out
    merge-spmm/bin/gbspmm --max_ncols=64 --device=$device ./data/snap/${ii}/${ii}.mtx &>> grb_test_out.out
    merge-spmm/bin/gbspmm --max_ncols=32 --device=$device ./data/snap/${ii}/${ii}.mtx &>> grb_test_out.out
done

for i in ./data/misc/*.mtx
do
    echo -n "$ii," >> spmm_test_out.out
    ./spmm_test $i $device 
    echo >> spmm_test_out.out
    merge-spmm/bin/gbspmm --max_ncols=512 --device=$device $i &>> grb_test_out.out
    merge-spmm/bin/gbspmm --max_ncols=256 --device=$device $i &>> grb_test_out.out
    merge-spmm/bin/gbspmm --max_ncols=128 --device=$device $i &>> grb_test_out.out
    merge-spmm/bin/gbspmm --max_ncols=64 --device=$device $i &>> grb_test_out.out
    merge-spmm/bin/gbspmm --max_ncols=32 --device=$device $i &>> grb_test_out.out
done
