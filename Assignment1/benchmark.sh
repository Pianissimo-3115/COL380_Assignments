#!/usr/bin/env bash

THREADS="1 4 8 12 16"
FREQ=1000000
SIZE=100000000
OUT=speedup.csv

make clean && make || exit 1

echo "threads,updateDisplay_speedup,printOrderStats_speedup,totalAmountTraded_speedup" > $OUT

for t in $THREADS; do
    export OMP_NUM_THREADS=$t
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores

    echo "Running with $t threads..."

    output=$(./testgen $FREQ $SIZE)

    seq_upd=$(echo "$output" | grep "Time taken by sequential updateDisplay" | awk '{print $(NF-1)}')
    par_upd=$(echo "$output" | grep "Time taken by parallel updateDisplay"   | awk '{print $(NF-1)}')

    seq_stats=$(echo "$output" | grep "Time taken by sequential printOrderStats" | awk '{print $(NF-1)}')
    par_stats=$(echo "$output" | grep "Time taken by parallel printOrderStats"   | awk '{print $(NF-1)}')

    seq_total=$(echo "$output" | grep "Time taken by sequential totalAmountTraded" | awk '{print $(NF-1)}')
    par_total=$(echo "$output" | grep "Time taken by parallel totalAmountTraded"   | awk '{print $(NF-1)}')

    upd_speedup=$(awk "BEGIN {print $seq_upd / $par_upd}")
    stats_speedup=$(awk "BEGIN {print $seq_stats / $par_stats}")
    total_speedup=$(awk "BEGIN {print $seq_total / $par_total}")

    echo "$t,$upd_speedup,$stats_speedup,$total_speedup" >> $OUT
done
