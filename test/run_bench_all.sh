#!/bin/bash
# Build and benchmark all implementations, output results to CSV.
# Usage: bash test/run_bench_all.sh [NUM_THREADS]
#   NUM_THREADS defaults to 64

set -e

THREADS=${1:-64}
IMPLS="baseline opt1 opt2 opt3 omp1 omp2 omp3"
OUTCSV="test/bench_results.csv"

# --- Implementation benchmark ---
# Create header if file doesn't exist
if [ ! -f "$OUTCSV" ]; then
    echo "impl,n,time,logdet,logdet_lapack" > "$OUTCSV"
fi

for impl in $IMPLS; do
    # Skip if this impl already has results
    if grep -q "^${impl}," "$OUTCSV" 2>/dev/null; then
        echo "=== Skipping $impl (already in $OUTCSV) ==="
        continue
    fi

    echo "=== Building $impl ==="
    make clean -s
    make IMPL=$impl BUILDDIR=build_bench -s 2>/dev/null

    gcc -O3 -march=native -funroll-loops -fopenmp -Iinclude \
        test/bench_all.c -Lbuild_bench -lcholesky_${impl} -lm -llapack -lblas -fopenmp \
        -o build_bench/bench_all

    echo "=== Running $impl (OMP_NUM_THREADS=$THREADS) ==="
    OMP_NUM_THREADS=$THREADS ./build_bench/bench_all "$impl" >> "$OUTCSV"
    rm -rf build_bench
done

echo ""
echo "Results written to $OUTCSV"

# --- Thread scaling benchmark for omp3 ---
SCALING_CSV="test/bench_scaling.csv"
THREAD_COUNTS="1 2 4 8 16 32 64"

# Create header if file doesn't exist
if [ ! -f "$SCALING_CSV" ]; then
    echo "threads,n,time" > "$SCALING_CSV"
fi

# Check which thread counts still need to be run
REMAINING=""
for t in $THREAD_COUNTS; do
    if ! grep -q "^${t}," "$SCALING_CSV" 2>/dev/null; then
        REMAINING="$REMAINING $t"
    fi
done

if [ -z "$REMAINING" ]; then
    echo "=== All thread counts already in $SCALING_CSV, skipping ==="
else
    echo "=== Building omp3 for thread scaling ==="
    make clean -s
    make IMPL=omp3 BUILDDIR=build_bench -s 2>/dev/null
    gcc -O3 -march=native -funroll-loops -fopenmp -Iinclude \
        test/bench_all.c -Lbuild_bench -lcholesky_omp3 -lm -llapack -lblas -fopenmp \
        -o build_bench/bench_all

    for t in $REMAINING; do
        echo "=== omp3 with OMP_NUM_THREADS=$t ==="
        OMP_NUM_THREADS=$t ./build_bench/bench_all "omp3_t${t}" | \
            awk -F',' -v threads="$t" '{print threads","$2","$3}' >> "$SCALING_CSV"
    done

    rm -rf build_bench
fi

echo ""
echo "Scaling results written to $SCALING_CSV"
