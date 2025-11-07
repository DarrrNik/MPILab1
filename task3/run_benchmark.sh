#!/bin/bash

echo "========================================="
echo "  Cannon algorithm benchmark"
echo "========================================="

mkdir -p results
mkdir -p plots

echo "[1/5] Program compiling..."
mpicc -O3 -o cannon main.c -lm

if [ $? -ne 0 ]; then
    echo "Compile error!"
    exit 1
fi

RESULTS_FILE="results/benchmark_results_final.csv"
echo "matrix_size,processes,parallel_time,sequential_time,speedup,efficiency" > $RESULTS_FILE

MATRIX_SIZES=(384 768 1152 1536 1920)
PROCESSES_LIST=(1 4 9)

echo "[2/5] Benchmark running..."

total_tests=$((${#MATRIX_SIZES[@]} * ${#PROCESSES_LIST[@]}))
current_test=0

for size in "${MATRIX_SIZES[@]}"; do
    for processes in "${PROCESSES_LIST[@]}"; do
        current_test=$((current_test + 1))
        
        grid_size=$(echo "sqrt($processes)" | bc)
        if [ $((size % grid_size)) -ne 0 ]; then
            echo "Skip: size $size doesn't divide into $grid_size (procs: $processes)"
            continue
        fi
        
        echo "Test $current_test/$total_tests: Matrix ${size}x${size}, Procs: $processes"
        
        if [ $processes -eq 1 ]; then
            result=$(./cannon $size 2>&1 | grep "CSV:" | sed 's/CSV://')
            echo "$result" >> $RESULTS_FILE
            echo "   Result: $result"
        else
            result=$(mpirun -np $processes ./cannon $size 2>&1 | grep "CSV:" | sed 's/CSV://')
            echo "$result" >> $RESULTS_FILE
            echo "   Result: $result"
        fi
        
        sleep 3
    done
done

echo "[3/5] Benchmark finished!"

if [ ! -s "$RESULTS_FILE" ]; then
    echo "ERROR: Results file is empty!"
    exit 1
fi

echo "[4/5] Painting plots..."

python3 << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('default')
sns.set_palette("tab10")

try:
    df = pd.read_csv('results/benchmark_results_final.csv')
    print("Read strings successfully:", len(df))
    print("Data:")
    print(df)
    
    if (df['parallel_time'] == 0).all() or (df['sequential_time'] == 0).all():
        print("WARNING: Zeroes values of time are detected!")
        print("Parallel time:", df['parallel_time'].unique())
        print("Sequential time:", df['sequential_time'].unique())
    
except Exception as e:
    print("Reading file error:", e)
    exit(1)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for processes in sorted(df['processes'].unique()):
    if processes == 1:
        continue
    subset = df[df['processes'] == processes].sort_values('matrix_size')
    
    if len(subset) > 0:
        plt.plot(subset['matrix_size'], subset['speedup'], 
                'o-', linewidth=3, markersize=8, 
                label=f'{processes} procs', alpha=0.8)
        
        ideal = [processes] * len(subset)
        plt.plot(subset['matrix_size'], ideal, '--', alpha=0.3, 
                linewidth=1, color='gray', 
                label=f'Ideal ({processes})' if processes == 4 else "")

plt.xlabel('Matrix size', fontsize=12)
plt.ylabel('Acceleration', fontsize=12)
plt.title('Acceleration of Cannon algorithm', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
for processes in sorted(df['processes'].unique()):
    if processes == 1:
        continue
    subset = df[df['processes'] == processes].sort_values('matrix_size')
    
    if len(subset) > 0:
        plt.plot(subset['matrix_size'], subset['efficiency'], 
                's-', linewidth=3, markersize=8, 
                label=f'{processes} procs', alpha=0.8)

plt.xlabel('Matrix size', fontsize=12)
plt.ylabel('Efficiency (%)', fontsize=12)
plt.title('Efficiency of Cannon algorithm', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
for processes in sorted(df['processes'].unique()):
    subset = df[df['processes'] == processes].sort_values('matrix_size')
    
    if len(subset) > 0:
        plt.plot(subset['matrix_size'], subset['parallel_time'], 
                '^-', linewidth=2, markersize=6, 
                label=f'{processes} procs', alpha=0.8)

plt.xlabel('Matrix size', fontsize=12)
plt.ylabel('Execution time (s)', fontsize=12)
plt.title('Parallel execution time', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/performance_final.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n=== Stats ===")
print(df.groupby('processes').agg({
    'speedup': ['count', 'mean', 'std', 'min', 'max'],
    'efficiency': ['mean', 'std', 'min', 'max'],
    'parallel_time': ['mean', 'std']
}).round(4))

print("\Plots are saved in folder plots/")
EOF

echo "[5/5] Clearing..."
rm -f cannon

echo "========================================="
echo "Benchmarking is completed!"
echo "Results: ${RESULTS_FILE}"
echo "Plots: plots/performance_final.png"
echo "========================================="