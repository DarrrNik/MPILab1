#!/bin/bash

CXX=mpic++
CXXFLAGS="-O3 -std=c++20"

COMMON_SRC="mpilab.cpp mpilab.hpp"
TARGETS=("rows" "cols" "blocks")
SOURCES=("row.cpp" "col.cpp" "block.cpp")

echo "=== Компиляция ==="
for i in ${!TARGETS[@]}; do
    echo "Компилируется ${TARGETS[$i]}..."
    $CXX $CXXFLAGS ${SOURCES[$i]} $COMMON_SRC -o ${TARGETS[$i]}
done

SIZES=(10000 15000 20000 25000 30000)
PROCS=(1 2 4 8)

echo "=== Запуск экспериментов ==="

for algo in "${TARGETS[@]}"; do
    for size in "${SIZES[@]}"; do
        for np in "${PROCS[@]}"; do
            echo "Запуск $algo с размером $size, процессоров $np..."
            total=0
            runs=3
            for ((i=1;i<=runs;i++)); do
                mpiexec -np $np ./${algo} $size
            done
        done
    done
done

echo "=== Построение графиков ==="

python3 << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

# === График времени выполнения ===
for size in sorted(df['size'].unique()):
    plt.figure()
    for algo in df['algo'].unique():
        sub = df[(df['algo']==algo) & (df['size']==size)]
        plt.plot(sub['np'], sub['time'], marker='o', label=algo)
    plt.xlabel('Количество процессов')
    plt.ylabel('Время выполнения (сек)')
    plt.title(f'Время выполнения (размер = {size})')
    plt.grid()
    plt.legend()
    plt.savefig(f'time_{size}.png')

# === График ускорения ===
for size in sorted(df['size'].unique()):
    plt.figure()
    for algo in df['algo'].unique():
        sub = df[(df['algo']==algo) & (df['size']==size)].set_index('np')
        T1 = sub.loc[1, 'time']
        speedup = T1 / sub['time']
        plt.plot(sub.index, speedup, marker='o', label=algo)
    plt.xlabel('Количество процессов')
    plt.ylabel('Ускорение (S = T1 / Tp)')
    plt.title(f'Ускорение (размер = {size})')
    plt.grid()
    plt.legend()
    plt.savefig(f'speedup_{size}.png')

# === График эффективности ===
for size in sorted(df['size'].unique()):
    plt.figure()
    for algo in df['algo'].unique():
        sub = df[(df['algo']==algo) & (df['size']==size)].set_index('np')
        T1 = sub.loc[1, 'time']
        efficiency = (T1 / sub['time']) / sub.index
        plt.plot(sub.index, efficiency, marker='o', label=algo)
    plt.xlabel('Количество процессов')
    plt.ylabel('Эффективность (E = S / P)')
    plt.title(f'Эффективность (размер = {size})')
    plt.grid()
    plt.legend()
    plt.savefig(f'efficiency_{size}.png')

print("Готово: сохранены графики time_*.png, speedup_*.png, efficiency_*.png")
EOF

echo "=== Эксперименты завершены ==="
