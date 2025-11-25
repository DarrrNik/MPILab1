#!/bin/bash

echo "========================================="
echo "  Matrix-Vector Multiplication Benchmark"
echo "========================================="

# Создаем директории
mkdir -p results
mkdir -p plots
mkdir -p executables

echo "[1/4] Compiling programs..."

# Компилируем все три версии
mpic++ -O3 -o executables/matrix_vector_rows row.cpp mpilab.cpp -lm
mpic++ -O3 -o executables/matrix_vector_cols col.cpp mpilab.cpp -lm
mpic++ -O3 -o executables/matrix_vector_blocks block.cpp mpilab.cpp -lm

if [ $? -ne 0 ]; then
    echo "Compilation error!"
    exit 1
fi

echo "Compilation successful!"

# Параметры тестирования
MATRIX_SIZES=(9000 18000 27000 36000)
PROCESSES_LIST=(1 4 9)
METHODS=("rows" "cols" "blocks")
RESULTS_FILE="results/benchmark_results.csv"

echo "matrix_size,processes,method,parallel_time,speedup,efficiency" > $RESULTS_FILE

echo "[2/4] Running benchmarks..."

# Функция для извлечения времени из вывода
extract_time() {
    local output="$1"
    # Ищем время в разных форматах: обычный и научная нотация
    local time=$(echo "$output" | grep "Execution time" | sed 's/.*Execution time.*<=> //' | sed 's/s.*//')
    
    if [[ "$time" == *"e-"* ]]; then
        # Конвертируем научную нотацию в обычное число
        echo "$time" | python3 -c "import sys; print(float(sys.stdin.read()))" 2>/dev/null
    else
        echo "$time"
    fi
}

# Функция для проверки делимости
check_divisibility() {
    local size=$1
    local processes=$2
    local method=$3
    
    case $method in
        "rows")
            # Для строчного метода: строки должны делиться на процессы
            return $((size % processes))
            ;;
        "cols")
            # Для колоночного метода: столбцы должны делиться на процессы
            return $((size % processes))
            ;;
        "blocks")
            # Для блочного метода: процессы должны быть perfect square и размеры делиться на sqrt(processes)
            local grid_size=$(echo "sqrt($processes)" | bc)
            if [ $((grid_size * grid_size)) -ne $processes ]; then
                return 1  # Не perfect square
            fi
            if [ $((size % grid_size)) -ne 0 ]; then
                return 1  # Не делится
            fi
            return 0
            ;;
    esac
}

# Запускаем тесты для каждого метода
for method in "${METHODS[@]}"; do
    echo "=== Testing $method method ==="
    
    # Запускаем последовательную версию для каждого размера матрицы
    for size in "${MATRIX_SIZES[@]}"; do
        echo "Sequential ${size}x${size} ($method method)"
        
        if ! check_divisibility $size 1 $method; then
            echo "   SKIP: Size $size not compatible with $method method"
            continue
        fi
        
        output=$(./executables/matrix_vector_$method $size $size 2>&1)
        time=$(extract_time "$output")
        
        if [ ! -z "$time" ] && [ "$time" != "0" ]; then
            echo "$size,1,$method,$time,1.0,100.0" >> $RESULTS_FILE
            echo "   Sequential time: ${time}s"
        else
            echo "   ERROR: Could not extract time"
            echo "$size,1,$method,0,0,0" >> $RESULTS_FILE
        fi
    done

    # Запускаем параллельные версии
    for size in "${MATRIX_SIZES[@]}"; do
        for processes in "${PROCESSES_LIST[@]}"; do
            # Пропускаем sequential для процессов > 1
            if [ $processes -eq 1 ]; then
                continue
            fi
            
            # Проверяем делимость
            if ! check_divisibility $size $processes $method; then
                echo "SKIP: Matrix ${size}x${size}, Processes: $processes ($method method - not compatible)"
                continue
            fi
            
            echo "Matrix ${size}x${size}, Processes: $processes ($method method)"
            
            # Запускаем MPI программу
            output=$(mpirun -np $processes ./executables/matrix_vector_$method $size $size 2>&1)
            parallel_time=$(extract_time "$output")
            
            if [ ! -z "$parallel_time" ] && [ "$parallel_time" != "0" ]; then
                # Находим sequential time для этого размера и метода
                sequential_line=$(grep "^$size,1,$method," $RESULTS_FILE)
                sequential_time=$(echo "$sequential_line" | cut -d',' -f4)
                
                if [ ! -z "$sequential_time" ] && [ "$sequential_time" != "0" ]; then
                    speedup=$(echo "scale=6; $sequential_time / $parallel_time" | bc -l 2>/dev/null || echo "0")
                    efficiency=$(echo "scale=6; $speedup / $processes * 100" | bc -l 2>/dev/null || echo "0")
                    
                    echo "$size,$processes,$method,$parallel_time,$speedup,$efficiency" >> $RESULTS_FILE
                    echo "   Parallel time: ${parallel_time}s, Speedup: $speedup, Efficiency: $efficiency%"
                else
                    echo "   WARNING: No valid sequential time found"
                    echo "$size,$processes,$method,$parallel_time,0,0" >> $RESULTS_FILE
                fi
            else
                echo "   ERROR: Could not extract time"
                echo "$size,$processes,$method,0,0,0" >> $RESULTS_FILE
            fi
            
            sleep 1
        done
    done
done

echo "[3/4] Generating plots..."

python3 << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('default')
sns.set_palette("tab10")

try:
    df = pd.read_csv('results/benchmark_results.csv')
    print("Data loaded successfully")
    print(f"Total records: {len(df)}")
    
    df = df[(df['parallel_time'] > 0) & (df['speedup'] >= 0)]
    print(f"Valid records after filtering: {len(df)}")
    
    if len(df) == 0:
        print("No valid data after filtering!")
        exit(1)
        
    print("\nData preview:")
    print(df.head(15))
    
except Exception as e:
    print(f"Error reading data: {e}")
    exit(1)

# Создаем графики с зависимостью от количества процессов
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# График 1: Время выполнения vs Количество процессов (по методам)
ax = axes[0, 0]
for method in df['method'].unique():
    for size in sorted(df['matrix_size'].unique()):
        subset = df[(df['method'] == method) & (df['matrix_size'] == size)].sort_values('processes')
        if len(subset) > 0:
            ax.plot(subset['processes'], subset['parallel_time'], 
                   'o-', linewidth=2, markersize=6, 
                   label=f'{method}, {size}x{size}')

ax.set_xlabel('Number of Processes')
ax.set_ylabel('Execution Time (s)')
ax.set_title('Execution Time vs Number of Processes')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xticks(sorted(df['processes'].unique()))

# График 2: Ускорение vs Количество процессов (по методам)
ax = axes[0, 1]
for method in df['method'].unique():
    for size in sorted(df['matrix_size'].unique()):
        subset = df[(df['method'] == method) & (df['matrix_size'] == size)].sort_values('processes')
        if len(subset) > 0:
            ax.plot(subset['processes'], subset['speedup'], 
                   's-', linewidth=2, markersize=6, 
                   label=f'{method}, {size}x{size}')
            
            # Идеальное ускорение для этого размера
            processes = subset['processes'].values
            ideal = processes
            ax.plot(processes, ideal, '--', alpha=0.3, color='gray')

ax.set_xlabel('Number of Processes')
ax.set_ylabel('Speedup')
ax.set_title('Speedup vs Number of Processes')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xticks(sorted(df['processes'].unique()))

# График 3: Эффективность vs Количество процессов (по методам)
ax = axes[1, 0]
for method in df['method'].unique():
    for size in sorted(df['matrix_size'].unique()):
        subset = df[(df['method'] == method) & (df['matrix_size'] == size)].sort_values('processes')
        if len(subset) > 0:
            ax.plot(subset['processes'], subset['efficiency'], 
                   '^-', linewidth=2, markersize=6, 
                   label=f'{method}, {size}x{size}')

ax.set_xlabel('Number of Processes')
ax.set_ylabel('Efficiency (%)')
ax.set_title('Efficiency vs Number of Processes')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xticks(sorted(df['processes'].unique()))
ax.axhline(y=100, color='red', linestyle='--', alpha=0.3, label='Ideal Efficiency')

# График 4: Сравнение методов для разных количеств процессов (средние значения)
ax = axes[1, 1]

# Группируем данные по методам и количеству процессов
methods_data = []
for method in df['method'].unique():
    for processes in sorted(df['processes'].unique()):
        subset = df[(df['method'] == method) & (df['processes'] == processes)]
        if len(subset) > 0:
            avg_speedup = subset['speedup'].mean()
            avg_efficiency = subset['efficiency'].mean()
            methods_data.append({
                'method': method,
                'processes': processes,
                'avg_speedup': avg_speedup,
                'avg_efficiency': avg_efficiency
            })

methods_df = pd.DataFrame(methods_data)

# Столбчатая диаграмма среднего ускорения по методам и процессам
bar_width = 0.25
processes_list = sorted(methods_df['processes'].unique())
methods_list = methods_df['method'].unique()
x_pos = np.arange(len(processes_list))

for i, method in enumerate(methods_list):
    method_data = methods_df[methods_df['method'] == method].sort_values('processes')
    if len(method_data) > 0:
        speeds = method_data['avg_speedup'].values
        ax.bar(x_pos + i * bar_width, speeds, bar_width, 
               label=method, alpha=0.7)

ax.set_xlabel('Number of Processes')
ax.set_ylabel('Average Speedup')
ax.set_title('Average Speedup by Method and Process Count')
ax.set_xticks(x_pos + bar_width * (len(methods_list) - 1) / 2)
ax.set_xticklabels(processes_list)
ax.legend()
ax.grid(True, alpha=0.3)

# Добавляем идеальное ускорение
ax.plot(x_pos, processes_list, 'ro--', label='Ideal Speedup', alpha=0.7)

plt.tight_layout()
plt.savefig('plots/processes_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Дополнительный график: тепловая карта эффективности по методам и процессам
plt.figure(figsize=(12, 8))

# Создаем сводную таблицу для тепловой карты
pivot_data = df.pivot_table(values='efficiency', 
                           index='method', 
                           columns='processes', 
                           aggfunc='mean')

sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', 
            cbar_kws={'label': 'Efficiency (%)'}, 
            square=True, linewidths=0.5)
plt.title('Efficiency Heatmap by Method and Process Count')
plt.xlabel('Number of Processes')
plt.ylabel('Method')
plt.tight_layout()
plt.savefig('plots/efficiency_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nPlots generated successfully!")
print(f"- processes_comparison.png")
print(f"- efficiency_heatmap.png")

print("\n=== Detailed Statistics by Process Count ===")
for processes in sorted(df['processes'].unique()):
    subset = df[df['processes'] == processes]
    if len(subset) > 0:
        print(f"\nProcesses: {processes}")
        print("-" * 40)
        for method in subset['method'].unique():
            method_data = subset[subset['method'] == method]
            if len(method_data) > 0:
                print(f"{method:8} | Time: {method_data['parallel_time'].mean():.3f}s | "
                      f"Speedup: {method_data['speedup'].mean():.2f} | "
                      f"Efficiency: {method_data['efficiency'].mean():.1f}%")

print("\n=== Overall Method Comparison ===")
overall_stats = df[df['processes'] > 1].groupby('method').agg({
    'speedup': ['mean', 'max', 'min'],
    'efficiency': ['mean', 'max', 'min'],
    'parallel_time': ['mean', 'min']
}).round(3)
print(overall_stats)

print(f"\nBest method by average speedup: {overall_stats[('speedup', 'mean')].idxmax()}")
print(f"Best method by average efficiency: {overall_stats[('efficiency', 'mean')].idxmax()}")
print(f"Fastest method: {overall_stats[('parallel_time', 'min')].idxmin()}")
EOF

echo "[4/4] Benchmark completed!"

echo "========================================="
echo "Results: results/benchmark_results.csv"
echo "Plots: plots/method_comparison.png"
echo "========================================="