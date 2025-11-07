import matplotlib.pyplot as plt
import numpy as np

def read_data():
    data = []
    try:
        with open('times.txt', 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 5:
                        nx = int(parts[0])
                        p = int(parts[2])
                        time = float(parts[4])
                        data.append((nx, p, time))
    except:
        print("Ошибка чтения times.txt")
        return []
    return data

def calculate_metrics(data):
    metrics = {}
    
    for nx, p, time in data:
        if nx not in metrics:
            metrics[nx] = {}
        metrics[nx][p] = time
    
    results = []
    for nx in metrics:
        if 1 in metrics[nx]:
            seq_time = metrics[nx][1]
            for p in metrics[nx]:
                if p > 0:
                    time = metrics[nx][p]
                    speedup = seq_time / time
                    efficiency = speedup / p
                    results.append((nx, p, time, speedup, efficiency))
    
    return results

def plot_all(results):
    if not results:
        print("Нет данных для построения графиков")
        return
        
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    sizes = sorted(set(r[0] for r in results))
    processes = sorted(set(r[1] for r in results))
    
    for p in processes:
        times = [r[2] for r in results if r[1] == p]
        corresponding_sizes = [r[0] for r in results if r[1] == p]
        ax1.plot(corresponding_sizes, times, 'o-', label=f'P={p}', markersize=6)
    
    ax1.set_xlabel('Размер сетки')
    ax1.set_ylabel('Время (сек)')
    ax1.set_title('Время выполнения')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for size in sizes:
        speedups = [r[3] for r in results if r[0] == size]
        corresponding_procs = [r[1] for r in results if r[0] == size]
        ax2.plot(corresponding_procs, speedups, 's-', label=f'N={size}', markersize=6)
    
    max_p = max(processes)
    ideal = list(range(1, max_p + 1))
    ax2.plot(ideal, ideal, 'k--', label='Идеальное', alpha=0.7)
    
    ax2.set_xlabel('Количество процессов')
    ax2.set_ylabel('Ускорение')
    ax2.set_title('Ускорение')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    for size in sizes:
        efficiencies = [r[4] for r in results if r[0] == size]
        corresponding_procs = [r[1] for r in results if r[0] == size]
        ax3.plot(corresponding_procs, efficiencies, '^-', label=f'N={size}', markersize=6)
    
    ax3.set_xlabel('Количество процессов')
    ax3.set_ylabel('Эффективность')
    ax3.set_title('Эффективность')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nРезультаты:")
    print("Размер | Процессы | Время(с) | Ускорение | Эффективность")
    print("-" * 55)
    for nx, p, time, speedup, efficiency in sorted(results):
        print(f"{nx:6} | {p:8} | {time:8.2f} | {speedup:8.2f} | {efficiency:11.2f}")

def main():
    print("Анализ производительности параллельного алгоритма Пуассона")
    print("=" * 60)
    
    data = read_data()
    if not data:
        print("Нет данных для анализа!")
        return
    
    results = calculate_metrics(data)
    if not results:
        print("Не удалось вычислить метрики!")
        return
    
    plot_all(results)
    print("\nГрафики сохранены в 'results.png'")

if __name__ == "__main__":
    main()