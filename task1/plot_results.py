import matplotlib.pyplot as plt
import numpy as np

def read_results(filename):
    processes = []
    times = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    processes.append(int(parts[0]))
                    times.append(float(parts[2]))
    return processes, times

def plot_performance(processes, times):
    sequential_time = times[0] if processes[0] == 1 else None
    
    speedup = [sequential_time / t for t in times]
    efficiency = [s/p for s, p in zip(speedup, processes)]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1.plot(processes, times, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Количество процессов')
    ax1.set_ylabel('Время выполнения (с)')
    ax1.set_title('Время выполнения')
    ax1.grid(True)
    
    # График ускорения
    ax2.plot(processes, speedup, 'ro-', linewidth=2, markersize=8, label='Фактическое')
    ax2.plot(processes, processes, 'g--', linewidth=1, label='Идеальное')
    ax2.set_xlabel('Количество процессов')
    ax2.set_ylabel('Ускорение')
    ax2.set_title('Ускорение')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(processes, efficiency, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Количество процессов')
    ax3.set_ylabel('Эффективность')
    ax3.set_title('Эффективность')
    ax3.grid(True)
    
    ax4.loglog(processes, times, 'mo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Количество процессов')
    ax4.set_ylabel('Время выполнения (с)')
    ax4.set_title('Логарифмический масштаб')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    processes, times = read_results('performance_results.txt')
    plot_performance(processes, times)