#!/bin/bash

echo "Компиляция программы..."
mpicc -o derihlet main.c -lm

if [ $? -ne 0 ]; then
    echo "Ошибка компиляции!"
    exit 1
fi

echo "Компиляция успешна!"
echo "Запуск экспериментов..."
> times.txt

# Размеры сеток для тестирования
SIZES=(100 200 300 400)
# Количество процессов
PROCS=(1 4 9 16)

for size in "${SIZES[@]}"; do
    for procs in "${PROCS[@]}"; do
        # Проверяем, что количество процессов - квадрат целого числа
        sqrt_proc=$(echo "sqrt($procs)" | bc)
        if [ $((sqrt_proc * sqrt_proc)) -ne $procs ]; then
            echo "Пропускаем P=$procs (не квадрат целого числа)"
            continue
        fi
        
        # Проверяем делимость
        if [ $((size % sqrt_proc)) -ne 0 ]; then
            echo "Пропускаем: размер $size не делится на $sqrt_proc"
            continue
        fi
        
        echo "Запуск: размер $size x $size, процессы: $procs"
        mpiexec --oversubscribe -n $procs ./derihlet $size $size
        
        if [ $? -eq 0 ]; then
            echo "✓ Успешно: $size x $size, P=$procs"
        else
            echo "✗ Ошибка: $size x $size, P=$procs"
        fi
        
        sleep 1
    done
done

echo "Все эксперименты завершены!"
echo "Построение графиков..."
python3 plot_results.py