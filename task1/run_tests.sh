#!/bin/bash

echo "Running performance tests..."

> performance_results.txt

mpicc -o monte_carlo_pi main.c -lm

POINTS=100000000

for PROCESSES in 1 2 4 8
do
    echo "Run with $PROCESSES procs..."
    mpirun -np $PROCESSES ./monte_carlo_pi $POINTS
done

echo "Results analyzing..."
gcc -o performance_analysis performance_analysis.c -lm
./performance_analysis

if [ -f "plot_results.py" ]; then
    echo "Painting plots..."
    python3 plot_results.py
    if [ $? -eq 0 ]; then
        echo "Plots painted successfully"
    else
        echo "Cannot paint plots"
    fi
fi

echo "Testing finished!"