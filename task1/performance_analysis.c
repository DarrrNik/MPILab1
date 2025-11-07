#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    int processes;
    long long points;
    double time;
    double pi;
} Result;

int main() {
    FILE *fp = fopen("performance_results.txt", "r");
    if (fp == NULL) {
        printf("Cannot open file with results\n");
        return 1;
    }
    
    Result results[100];
    int count = 0;
    double sequential_time = 0;
    
    while (fscanf(fp, "%d %lld %lf %lf", 
                  &results[count].processes, 
                  &results[count].points, 
                  &results[count].time, 
                  &results[count].pi) == 4) {
        if (results[count].processes == 1) {
            sequential_time = results[count].time;
        }
        count++;
    }
    fclose(fp);
    
    printf("\nPerformance analysis\n");
    printf("========================\n");
    printf("%-10s %-12s %-10s %-12s %-12s\n", 
           "Processes", "Time(s)", "Acceleration", "Efficienty", "Error Pi");
    printf("------------------------------------------------------------------------\n");
    
    for (int i = 0; i < count; i++) {
        double acceleration = sequential_time / results[i].time;
        double efficiency = acceleration / results[i].processes;
        double error = fabs(results[i].pi - M_PI);
        
        printf("%-10d %-12.6f %-10.2f %-12.2f %-12.8f\n",
               results[i].processes, results[i].time, acceleration, efficiency, error);
    }
    
    return 0;
}