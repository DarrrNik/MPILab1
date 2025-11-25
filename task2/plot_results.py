import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

for binary in df["binary"].unique():
    sub = df[df["binary"] == binary]
    grouped = sub.groupby("procs")["time_sec"].mean()

    plt.figure()
    plt.plot(grouped.index, grouped.values, marker="o")
    plt.xlabel("Processes")
    plt.ylabel("Time (s)")
    plt.title(f"Execution time: {binary}")
    plt.grid(True)
    plt.savefig(f"{binary}_plot.png")
