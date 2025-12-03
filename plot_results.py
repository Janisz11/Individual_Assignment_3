import pandas as pd
import matplotlib.pyplot as plt
import os


os.makedirs("plots", exist_ok=True)


df = pd.read_csv("results.csv",encoding="utf-16")

print("=== Raw results loaded ===")
print(df)


parallel = df[df["variant"] == "parallel"]

plt.figure(figsize=(8, 5))
for size, sub in parallel.groupby("size"):
    sub_sorted = sub.sort_values("threads")
    plt.plot(sub_sorted["threads"], sub_sorted["speedup"],
             marker="o", label=f"n={size}")

plt.xlabel("Number of threads")
plt.ylabel("Speedup (vs basic)")
plt.title("Parallel speedup vs thread count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/speedup_vs_threads.png", dpi=300)
plt.close()

print("Saved plots/speedup_vs_threads.png")


sizes = sorted(df["size"].unique())

for size in sizes:
    sub = df[df["size"] == size]

    basic = sub[(sub["variant"] == "basic") & (sub["threads"] == 1)].iloc[0]
    vectorized = sub[(sub["variant"] == "vectorized") & (sub["threads"] == 1)].iloc[0]

    par_sub = sub[sub["variant"] == "parallel"]
    max_threads = par_sub["threads"].max()
    parallel_best = par_sub[par_sub["threads"] == max_threads].iloc[0]

    labels = [
        "Basic (1 thread)",
        "Vectorized (1 thread)",
        f"Parallel ({int(max_threads)} threads)"
    ]
    times = [
        basic["time_ms"],
        vectorized["time_ms"],
        parallel_best["time_ms"]
    ]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, times)
    plt.ylabel("Time [ms]")
    plt.title(f"Execution time comparison (n={size})")
    plt.tight_layout()
    
    fname = f"plots/times_n{size}.png"
    plt.savefig(fname, dpi=300)
    plt.close()

    print(f"Saved {fname}")

print("\nAll plots saved in ./plots/")
