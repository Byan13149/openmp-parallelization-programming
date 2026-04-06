#!/usr/bin/env python3
"""
Plot benchmark results: execution time and log|C| accuracy vs LAPACK reference.

Usage: python3 test/plot_bench.py [test/bench_results.csv]

Produces test/bench_time.png and test/bench_accuracy.png
"""
import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else "test/bench_results.csv"

# Parse CSV: impl -> [(n, time, logdet, logdet_lapack), ...]
data = {}
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        impl = row["impl"]
        n = int(row["n"])
        t = float(row["time"])
        logdet = float(row["logdet"])
        logdet_ref = float(row["logdet_lapack"])
        data.setdefault(impl, []).append((n, t, logdet, logdet_ref))

# Sort each impl's entries by n
for impl in data:
    data[impl].sort()

# Plot order
order = ["baseline", "opt1", "opt2", "opt3", "omp1", "omp2", "omp3"]
impls = [i for i in order if i in data]
impls += [i for i in data if i not in impls]

markers = ["o", "s", "^", "D", "v", "P", "X"]
colors = ["#888888", "#1f77b4", "#2ca02c", "#d62728",
          "#ff7f0e", "#9467bd", "#17becf"]

# --- Time plot ---
fig, ax = plt.subplots(figsize=(8, 5))
for idx, impl in enumerate(impls):
    ns = [r[0] for r in data[impl]]
    times = [r[1] for r in data[impl]]
    ax.plot(ns, times, marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)], label=impl, linewidth=1.5)

ax.set_xlabel("Matrix size n")
ax.set_ylabel("Time (s)")
ax.set_title("Cholesky Factorization: Execution Time")
ax.legend()
ax.set_yscale("log")
# ax.set_xscale("log")
ax.grid(True, which="both", ls="--", alpha=0.5)
fig.tight_layout()
fig.savefig("test/bench_time.png", dpi=150)
print("Saved test/bench_time.png")

# --- Accuracy plot (relative error vs LAPACK dpotrf) ---
fig, ax = plt.subplots(figsize=(8, 5))
for idx, impl in enumerate(impls):
    ns = [r[0] for r in data[impl]]
    rel_diffs = [abs(r[2] - r[3]) / (abs(r[3]) + 1e-30) for r in data[impl]]
    ax.plot(ns, rel_diffs, marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)], label=impl, linewidth=1.5)

ax.set_xlabel("Matrix size n")
ax.set_ylabel("Relative error in log|C|")
ax.set_title("Cholesky Factorization: Accuracy vs LAPACK dpotrf")
ax.legend()
# ax.set_xscale("log")
# ax.set_yscale("log")
ax.grid(True, which="both", ls="--", alpha=0.5)
fig.tight_layout()
fig.savefig("test/bench_accuracy.png", dpi=150)
print("Saved test/bench_accuracy.png")

# --- Thread scaling plot ---
import os
scaling_path = "test/bench_scaling.csv"
if os.path.exists(scaling_path):
    # Parse: n -> [(threads, time), ...]
    scaling = {}
    with open(scaling_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            threads = int(row["threads"])
            n = int(row["n"])
            t = float(row["time"])
            scaling.setdefault(n, []).append((threads, t))

    for n in scaling:
        scaling[n].sort()

    sizes = sorted(scaling.keys())
    markers_s = ["o", "s", "^", "D", "v", "P", "X", "h"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, n in enumerate(sizes):
        threads = [r[0] for r in scaling[n]]
        times = [r[1] for r in scaling[n]]
        # Speedup relative to 1 thread
        t1 = times[0]  # first entry is 1 thread (sorted)
        speedups = [t1 / t for t in times]
        ax.plot(threads, speedups, marker=markers_s[idx % len(markers_s)],
                label=f"n={n}", linewidth=1.5)

    # Ideal linear speedup
    all_threads = sorted(set(t for n in sizes for t, _ in scaling[n]))
    ax.plot(all_threads, all_threads, ls="--", color="black",
            linewidth=1, label="Ideal (linear)")

    ax.set_xlabel("Number of threads")
    ax.set_ylabel("Speedup vs 1 thread")
    ax.set_title("omp3: Thread Scaling")
    ax.legend(loc="upper left", fontsize=6.5)
    try:
        ax.set_xscale("log", base=2)
    except (TypeError, ValueError):
        ax.set_xscale("log", basex=2)
    # try:
    #     ax.set_yscale("log", base=2)
    # except TypeError:
    #     ax.set_yscale("log", basey=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig("test/bench_scaling.png", dpi=150)
    print("Saved test/bench_scaling.png")
else:
    print("No scaling data found (test/bench_scaling.csv), skipping scaling plot.")
