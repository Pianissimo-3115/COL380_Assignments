#!/usr/bin/env python3
"""
Generate test cases and benchmark KNN / K-Means / Approx-KNN.
Usage:  python3 bench.py          (builds, generates, runs, prints LaTeX table)
"""

import random, os, subprocess, time, sys

BINARY = "./a2"
TESTS_DIR = "bench_inputs"
MODES = ["knn", "kmeans", "approx_knn"]

# (n, k, T) combos that exercise different scaling regimes
CASES = [
    # small
    (1000,   8,  10),
    (5000,  16,  10),
    # medium
    (10000,  32, 20),
    (20000,  64, 20),
    # large
    (50000,  64, 30),
    (50000, 128, 30),
    (100000, 64, 30),
    (100000,128, 50),
]

def generate(n, k, T, path):
    """Write a random point-cloud file."""
    rng = random.Random(42)
    pts = set()
    with open(path, "w") as f:
        f.write(f"{n}\n{k}\n{T}\n")
        while len(pts) < n:
            x = rng.randint(-10000, 10000)
            y = rng.randint(-10000, 10000)
            z = rng.randint(-10000, 10000)
            if (x, y, z) in pts:
                continue
            pts.add((x, y, z))
            I = rng.randint(0, 255)
            f.write(f"{x} {y} {z} {I}\n")

def run_timed(binary, infile, mode, repeats=1):
    """Run binary and return wall-clock seconds (best of repeats)."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = subprocess.run([binary, infile, mode],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        t1 = time.perf_counter()
        if r.returncode != 0:
            return None
        best = min(best, t1 - t0)
    return best

def main():
    # ── Build ──────────────────────────────────────────────
    print("Building …")
    r = subprocess.run(["make", "-j"], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        sys.exit(1)
    print("Build OK\n")

    os.makedirs(TESTS_DIR, exist_ok=True)

    # ── Generate inputs ────────────────────────────────────
    print("Generating inputs …")
    paths = {}
    for n, k, T in CASES:
        name = f"n{n}_k{k}_T{T}"
        p = os.path.join(TESTS_DIR, f"{name}.txt")
        if not os.path.exists(p):
            generate(n, k, T, p)
        paths[(n, k, T)] = p
    print("Done\n")

    # ── Benchmark ──────────────────────────────────────────
    results = {}   # (n,k,T) -> {mode: seconds}
    for (n, k, T), infile in paths.items():
        results[(n, k, T)] = {}
        tag = f"n={n:>6}, k={k:>3}, T={T:>2}"
        for mode in MODES:
            t = run_timed(BINARY, infile, mode, repeats=1)
            ts = f"{t:.4f}s" if t is not None else "FAIL"
            print(f"  {tag}  {mode:<12} {ts}")
            results[(n, k, T)][mode] = t
        print()

    # ── Print LaTeX table ─────────────────────────────────
    print("\n% ── paste into report.tex ──")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Wall-clock runtime (seconds) on the test GPU}")
    print(r"\label{tab:bench}")
    print(r"\small")
    print(r"\renewcommand{\arraystretch}{1.2}")
    print(r"\begin{tabular}{@{}rrr|rrr@{}}")
    print(r"\toprule")
    print(r"$n$ & $k$ & $T$ & \textbf{KNN} & \textbf{K-Means} & \textbf{Approx KNN} \\")
    print(r"\midrule")
    for (n, k, T) in CASES:
        r = results[(n, k, T)]
        def fmt(v):
            return f"{v:.4f}" if v is not None else "---"
        print(f"{n:>6} & {k:>3} & {T:>2} & "
              f"{fmt(r.get('knn'))} & {fmt(r.get('kmeans'))} & "
              f"{fmt(r.get('approx_knn'))} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # also save raw CSV
    csv_path = "bench_results.csv"
    with open(csv_path, "w") as f:
        f.write("n,k,T,knn,kmeans,approx_knn\n")
        for (n, k, T) in CASES:
            r = results[(n, k, T)]
            def fmt(v):
                return f"{v:.6f}" if v is not None else ""
            f.write(f"{n},{k},{T},{fmt(r.get('knn'))},{fmt(r.get('kmeans'))},"
                    f"{fmt(r.get('approx_knn'))}\n")
    print(f"\nRaw CSV saved to {csv_path}")

if __name__ == "__main__":
    main()
