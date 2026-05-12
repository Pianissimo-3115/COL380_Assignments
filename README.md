# COL380 — Parallel and Distributed Programming

Implementations of three assignments from COL380 (Parallel and Distributed Programming) at IIT Delhi, each targeting a different parallel-computing paradigm.

| # | Topic | Paradigm | Result |
|---|---|---|---|
| 1 | Order-book analytics on a bit-stuffed packet stream | OpenMP | 9.11× on 16 threads |
| 2 | k-NN, k-Means, and approximate k-NN on a 3D point cloud | CUDA | n = 100 000, k = 128 in ~4 s |
| 3 | Budget-constrained Maximum Weight Clique | MPI (one-sided RMA) | 7.32× on 16 processes |

---

## Assignment 1 — OpenMP: Order-Book Analytics

OpenMP implementations of three routines that process a stream of stock-exchange order-book packets. Each incoming record is a 49-bit field (`stockID`, buy/sell flag, quantity, price) bit-stuffed into a 64-bit integer using HDLC-style stuffing — a `0` is inserted after every five consecutive `1`s.

Functions defined in `functions.cpp`:

- **`updateDisplay(book, freq)`** — writes `snap_<k>.txt` after every `freq` packets, listing stocks in decreasing order of bid-ask spread (ties broken by `stockID`).
- **`totalAmountTraded(book)`** — computes the sum of `qty × price` across the order book.
- **`printOrderStats(book)`** — writes per-stock minimum sell, maximum buy, and average order value to `stats.txt`.

### Implementation notes

- Snapshot generation is data-parallel across snapshot indices using `#pragma omp parallel for schedule(dynamic)`.
- Per-stock statistics aggregation avoids contention by maintaining thread-local hash maps that are merged once at the end inside a single `critical` section, rather than locking on every update.
- `totalAmountTraded` uses a straightforward `reduction(+:total)`.

### Build and run

```bash
cd Assignment1
make all                          # produces ./testgen
./testgen <freq> <size>
./benchmark.sh                    # scaling sweep over 1, 4, 8, 12, 16 threads
```

### Speedup (n = 10⁸ packets, freq = 10⁶)

| threads | updateDisplay | printOrderStats | totalAmountTraded |
|---:|---:|---:|---:|
| 1  | 1.09× | 1.09× | 0.93× |
| 4  | 3.44× | 3.48× | 2.93× |
| 8  | 6.09× | 5.31× | 4.48× |
| 12 | 8.21× | 7.66× | 6.42× |
| 16 | 9.11× | 8.60× | 7.03× |

---

## Assignment 2 — CUDA: 3D Point-Cloud Search

CUDA implementations of three geometric routines over a 3D point cloud (up to 100 000 points, each with an 8-bit intensity). A single executable selects the routine via a command-line flag:

- **`knn`** — exact k-nearest-neighbours followed by per-point histogram equalisation of intensity over the neighbourhood.
- **`kmeans`** — k-means clustering with at most `T` iterations.
- **`approx_knn`** — approximate k-NN using an IVF index built on k-means centroids; falls back to exact k-NN when the problem size does not justify the index overhead.

### Implementation notes

- **Structure-of-Arrays layout.** Points are stored as four separate `int` arrays (`x`, `y`, `z`, `I`), so warp-level accesses coalesce into 128-byte transactions.
- **Batched allocation.** Each pipeline issues a single `cudaMalloc` and carves device buffers from it via pointer arithmetic, avoiding repeated allocator overhead.
- **Tiled shared memory.** Blocks of 128 threads cooperatively load 128 candidate points into shared memory per tile, reducing 128² global reads to 128 global plus 128² shared reads.
- **Register-resident priority queue.** The k-NN max-heap uses split key/value arrays small enough for the compiler to keep in registers or L1 local memory; all heap operations are marked `__forceinline__`.
- **Fused k-NN kernel.** Distance search, histogram construction, CDF computation, and intensity remapping are performed in a single kernel launch, keeping intermediate neighbour lists out of global memory.
- **int32 arithmetic.** The maximum squared distance, `3 · (2·10⁴)² ≈ 1.2·10⁹`, fits within `INT_MAX`, so 64-bit operations are avoided throughout.

### Build and run

```bash
cd Assignment2
make                                # produces ./a2
python3 dataset_generator.py        # writes input.txt
./a2 input.txt knn                  # writes knn.txt
./a2 input.txt kmeans               # writes kmeans.txt
./a2 input.txt approx_knn           # writes approx_knn.txt
python3 mae_loss.py                 # compares approx vs exact accuracy
```

Input format: first line `n k T`, then `n` lines of `x y z I`.

### Wall-clock benchmarks (seconds)

| n | k | exact knn | k-means | approx_knn |
|---:|---:|---:|---:|---:|
| 1 000   | 8   | 0.27 | 0.21 | 0.18 |
| 10 000  | 32  | 0.20 | 0.18 | 0.20 |
| 50 000  | 128 | 1.82 | 0.20 | 0.67 |
| 100 000 | 128 | 3.97 | 0.24 | 2.46 |

---

## Assignment 3 — MPI: Budget-Constrained Maximum Weight Clique

MPI-parallel branch-and-bound solver for the budget-constrained Maximum Weight Clique problem: given an undirected graph in which each vertex carries a profit `p(v)` and a cost `c(v)`, and a budget `B`, find the clique with maximum total profit whose total cost is at most `B`.

### Sequential branch-and-bound

- Vertices are sorted once by decreasing `p/c` ratio; the order propagates through recursion (each `C_next` is a subsequence of its parent), eliminating re-sorting and surfacing high-profit cliques early to tighten `P_max`.
- Pruning uses two bounds: fractional knapsack on `p/c`, and greedy graph colouring (the clique number is bounded by the chromatic number of the candidate subgraph).
- The adjacency matrix is stored as a flat `uint8_t[N × N]` to avoid bit-unpacking in the inner adjacency checks and to keep cache utilisation predictable.
- Colour-class and scratch buffers are reused across recursive calls; only previously-used entries are cleared between calls.

### Parallelisation: dynamic task pool over one-sided RMA

There is no manager rank and no per-task message exchange. Work distribution is mediated by atomic operations on shared MPI windows.

1. **Task generation.** Rank 0 expands the search tree to depth 1 or 2 — whichever produces at least `4 × nprocs` tasks — and broadcasts the serialised pool.
2. **Atomic claim and solve.** Each rank claims its next task with `MPI_Fetch_and_op` (fetch-and-increment) against a window-resident counter. Every 500 B&B nodes, ranks push their local best to a shared `P_max` via `MPI_Fetch_and_op(MPI_MAX)`, propagating tight pruning bounds across ranks without barriers. When fewer than `2 × nprocs` tasks remain unclaimed, ranks donate unexplored subtrees to a spare buffer, which is `MPI_Allgatherv`'d at round end to form the next round's pool, reducing late-round idling.
3. **Reduction.** `MPI_Allreduce` with `MPI_MAXLOC` identifies the rank holding the best solution, which then broadcasts its clique.

### Build and run

```bash
cd Assignment3
mpic++ -O3 -std=c++17 main.cpp -o main_new
mpirun -np 8 ./main_new input.txt output.txt
```

Input format: first line `N E B`, then `N` lines of `p[i] c[i]`, followed by `E` edges `u v`.
Output: first line is the optimal profit; second line lists the clique vertices in ascending order.

### Scaling (provided test graph: N = 500, |E| ≈ 75 000, B = 90)

| processes | time (s) | speedup | efficiency |
|---:|---:|---:|---:|
| 1  | 8.08 | 1.00× | 1.00 |
| 2  | 4.18 | 1.93× | 0.97 |
| 4  | 2.30 | 3.51× | 0.88 |
| 8  | 1.43 | 5.66× | 0.71 |
| 12 | 1.18 | 6.87× | 0.57 |
| 16 | 1.10 | 7.32× | 0.46 |

Strong scaling holds through 8 processes. Beyond that, `Allgatherv` round overhead and shrinking per-task granularity erode efficiency, and the search tree's intrinsic parallelism becomes the limiting factor.

---

## Build dependencies

- C++17 compiler with OpenMP support
- CUDA toolkit (compute capability ≥ 7.0 recommended)
- An MPI implementation providing `mpic++` and `mpirun` (Open MPI or MPICH)
- Python 3 with `numpy` and `matplotlib` for the benchmark and plotting scripts

---

Popat Nihal Alkesh · 2023CS10058 · IIT Delhi
