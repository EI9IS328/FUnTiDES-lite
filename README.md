# FUnTiDES: Fast Unstructured Time Dynamic Equation Solver

**FUnTiDES** is a collection of simplified codes that represent real scientific applications. It serves as a standard tool for evaluating and comparing the performance of various high-performance computing (HPC) systems, particularly those used for scientific simulations.

---

## Included Applications

The current implementation includes two proxy applications for solving the 2nd-order acoustic wave equation in 2D and 3D:

- **SEM (Spectral Element Method)**
  A benchmark designed to simulate wave propagation using SEM, a Galerkin-based finite element method for solving partial differential equations (PDEs).

- **FD (Finite Difference Method)**
  A benchmark that uses finite-difference stencil operators to simulate wave propagation and solve PDEs.

A key feature of these proxy applications is their adaptability to different programming models and HPC architectures. They are also easy to build and run, making them accessible to both researchers and developers.

---

## Supported Programming Models

The SEM proxy currently supports:

- [Kokkos](https://kokkos.github.io/kokkos-core-wiki/) — for performance portability

> **Note**: Kokkos is included as a Git submodule and will be compiled automatically when enabled.

---

## Supported Data Containers

The current SEM proxy supports the following data container:

- `std::vector` (default for serial )

---

## Quick Start: Build and Run

### Step 1: Compile and Install

```sh
mkdir build
cd build
cmake ..
make install
```

By default, this builds the applications in sequential mode using `std::vector`.
Both SEM and FD applications are compiled.

### Step 2: Run Examples

```sh
# Run SEM simulation with 100 x 100 x 100 elements
./src/main/semproxy -ex 100

# Run FD simulation
./src/main/fdproxy
```

---

## CMake Options

The following options can be used to configure your build:

| Option                 | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `COMPILE_FD`           | Enable compilation of the FD proxy (default: ON)                            |
| `COMPILE_SEM`          | Enable compilation of the SEM proxy (default: ON)                           |
| `ENABLE_CUDA`          | Enable CUDA backend (used by Kokkos)                                        |
| `ENABLE_PYWRAP`        | Enable Python bindings via pybind11 (experimental)                          |
| `USE_KOKKOS`           | Enable Kokkos support (serial by default, CUDA/OpenMP with flags)           |
| `USE_VECTOR`           | Use `std::vector` for data arrays (enabled by default unless Kokkos is used)|

---

## TP2: separate snapshot and sismogram comparisons

Rebuild `semproxy` after these C++ changes. The new simulation option is
`--outputs snapshots|sismograms|both` (default: `both`). It selects raw outputs
in ad-hoc mode and the corresponding statistics in in-situ mode. Snapshot-only
runs skip receiver allocation and interpolation; signal-only runs skip field
statistics and snapshots. The pressure solver is unchanged.

From the repository root, using your Python environment with NumPy/Matplotlib:

```sh
python3 benchmarks/compare.py benchmarks/compare_config.json --cases S
python3 benchmarks/compare.py benchmarks/compare_config.json --cases S --data snapshots
python3 benchmarks/compare.py benchmarks/compare_config.json --cases S M L --data sismograms
```

`--cases` defaults to S M L; `--data` defaults to both independent campaigns.
Edit sizes, repetitions, physics and `precision_intervals` in the configuration.
Paths are relative to that JSON. Use the same executable and resource settings
throughout; for OpenMP, set `OMP_NUM_THREADS` before launching Python. This is
a problem-size study, not strong/weak scaling.

For each type and size, the script alternates ad-hoc/in-situ order across
repetitions. Both simulate every physical step. For timing, field statistics
use the same steps as the saved snapshots (0, k, 2k, ...); receiver statistics
use every step in both modes. The full workflow clock includes process startup,
simulation, raw writes, Python reads/analysis for ad-hoc, and statistics JSON
writing. JSON reading for reporting and report/figure generation are outside
this clock. No normalization per snapshot is applied.

Writing results is distinct from calculating statistics. In-situ field statistics
are streamed into JSON at selected steps; receiver statistics are written at the
end. No raw pressure histories are kept in situ. Reads of pressure in memory
belong to statistical calculation; raw disk reads occur only in ad-hoc.

Snapshots also get one in-situ reference at every step, then actual ad-hoc
outputs for each configured precision interval. The first timed ad-hoc trial is
reused at the timing cadence. These extra runs are excluded from timing medians.
The script prints the simulation count and estimated raw snapshot storage;
saving at k=1 can require substantial space. With default configuration, group S
and both data types run 75 simulations (60 timing + 15 precision).

Dated `benchmarks/results/tp2-*` directories contain independent runs, logs,
configuration/environment copies and progressive CSV files:

- `runs.csv`: every timed run; `summary.csv`: medians by type, size and mode.
- `validation.csv`: paired statistics and absolute differences, without an
  arbitrary acceptance threshold. Invalid files/metadata still cause an error.
- `precision.csv`: actual snapshot peaks and temporal means for each k and the
  in-situ reference; not a physical exact solution.

After all measurements, the script generates:

- `costs_snapshots.svg`, `costs_sismograms.svg`: log time axis, separate raw
  write, raw read, statistics calculation and statistics JSON write curves.
  Zero raw in-situ I/O is stated explicitly and omitted from the log axis.
- `total_snapshots.svg`, `total_sismograms.svg`: actual complete workflow times
  from the independent campaigns, including physics and all result writes.
- `writes.svg`: grouped bars, rows for snapshots/signals, columns for total
  write time and volume (raw + statistics). Values are labelled above bars.
- `peak_evolution_n*.svg`: in-situ maximum absolute pressure at every step versus
  actual ad-hoc sample points. No additional error-curve figure.
- `statistics.html`: four tables: full time/storage, operation medians, snapshot
  quality by cadence, and per-receiver statistical values from the first trial.
  It contains no figures and is generated once, outside the timed workflows.

Performance figures display medians only. Raw individual results remain in CSV.
Different Python/NumPy and C++/Kokkos implementations affect analysis costs;
GPU/host transfer times are not isolated. Disk caches are not explicitly
controlled. Logs, metric files and reporting artifacts are excluded from data
volume totals. The referenced measurement-methodology annex is absent from
the supplied `doc/` files. Compilation and actual backend execution remain to
be checked locally.
