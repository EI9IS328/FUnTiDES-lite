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
and both data types run 25 simulations (20 timing + 5 precision).

Dated `benchmarks/results/tp2-*` directories contain independent runs, logs,
configuration/environment copies and progressive CSV files:

- `runs.csv`: every timed run; `summary.csv`: medians by type, size and mode.
- `validation.csv`: paired statistics and absolute differences, without an
  arbitrary acceptance threshold. Invalid files/metadata still cause an error.
- `precision.csv`: actual snapshot peaks and temporal means for each k and the
  in-situ reference; not a physical exact solution.

After all measurements, the script puts SVGs in the campaign's `figure/`
subdirectory. CSVs and `statistics.html` remain at the campaign root:

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

## Comparaison des extrema CPU hôte / GPU

Le protocole, les formules, les chronomètres et les limites sont détaillés dans
[le rapport CPU/GPU](benchmarks/CPU_GPU_REPORT.md), avec
[le schéma du protocole](benchmarks/cpu_gpu_protocol.svg).
Le code est fourni ; la compilation et la validation CUDA restent à effectuer
sur la machine GPU. Les vérifications locales ne constituent pas une mesure GPU.

```sh
python3 benchmarks/compare.py benchmarks/cpu_gpu_config.json --compare-cpu-gpu --cases S M L
```

Le fichier dédié contient seulement exécutable, tailles, répétitions et paramètres
physiques. Le pilote lance deux chemins in-situ sur le même exécutable CUDA :
`--extrema-backend cpu` (parcours séquentiel après mise à disposition sur l'hôte)
et `--extrema-backend kokkos` (réduction GPU et retour des résultats).
Seuls les extrema changent de lieu de calcul, pas l'onde ni moyenne/variance.
Sans l'option C++ expérimentale, le traitement habituel reste automatique.
Kokkos/OpenMP permet de vérifier les chemins localement après compilation,
mais le pilote `--compare-cpu-gpu` exige CUDA et refuse de le présenter comme GPU.

Pour trois tailles et cinq répétitions, 30 simulations sont lancées, sans
snapshots bruts. Les nouvelles campagnes `results/tp2-cpu-gpu-*` contiennent
`runs.csv`, `summary.csv`, `extrema.csv`, `cpu_gpu.html`, le rapport du protocole,
la configuration, l'environnement et les fichiers de chaque essai.
La figure de comparaison et le schéma sont dans `figure/`.
La génération du rapport et des figures est hors des chronomètres.

## Strong scaling Kokkos/OpenMP

Cette expérience garde le problème fixe et fait varier seulement le nombre de
threads CPU. Utiliser un même exécutable **Release, Kokkos/OpenMP, sans CUDA**
pour toute la campagne et le recompiler après les modifications de `main.cc`.
Les fichiers CMake ne sont pas modifiés. Le script vérifie le mode Release
dans le `CMakeCache.txt` du build et le backend réellement annoncé par Kokkos.

Depuis la racine, avec NumPy et Matplotlib dans l'environnement Python :

```sh
python3 benchmarks/strong_scaling.py benchmarks/strong_scaling_config.json
```

La configuration contient une seule `size` : `ex = ey = ez = size`. Les
paramètres physiques, l'ordre et la cadence restent fixes pour toutes les
exécutions. Les chemins sont relatifs au JSON. Adapter `executable` à votre
build et `threads` aux cœurs disponibles sur la machine ; la liste commence
par 1 pour fournir la référence. Le script refuse de dépasser le nombre de
cœurs physiques détectés dans l'affinité du processus. Si cette topologie
n'est pas accessible, il vérifie les processeurs logiques et le signale.
Il ne réserve pas les ressources : utiliser une machine ou une allocation
disponible, sans autres travaux concurrents.

Pour chaque nombre de threads, snapshots et sismogrammes sont mesurés
séparément, en ad-hoc et en in-situ, cinq fois par défaut. Les snapshots sont
analysés aux mêmes indices `0, k, 2k, ...` dans les deux approches ; les
sismogrammes utilisent les mêmes récepteurs et tous les pas. Il n'y a pas de
campagne de précision supplémentaire. Le réglage par défaut `N=40`,
`threads=[1,2,4,8]`, cinq répétitions correspond à **80 simulations** et environ
**16 GB de snapshots bruts conservés**. L'estimation utilise le nombre nominal
de pas ; les compteurs mesurés restent la référence.

Le pilote demande explicitement les threads à Kokkos et OpenMP, désactive
l'ajustement dynamique et utilise `OMP_PLACES=cores`, `OMP_PROC_BIND=spread`.
Le pool Kokkos effectif est contrôlé dans chaque log, ainsi que le backend.
Les processus Python d'analyse gardent OpenMP et les bibliothèques numériques
à un thread. C'est donc la scalabilité du **workflow avec le solveur et les
analyses Kokkos parallélisés**, avec un post-traitement Python fixe. Les E/S et
le code séquentiel ne deviennent pas automatiquement parallèles.

Chaque répétition alterne l'ordre ad-hoc/in-situ ; le parcours des nombres de
threads est également inversé une répétition sur deux. Aucun échauffement ni
vidage des caches n'est ajouté. Les appels `--help` de vérification initialisent
les pools de threads, sans exécuter la physique et hors chronométrage.

Un chronomètre externe couvre le workflow complet : démarrage et fin des
processus, simulation et écritures, puis lecture/analyse Python en ad-hoc.
Le JSON de statistiques est inclus dans les deux approches. La lecture des
rapports, les contrôles et la génération des figures sont hors de ce temps.
Les sorties de processus sont redirigées vers leurs logs dans les deux cas.

Les résultats sont dans `benchmarks/results/tp2-strong-scaling-…/` :

- `runs.csv` : toutes les mesures individuelles, avec nombre de threads,
  approche, répétition, type de données et compteurs de calcul/analyse/E/S ;
- configuration, récepteurs et `environment.json` : machine, affinité,
  compilation, versions Python/NumPy/Matplotlib et réglages des sous-processus ;
- un dossier par exécution contenant logs, métriques, statistiques et données
  brutes éventuelles ; aucun fichier brut n'est supprimé par le pilote ;
- `figure/strong_scaling_snapshots.svg` et `figure/strong_scaling_sismograms.svg`.

Chaque figure contient trois panneaux selon le nombre de threads `p` :
temps médian du workflow `T(p)`, accélération `S(p)=T(1)/T(p)` et efficacité
`E(p)=100*S(p)/p`. Chaque approche utilise sa propre référence à un thread.
Ad-hoc est en ligne continue et in-situ en ligne discontinue ; seules les
médianes sont représentées. Les références idéales sont `S(p)=p` et `E(p)=100 %`.
Les rapports sont calculés à partir des médianes, pas en moyennant des rapports
entre essais. Une approche peut être plus rapide en temps absolu tout en ayant
une accélération plus faible.

Un plateau doit être confronté aux compteurs : E/S, parties séquentielles,
synchronisations ou bande passante mémoire peuvent limiter le gain. Ces temps
ne suffisent pas à identifier seuls la cause. Une efficacité supérieure à
100 % est possible, notamment avec des effets de cache, et doit être examinée.
Les caches disque ne sont pas contrôlés et aucun `fsync` n'est ajouté : les
mesures d'E/S ne représentent pas un débit physique garanti du disque.
L'expérience complète l'étape 5 du TP2 ; elle ne remplace pas son modèle
d'évolution des coûts avec la taille du maillage et la contrainte CFL.
