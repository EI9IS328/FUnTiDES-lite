#!/usr/bin/env python3
"""Estimate the raw data retained by the four TP2 campaigns, without running them."""

import json
import math
from pathlib import Path
import sys

# This estimator only reads files, including when importing the shared helpers.
sys.dont_write_bytecode = True
from estimate_storage import count_receivers, estimate, positive_integer


def read_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    params = config["parameters"]
    order = positive_integer(params["order"], "order")
    repetitions = positive_integer(config["repetitions"], "repetitions")
    cadence = positive_integer(params["snap-time-interval"], "snap-time-interval")
    if order not in (1, 2, 3) or repetitions < 2:
        raise ValueError(f"{path.name} : order doit être 1, 2 ou 3 et repetitions >= 2")
    for key in ("dt", "timemax"):
        if type(params[key]) not in (int, float) or not math.isfinite(params[key]) or params[key] <= 0:
            raise ValueError(f"{path.name} : {key} doit être positif et fini")
    steps = int(params["timemax"] / params["dt"])
    if steps < 1:
        raise ValueError(f"{path.name} : au moins un pas de temps requis")
    return config, order, repetitions, cadence, steps


def estimate_all(folder):
    rows = []
    path = folder / "compare_config.json"
    read_config(path)
    _, _, sizes = estimate(path, ["S", "M", "L"], ["snapshots", "sismograms"])
    rows.append({"config": path.name,
                 "snapshots_bytes": sum(row["snapshots_bytes"] for row in sizes),
                 "sismograms_bytes": sum(row["sismograms_bytes"] for row in sizes)})

    for name in ("strong_scaling_config.json", "weak_scaling_config.json"):
        path = folder / name
        config, order, repetitions, cadence, steps = read_config(path)
        if name == "strong_scaling_config.json":
            size = positive_integer(config["size"], "size")
            threads = config["threads"]
            if not isinstance(threads, list) or len(threads) < 2:
                raise ValueError("threads doit contenir au moins deux valeurs")
            for p in threads:
                positive_integer(p, "threads")
            if threads != sorted(set(threads)) or threads[0] != 1:
                raise ValueError("Threads distincts croissants, avec référence à 1")
            node_sum = (order * size + 1) ** 3 * len(threads)
            cases = len(threads)
        else:
            cases_list = config["cases"]
            if not isinstance(cases_list, list) or len(cases_list) < 2:
                raise ValueError("cases doit contenir au moins deux configurations")
            node_sum = 0
            base_elements = None
            previous = 0
            for case in cases_list:
                p = positive_integer(case["threads"], "threads")
                if p <= previous:
                    raise ValueError("Threads weak scaling distincts et croissants requis")
                previous = p
                dimensions = [positive_integer(case[axis], axis) for axis in ("ex", "ey", "ez")]
                elements = math.prod(dimensions)
                if base_elements is None:
                    if p != 1:
                        raise ValueError("La référence weak scaling doit utiliser un thread")
                    base_elements = elements
                if elements != p * base_elements:
                    raise ValueError("Weak scaling : éléments par thread non constants")
                node_sum += math.prod(order * n + 1 for n in dimensions)
            cases = len(cases_list)
        receivers = count_receivers((folder / config["receivers"]).resolve())
        # One ad-hoc snapshot campaign and one ad-hoc signal campaign per case.
        # Their in-situ counterparts produce no raw binary output.
        fields = (steps + cadence - 1) // cadence
        rows.append({"config": name,
                     "snapshots_bytes": 20 * node_sum * fields * repetitions,
                     "sismograms_bytes": cases * repetitions * (8 + 4 * steps * (receivers + 1))})

    path = folder / "cpu_gpu_config.json"
    read_config(path)
    # Both extrema paths run in-situ: JSON/CSV/figures, but no raw snapshots/signals.
    rows.append({"config": path.name, "snapshots_bytes": 0, "sismograms_bytes": 0})
    return rows


def main():
    try:
        if len(sys.argv) != 1:
            raise ValueError("Ce script s'utilise sans argument")
        rows = estimate_all(Path(__file__).resolve().parent)
        print("Stockage brut estimé à la fin des quatre campagnes TP2")
        print(f"{'Configuration':31} {'Snapshots (GB)':>15} {'Sismos (GB)':>14} {'Total brut (GB)':>17}")
        snapshots = signals = 0
        for row in rows:
            snapshots += row["snapshots_bytes"]
            signals += row["sismograms_bytes"]
            total = row["snapshots_bytes"] + row["sismograms_bytes"]
            print(f"{row['config']:31} {row['snapshots_bytes']/1e9:15.5f} "
                  f"{row['sismograms_bytes']/1e9:14.5f} {total/1e9:17.5f}")
        print(f"{'SOMME DES QUATRE CONFIGS':31} {snapshots/1e9:15.5f} "
              f"{signals/1e9:14.5f} {(snapshots+signals)/1e9:17.5f}")
        print("GB = 1 000 000 000 octets. Répétitions et sorties de précision incluses.")
        print("CPU/GPU : 0 donnée brute ; ses statistiques et figures occupent néanmoins du stockage.")
        print("Hors JSON, CSV, logs, figures, build/, espace de gestion du système de fichiers et résultats déjà présents.")
        print("Nombre de pas estimé par int(timemax/dt) ; les arrondis float32 du C++ peuvent le modifier.")
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        sys.exit(f"Erreur : {error}")


if __name__ == "__main__":
    main()
