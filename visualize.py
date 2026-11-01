#!/usr/bin/env python3
"""Plot SEM proxy CSV sismos and horizontal binary snapshot slices."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_sismos(csv_path, output):
    with csv_path.open(newline="") as source:
        names = next(csv.reader(source))
    values = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    if names[0] != "time" or len(names) < 2 or values.shape[1] != len(names):
        raise ValueError("Expected: time,receiver_0,receiver_1,...")

    fig, ax = plt.subplots()
    for column, name in enumerate(names[1:], start=1):
        ax.plot(values[:, 0], values[:, column], label=name)
    ax.set(xlabel="Temps (s)", ylabel="Pression", title="Sismogrammes")
    ax.grid(True)
    ax.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(output)


def load_snapshot(path):
    # No header: int32 node, then float32 x, y, z, pressure (20 bytes/node).
    # '=' means native byte order, matching the C++ writer; no padding.
    node_type = np.dtype([("node", "=i4"), ("values", "=f4", (4,))])
    size = path.stat().st_size
    if size == 0 or size % node_type.itemsize != 0:
        raise ValueError(f"Expected nonempty binary snapshot, 20 bytes/node: {path}")
    return np.fromfile(path, dtype=node_type)["values"]


def edges(centers):
    midpoints = (centers[:-1] + centers[1:]) / 2
    return np.r_[centers[0], midpoints, centers[-1]]


def plot_snapshots(paths, requested_z, out_dir):
    first = load_snapshot(paths[0])
    levels = np.unique(first[:, 2])
    middle_z = (levels[0] + levels[-1]) / 2
    z = levels[np.argmin(abs(levels - (requested_z if requested_z is not None else middle_z)))]

    planes = []
    for index, path in enumerate(paths):
        data = first if index == 0 else load_snapshot(path)
        points = data[data[:, 2] == z]
        if not len(points):
            raise ValueError(f"No nodes at z={z:g} in {path}")
        x, x_index = np.unique(points[:, 0], return_inverse=True)
        y, y_index = np.unique(points[:, 1], return_inverse=True)
        if len(x) < 2 or len(y) < 2:
            raise ValueError(f"At least two x and y coordinates are needed in {path}")
        pressure = np.full((len(y), len(x)), np.nan)
        pressure[y_index, x_index] = points[:, 3]
        planes.append((path, x, y, pressure))

    scale = max(np.nanmax(np.abs(pressure)) for _, _, _, pressure in planes)
    if scale == 0:
        scale = 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, x, y, pressure in planes:
        fig, ax = plt.subplots()
        image = ax.pcolormesh(edges(x), edges(y), pressure,
                              cmap="RdBu_r", vmin=-scale, vmax=scale)
        ax.set(xlabel="x", ylabel="y", title=f"{path.name} — z = {z:g}")
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, label="Pression")
        output = out_dir / f"{path.stem}.svg"
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sismos = commands.add_parser("sismos")
    sismos.add_argument("csv", type=Path)
    sismos.add_argument("--output", type=Path, default=Path("sismos.svg"))
    snapshots = commands.add_parser("snapshots")
    snapshots.add_argument(
        "directory", type=Path, help="directory containing snapshot_<step>.bin files"
    )
    snapshots.add_argument("--z", type=float, help="nearest available z plane")
    snapshots.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    if args.command == "sismos":
        plot_sismos(args.csv, args.output)
    else:
        if not args.directory.is_dir():
            parser.error(f"Snapshot directory does not exist: {args.directory}")
        paths = sorted(
            (path for path in args.directory.glob("snapshot_*.bin")
             if path.is_file() and path.stem.split("_")[-1].isdigit()),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        if not paths:
            parser.error(f"No snapshot_<step>.bin files found in {args.directory}")
        plot_snapshots(paths, args.z, args.out_dir)


if __name__ == "__main__":
    main()
