import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


THREADS = {
    1: {"sismo_adhoc": 22.665663572959602, "sismo_insitu_cufft": 29.821133884135634},
    2: {"sismo_adhoc": 11.448844311758876, "sismo_insitu_cufft": 18.706175821833313},
    4: {"sismo_adhoc": 6.028427044395357, "sismo_insitu_cufft": 13.176377880387008},
}

SIZES = {
    "S": {"h2d": 0.02499793616430997, "d2h": 0.029751667960786658},
    "M": {"h2d": 0.03524317971038048, "d2h": 0.05456853540728213},
    "L": {"h2d": 0.0631342906197966, "d2h": 0.06222042650020978},
}

GPU = {
    "sans_stats_s": 0.090,
    "insitu_s": 0.088,
    "hostcopy_s": 0.177,
    "cpu_uvm_s": 0.570,
    "insitu_ms": 0.065,
    "hostcopy_ms": 0.187,
    "field_bytes": 2125764,
}


def repertoire_benchmark():
    for base in [Path(__file__).resolve().parent, Path.cwd().resolve()]:
        for parent in [base, *base.parents]:
            if (parent / "tp2_benchmark_config.json").is_file():
                return parent
            candidate = parent / "stats" / "tp2" / "benchmark"
            if (candidate / "tp2_benchmark_config.json").is_file():
                return candidate
    return Path(__file__).resolve().parent


def lire_csv(chemin):
    if not chemin.is_file():
        return []
    with chemin.open(newline="", encoding="utf-8-sig") as fichier:
        return list(csv.DictReader(fichier))


def lire_mesures(dossier):
    threads = {t: mesures.copy() for t, mesures in THREADS.items()}
    tailles = {s: mesures.copy() for s, mesures in SIZES.items()}
    dossier_resultats = dossier / "results_tp2_bench"
    candidats_threads = [
        dossier_resultats / "threads" / "tp2_bench_threads.csv",
        dossier_resultats / "tp2_bench_threads.csv",
    ]
    for chemin in candidats_threads:
        lignes = lire_csv(chemin)
        if lignes:
            for ligne in lignes:
                if ligne.get("size") != "S":
                    continue
                mode = ligne.get("mode")
                if mode not in {"sismo_adhoc", "sismo_insitu_cufft"}:
                    continue
                try:
                    n = int(ligne["threads"])
                    temps = float(ligne["workflow_s_mean"])
                    if n in threads and temps > 0:
                        threads[n][mode] = temps
                except (ValueError, KeyError, TypeError):
                    pass
            break
    for ligne in lire_csv(dossier_resultats / "tp2_bench_summary.csv"):
        if ligne.get("mode") != "sismo_insitu_cufft":
            continue
        taille = ligne.get("size")
        if taille not in tailles:
            continue
        for champ, colonne in (("h2d", "cufft_h2d_gbs_mean"), ("d2h", "cufft_d2h_gbs_mean")):
            try:
                valeur = float(ligne[colonne])
                if valeur > 0:
                    tailles[taille][champ] = valeur
            except (ValueError, KeyError, TypeError):
                pass
    return threads, tailles


def enregistrer(fig, dossier, nom):
    fig.savefig(dossier / (nom + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(dossier / (nom + ".svg"), bbox_inches="tight")
    plt.close(fig)


def graphique_temps(dossier, threads):
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.6), gridspec_kw={"width_ratios": [1, 1.05]})
    fig.suptitle("TP2 — Sismogrammes CPU/cuFFT et statistiques GPU in-situ", fontsize=16, weight="bold")

    valeurs_cpu = [threads[4]["sismo_adhoc"], threads[4]["sismo_insitu_cufft"]]
    libelles_cpu = ["CPU + sismos\nad-hoc", "CPU + GPU\nsismos cuFFT"]
    barres_cpu = axes[0].barh(libelles_cpu, valeurs_cpu, color=["tab:blue", "tab:orange"], height=0.52)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(valeurs_cpu) * 1.28)
    axes[0].set_title("Workflow complet — cas S, 4 threads\nPlaFRIM / NVIDIA K40", fontsize=12)
    axes[0].set_xlabel("Temps total (s)")
    axes[0].grid(axis="x", alpha=0.2)
    axes[0].set_axisbelow(True)
    for barre, valeur in zip(barres_cpu, valeurs_cpu):
        axes[0].text(valeur + max(valeurs_cpu) * 0.025, barre.get_y() + barre.get_height() / 2,
                     f"{valeur:.3f} s", va="center", fontweight="bold")

    libelles_gpu = ["Sans statistiques", "GPU in-situ", "GPU → CPU (hostcopy)", "CPU UVM"]
    valeurs_gpu = [GPU["sans_stats_s"], GPU["insitu_s"], GPU["hostcopy_s"], GPU["cpu_uvm_s"]]
    barres_gpu = axes[1].barh(libelles_gpu, valeurs_gpu,
                             color=["0.65", "tab:green", "tab:orange", "tab:red"], height=0.62)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(valeurs_gpu) * 1.28)
    axes[1].set_title("Solveur + statistiques du champ — 40³\nRTX 4000 Ada (README branche sismos)", fontsize=12)
    axes[1].set_xlabel("Elapsed Compute Time (s)")
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].set_axisbelow(True)
    for barre, valeur in zip(barres_gpu, valeurs_gpu):
        axes[1].text(valeur + max(valeurs_gpu) * 0.025, barre.get_y() + barre.get_height() / 2,
                     f"≈ {valeur:.3f} s", va="center", fontweight="bold")

    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.22, top=0.79, wspace=0.43)
    enregistrer(fig, dossier, "tp2_comparaison_cpu_gpu_sismos")


def graphique_debit(dossier, tailles):
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.6), gridspec_kw={"width_ratios": [1.03, 1]})
    fig.suptitle("TP2 — Débits des transferts CPU/GPU et traitement GPU in-situ", fontsize=16, weight="bold")

    cas = ["S", "M", "L"]
    positions = list(range(len(cas)))
    largeur = 0.34
    h2d = [tailles[taille]["h2d"] for taille in cas]
    d2h = [tailles[taille]["d2h"] for taille in cas]
    barres_h2d = axes[0].bar([x - largeur / 2 for x in positions], h2d, largeur,
                             label="CPU → GPU (H2D)", color="tab:blue")
    barres_d2h = axes[0].bar([x + largeur / 2 for x in positions], d2h, largeur,
                             label="GPU → CPU (D2H)", color="tab:orange")
    axes[0].set_xticks(positions, cas)
    axes[0].set_ylim(0, max(h2d + d2h) * 1.30)
    axes[0].set_xlabel("Taille du cas")
    axes[0].set_ylabel("Débit effectif (GB/s)")
    axes[0].set_title("cudaMemcpy du workflow cuFFT\nPlaFRIM / NVIDIA K40", fontsize=12)
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].set_axisbelow(True)
    for barre in [*barres_h2d, *barres_d2h]:
        axes[0].text(barre.get_x() + barre.get_width() / 2, barre.get_height() + 0.001,
                     f"{barre.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    octets = GPU["field_bytes"]
    debit_hostcopy = octets / (GPU["hostcopy_ms"] * 1e-3) / 1e9
    debit_gpu = octets / (GPU["insitu_ms"] * 1e-3) / 1e9
    noms = ["GPU → CPU\n(hostcopy / deep_copy)", "GPU seul\n"]
    valeurs = [debit_hostcopy, debit_gpu]
    barres = axes[1].barh(noms, valeurs, color=["tab:orange", "tab:green"], height=0.56)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(valeurs) * 1.28)
    axes[1].set_xlabel("GB/s (10⁹ octets/s)")
    axes[1].set_title("Statistiques de pression, 40³\nRTX 4000 Ada (README branche sismos)", fontsize=12)
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].set_axisbelow(True)
    for barre, valeur in zip(barres, valeurs):
        axes[1].text(valeur + max(valeurs) * 0.025, barre.get_y() + barre.get_height() / 2,
                     f"{valeur:.2f} GB/s", va="center", fontweight="bold")

    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.24, top=0.79, wspace=0.32)
    fig.text(0.5, 0.125,
             "GPU → CPU : 2 125 764 octets / 0,187 ms ; GPU seul : "
             "2 125 764 octets / 0,065 ms.", ha="center", fontsize=9)
    enregistrer(fig, dossier, "tp2_debit_cudamemcpy_vs_gpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    benchmark = repertoire_benchmark()
    sortie = args.out_dir.expanduser().resolve() if args.out_dir else benchmark / "graphes_gpu_ajoutes"
    sortie.mkdir(parents=True, exist_ok=True)
    threads, tailles = lire_mesures(benchmark)
    graphique_temps(sortie, threads)
    graphique_debit(sortie, tailles)
    print(f"Deux graphiques générés dans : {sortie}")
    print("Aucun benchmark ni exécutable de simulation lancé.")


if __name__ == "__main__":
    main()