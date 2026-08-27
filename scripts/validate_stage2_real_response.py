from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mlem import forward_project, normalize_sum, relative_l2, run_mlem
from src.response_matrix import (
    diagnose_operator_type,
    load_response_matrix,
    validate_response_matrix,
)
from src.synthetic import make_synthetic_distribution, simulate_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate real ADITYA-U response matrix and MLEM baseline.")
    parser.add_argument("--matrix", default="data/response_matrix.csv", help="Path to response matrix CSV.")
    parser.add_argument("--out-dir", default="results/stage2_real_response", help="Directory for metrics and figures.")
    parser.add_argument("--n-iter", type=int, default=300)
    parser.add_argument("--total-counts", type=float, default=1e5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["bulk_plus_tail", "single_exp", "delta_2mev", "two_peaks"],
        help="Synthetic distribution cases to validate.",
    )
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_response_matrix(ROOT / args.matrix)
    validation = validate_response_matrix(matrix)
    operator_rows = diagnose_operator_type(matrix)

    case_results = [
        _validate_case(
            case=case,
            matrix=matrix,
            n_iter=args.n_iter,
            total_counts=args.total_counts,
            seed=args.seed + offset,
        )
        for offset, case in enumerate(args.cases)
    ]
    primary = case_results[0]
    x_true = primary["x_true"]
    y_clean = primary["y_clean"]
    y_noisy = primary["y_noisy"]
    noiseless = primary["noiseless"]
    noisy = primary["noisy"]
    l2 = noisy.relative_l2
    chi2 = noisy.chi2
    if l2 is None:
        raise RuntimeError("Expected relative L2 history for noisy validation")

    it_l2_min = primary["noisy_l2_min_iteration"]
    it_chi2_near_1 = primary["noisy_chi2_near_1_iteration"]

    smoothing_results = []
    for smooth_every in [0, 5, 8, 10]:
        for final_smooth in [False, True]:
            result = run_mlem(
                y_noisy,
                matrix.h,
                n_iter=it_l2_min,
                smooth_every=smooth_every,
                final_smooth=final_smooth,
            )
            smoothing_results.append(
                {
                    "smooth_every": smooth_every,
                    "final_smooth": final_smooth,
                    "relative_l2": relative_l2(normalize_sum(result.x), x_true),
                }
            )
    smoothing_results.sort(key=lambda row: row["relative_l2"])

    x_best = run_mlem(
        y_noisy,
        matrix.h,
        n_iter=it_l2_min,
        smooth_every=smoothing_results[0]["smooth_every"],
        final_smooth=smoothing_results[0]["final_smooth"],
    ).x
    y_reprojected = forward_project(matrix.h, normalize_sum(x_best))
    y_reprojected = y_reprojected / y_reprojected.sum() * y_noisy.sum()
    reprojection_chi2 = float(np.sum((y_noisy - y_reprojected) ** 2 / np.clip(y_reprojected, 1.0, None)) / len(y_noisy))

    metrics = {
        "matrix": asdict(validation),
        "operator_diagnostic": [asdict(row) for row in operator_rows],
        "synthetic_case": {
            "kind": primary["case"],
            "total_counts": args.total_counts,
            "seed": args.seed,
            "true_peak_keV": float(matrix.e_true_keV[int(x_true.argmax())]),
        },
        "mlem": {
            "n_iter": args.n_iter,
            "noiseless_final_relative_l2": primary["noiseless_final_relative_l2"],
            "noiseless_recovered_peak_keV": primary["noiseless_recovered_peak_keV"],
            "noisy_l2_min_iteration": it_l2_min,
            "noisy_l2_min": primary["noisy_l2_min"],
            "noisy_chi2_near_1_iteration": it_chi2_near_1,
            "noisy_chi2_at_l2_min": float(chi2[it_l2_min - 1]),
            "noisy_chi2_near_1": float(chi2[it_chi2_near_1 - 1]),
            "best_smoothing": smoothing_results[0],
            "reprojection_chi2_reduced": reprojection_chi2,
        },
        "validation_cases": [
            {
                key: value
                for key, value in result.items()
                if key not in {"x_true", "y_clean", "y_noisy", "noiseless", "noisy"}
            }
            for result in case_results
        ],
        "smoothing_sweep": smoothing_results,
        "open_physics_note": (
            "Operator diagnostics can suggest H_tot vs H_d behavior, but final interpretation "
            "must be confirmed from the response-matrix source/model."
        ),
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _plot_forward(matrix.e_meas_keV, y_clean, y_noisy, fig_dir / "forward_poisson.png")
    _plot_ucurve(l2, chi2, it_l2_min, it_chi2_near_1, fig_dir / "ucurve_chi2.png")
    _plot_reconstruction(
        matrix.e_true_keV,
        x_true,
        normalize_sum(noiseless.x),
        normalize_sum(x_best),
        fig_dir / "reconstruction.png",
    )
    _plot_operator_columns(matrix, fig_dir / "operator_columns.png")

    print(f"Loaded H shape: {matrix.h.shape}")
    print(f"Column sums: {validation.column_sum_min:.6f} .. {validation.column_sum_max:.6f}")
    for result in case_results:
        print(
            f"Case {result['case']}: noiseless L2 {result['noiseless_final_relative_l2']:.6f}; "
            f"noisy best iter {result['noisy_l2_min_iteration']}, L2 {result['noisy_l2_min']:.6f}"
        )
    print(f"Primary chi2 closest to 1: iter {it_chi2_near_1}, chi2 {metrics['mlem']['noisy_chi2_near_1']:.6f}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved figures: {fig_dir}")


def _validate_case(
    case: str,
    matrix,
    n_iter: int,
    total_counts: float,
    seed: int,
) -> dict:
    x_true = make_synthetic_distribution(matrix.e_true_keV, case)
    y_clean, y_noisy = simulate_counts(matrix.h, x_true, total_counts, seed=seed)
    noiseless = run_mlem(y_clean, matrix.h, n_iter=n_iter, x_true=x_true)
    noisy = run_mlem(y_noisy, matrix.h, n_iter=n_iter, x_true=x_true)
    if noisy.relative_l2 is None:
        raise RuntimeError("Expected relative L2 history for noisy validation")

    noisy_l2_min_iteration = int(noisy.relative_l2.argmin()) + 1
    noisy_chi2_near_1_iteration = int(np.argmin(np.abs(noisy.chi2 - 1.0))) + 1

    return {
        "case": case,
        "seed": seed,
        "true_peak_keV": float(matrix.e_true_keV[int(x_true.argmax())]),
        "noiseless_final_relative_l2": relative_l2(normalize_sum(noiseless.x), x_true),
        "noiseless_recovered_peak_keV": float(matrix.e_true_keV[int(noiseless.x.argmax())]),
        "noisy_l2_min_iteration": noisy_l2_min_iteration,
        "noisy_l2_min": float(noisy.relative_l2[noisy_l2_min_iteration - 1]),
        "noisy_chi2_near_1_iteration": noisy_chi2_near_1_iteration,
        "noisy_chi2_near_1": float(noisy.chi2[noisy_chi2_near_1_iteration - 1]),
        "x_true": x_true,
        "y_clean": y_clean,
        "y_noisy": y_noisy,
        "noiseless": noiseless,
        "noisy": noisy,
    }


def _plot_forward(e_meas: np.ndarray, y_clean: np.ndarray, y_noisy: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(9, 4))
    plt.plot(e_meas, y_clean, lw=1, label="clean")
    plt.plot(e_meas, y_noisy, ".", ms=1.5, label="poisson")
    plt.xlabel("Measured energy (keV)")
    plt.ylabel("Counts")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_ucurve(l2: np.ndarray, chi2: np.ndarray, it_l2_min: int, it_chi2_near_1: int, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    iterations = np.arange(1, len(l2) + 1)
    ax[0].plot(iterations, l2)
    ax[0].axvline(it_l2_min, color="red", linestyle="--")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Relative L2")
    ax[1].plot(iterations, chi2)
    ax[1].axhline(1.0, color="black", linestyle=":")
    ax[1].axvline(it_chi2_near_1, color="green", linestyle="--")
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Reduced chi-square")
    ax[1].set_yscale("log")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_reconstruction(
    e_true: np.ndarray,
    x_true: np.ndarray,
    x_noiseless: np.ndarray,
    x_noisy: np.ndarray,
    out_path: Path,
) -> None:
    plt.figure(figsize=(9, 4))
    plt.plot(e_true, x_true, "k--", lw=1.5, label="truth")
    plt.plot(e_true, x_noiseless, lw=1, label="noiseless MLEM")
    plt.plot(e_true, x_noisy, lw=1, label="noisy MLEM")
    plt.xlabel("True energy grid (keV)")
    plt.ylabel("Normalized distribution")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_operator_columns(matrix, out_path: Path) -> None:
    energies = [matrix.e_true_keV[0], 1500.0, matrix.e_true_keV[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, energy in zip(axes, energies):
        j = int(np.argmin(np.abs(matrix.e_true_keV - energy)))
        ax.semilogy(matrix.e_meas_keV, matrix.h[:, j] + 1e-12)
        ax.axvline(matrix.e_true_keV[j], color="red", linestyle="--")
        ax.set_title(f"E_true={matrix.e_true_keV[j]:.0f} keV")
        ax.set_xlabel("Measured energy (keV)")
    axes[0].set_ylabel("Response")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
