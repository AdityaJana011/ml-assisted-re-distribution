import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.drf import (
    construct_drf_matrix, 
    construct_bremsstrahlung_kernel, 
    construct_total_response_matrix
)
from src.data_generator import generate_synthetic_dataset
from src.mlem import run_mlem_inversion
from src.train import train_deconvoluter
from src.utils import load_and_clean_digitized_data, plot_and_save_figure_1a, plot_and_save_figure_1b, plot_and_save_figure_1c

def main():
    parser = argparse.ArgumentParser(description="DeGaSum Inversion & Deconvolution Pipeline")
    parser.add_argument(
        '--step', 
        type=str, 
        default='all',
        choices=['all', 'drf', 'data_gen', 'mlem', 'train', 'plot'],
        help="Pipeline step to run: drf, data_gen, mlem, train, plot, or all"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='cnn',
        choices=['cnn'],
        help="Architecture for training: cnn only on this branch"
    )
    parser.add_argument(
        '--two-layer',
        action='store_true',
        help="Use two-layer forward model H_tot = H_d @ H_e"
    )
    parser.add_argument(
        '--refine',
        type=int,
        default=1,
        help="Channel refinement scale factor (N_fine = REFINE * N_coarse)"
    )
    
    args = parser.parse_args()
    
    n_coarse = 100
    n_fine = n_coarse * args.refine
    
    if args.step in ['all', 'drf']:
        print("\n=== Running DRF Construction ===")
        # Build physical DRF
        construct_drf_matrix(
            raw_data_dir='data/raw', 
            processed_data_dir='data/processed',
            n_channels_measured=n_coarse,
            n_channels_true=n_fine,
            resolution_scale=1.0
        )
        # Build 30% narrower DRF for deconvolution
        construct_drf_matrix(
            raw_data_dir='data/raw', 
            processed_data_dir='data/processed',
            n_channels_measured=n_coarse,
            n_channels_true=n_fine,
            resolution_scale=0.7
        )
        
        if args.two_layer:
            print("\n=== Building Two-Layer Bremsstrahlung Operator ===")
            # Build H_e
            construct_bremsstrahlung_kernel(
                n_channels_true=n_fine,
                n_channels_photon=n_fine,
                processed_data_dir='data/processed'
            )
            # Build physical H_tot = H_d @ H_e
            construct_total_response_matrix(
                h_d_path='data/processed/detector_response_matrix.npy',
                h_e_path='data/processed/bremsstrahlung_emission_kernel.npy',
                out_path='data/processed/total_response_matrix.npy'
            )
            # Build deconvolution (narrowed) H_tot_narrow = H_d_narrow @ H_e
            construct_total_response_matrix(
                h_d_path='data/processed/detector_response_matrix_narrow.npy',
                h_e_path='data/processed/bremsstrahlung_emission_kernel.npy',
                out_path='data/processed/total_response_matrix_narrow.npy'
            )
        
    if args.step in ['all', 'data_gen']:
        print("\n=== Running Synthetic Data Generation ===")
        generate_synthetic_dataset(
            h_matrix_path='data/processed/detector_response_matrix.npy',
            out_npz_path='data/processed/degas_ml_training_data.npz',
            plot_sample=True,
            plot_out_path='outputs/diagnostics/Sample_pair.png',
            two_layer=args.two_layer
        )
        
    if args.step in ['all', 'train']:
        print(f"\n=== Running {args.model.upper()} Model Training ===")
        # On this branch, we train the 1D CNN model. We check if two_layer dataset is available.
        # Physics-informed deconvolution loss is not implemented on master, so training is supervised.
        train_deconvoluter(
            dataset_path='data/processed/degas_ml_training_data.npz',
            weights_save_path=f"models/degas_{args.model}_weights.pth",
            epochs=40
        )
        
    if args.step in ['all', 'mlem']:
        print("\n=== Running Stabilized ML-EM Inversion ===")
        
        # Determine paths based on layer mode
        if args.two_layer:
            h_forward_path = 'data/processed/total_response_matrix.npy'
            h_backward_path = 'data/processed/total_response_matrix_narrow.npy'
        else:
            h_forward_path = 'data/processed/detector_response_matrix.npy'
            h_backward_path = 'data/processed/detector_response_matrix_narrow.npy'
            
        # Re-build if missing
        if not os.path.exists(h_forward_path) or not os.path.exists(h_backward_path):
            print("Response matrices missing. Triggering DRF construction...")
            # Run DRF step
            construct_drf_matrix(n_channels_measured=n_coarse, n_channels_true=n_fine, resolution_scale=1.0)
            construct_drf_matrix(n_channels_measured=n_coarse, n_channels_true=n_fine, resolution_scale=0.7)
            if args.two_layer:
                construct_bremsstrahlung_kernel(n_channels_true=n_fine, n_channels_photon=n_fine)
                construct_total_response_matrix(
                    h_d_path='data/processed/detector_response_matrix.npy',
                    h_e_path='data/processed/bremsstrahlung_emission_kernel.npy',
                    out_path='data/processed/total_response_matrix.npy'
                )
                construct_total_response_matrix(
                    h_d_path='data/processed/detector_response_matrix_narrow.npy',
                    h_e_path='data/processed/bremsstrahlung_emission_kernel.npy',
                    out_path='data/processed/total_response_matrix_narrow.npy'
                )
                
        H_forward = np.load(h_forward_path)
        H_backward = np.load(h_backward_path)
        
        epsilon_prime = np.linspace(0.01, 10.0, n_fine)
        d_epsilon_prime = epsilon_prime[1] - epsilon_prime[0]
        
        # Set up a target physical spectrum with a background and 2 peaks
        x_true = np.exp(-epsilon_prime / 3.0) * 100
        x_true += 500 * np.exp(-((epsilon_prime - 4.0) ** 2) / (2 * 0.15**2))
        x_true += 200 * np.exp(-((epsilon_prime - 7.0) ** 2) / (2 * 0.20**2))
        
        # Convolve physical target with the forward operator
        y_ideal = np.dot(H_forward, x_true) * d_epsilon_prime
        y_ideal = (y_ideal / np.sum(y_ideal)) * 10000
        
        np.random.seed(42)
        y_measured = np.random.poisson(y_ideal)
        
        # Run stabilized MLEM with the split-matrix deconvolution (Trick 3)
        x_recon, x_smoothed = run_mlem_inversion(
            y_measured=y_measured, 
            h_matrix=H_forward, 
            h_backward=H_backward, 
            num_iterations=100,
            smooth_every=5,
            init_mode='shevelev',
            final_smooth=True
        )
        
        # Area-normalize values to avoid scaling mismatches in visualization
        x_true_norm = x_true / np.sum(x_true)
        x_recon_norm = x_recon / np.sum(x_recon)
        x_smoothed_norm = x_smoothed / np.sum(x_smoothed)
        
        os.makedirs('outputs/diagnostics', exist_ok=True)
        plt.figure(figsize=(7, 4.5))
        plt.plot(epsilon_prime, x_true_norm, label="True Target (Dashed)", color="black", linestyle="--", alpha=0.7)
        plt.plot(epsilon_prime, x_recon_norm, label="MLEM (Raw)", color="red", linestyle=":", alpha=0.4)
        plt.plot(epsilon_prime, x_smoothed_norm, label="MLEM (Stabilized)", color="red", linewidth=1.8)
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Area-Normalized Yield")
        plt.title("Stabilized MLEM Reconstruction", fontsize=12, fontweight="bold")
        plt.xlim(0, 10)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig('outputs/diagnostics/degas_reconstruction_performance.png', dpi=300)
        plt.close()
        print("ML-EM deconvolution complete. Saved performance plot to outputs/diagnostics/degas_reconstruction_performance.png")
        
    if args.step in ['all', 'plot']:
        print("\n=== Running Figure Reconstruction ===")
        file_mapping = {
            'a_3MeV': 'data/raw/graph_a(3MeV).csv',
            'a_6MeV': 'data/raw/graph_a(6MeV).csv',
            'a_8MeV': 'data/raw/graph_a(8MeV).csv',
            'b_test': 'data/raw/graph_b(test_spectrum_points).csv',
            'b_recon': 'data/raw/graph_b(reconstructed_spectrum_line).csv',
            'c_test': 'data/raw/graph_c(test_spectrum_dashed_line).csv',
            'c_recon': 'data/raw/graph_c(reconstructed_spectrum_solid_line).csv'
        }
        
        data = {key: load_and_clean_digitized_data(fn) for key, fn in file_mapping.items()}
        plot_and_save_figure_1a(data, out_path='outputs/figures/recon_figure_1a.png')
        plot_and_save_figure_1b(data, out_path='outputs/figures/recon_figure_1b.png')
        plot_and_save_figure_1c(data, out_path='outputs/figures/recon_figure_1c.png')
        print("Figure reconstruction complete. Saved to outputs/figures/")

if __name__ == "__main__":
    main()
