import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.drf import construct_drf_matrix
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
        choices=['cnn', 'transformer'],
        help="Architecture for training: cnn or transformer"
    )
    
    args = parser.parse_args()
    
    if args.step in ['all', 'drf']:
        print("\n=== Running DRF Construction ===")
        construct_drf_matrix(raw_data_dir='data/raw', processed_data_dir='data/processed')
        
    if args.step in ['all', 'data_gen']:
        print("\n=== Running Synthetic Data Generation ===")
        generate_synthetic_dataset(
            h_matrix_path='data/processed/detector_response_matrix.npy',
            out_npz_path='data/processed/degas_ml_training_data.npz',
            plot_sample=True,
            plot_out_path='outputs/diagnostics/Sample_pair.png'
        )
        
    if args.step in ['all', 'train']:
        print(f"\n=== Running {args.model.upper()} Model Training ===")
        train_deconvoluter(
            dataset_path='data/processed/degas_ml_training_data.npz',
            model_type=args.model,
            epochs=40
        )
        
    if args.step in ['all', 'mlem']:
        print("\n=== Running Stabilized ML-EM Inversion ===")
        h_matrix_path = 'data/processed/detector_response_matrix.npy'
        if not os.path.exists(h_matrix_path):
            construct_drf_matrix(raw_data_dir='data/raw', processed_data_dir='data/processed')
            
        h_matrix = np.load(h_matrix_path)
        n_channels = h_matrix.shape[1]
        epsilon_prime = np.linspace(0.01, 10.0, n_channels)
        d_epsilon_prime = epsilon_prime[1] - epsilon_prime[0]
        
        x_true = np.exp(-epsilon_prime / 3.0) * 100
        x_true += 500 * np.exp(-((epsilon_prime - 4.0) ** 2) / (2 * 0.15**2))
        x_true += 200 * np.exp(-((epsilon_prime - 7.0) ** 2) / (2 * 0.20**2))
        
        y_ideal = np.dot(h_matrix, x_true) * d_epsilon_prime
        y_ideal = (y_ideal / np.sum(y_ideal)) * 10000
        
        np.random.seed(42)
        y_measured = np.random.poisson(y_ideal)
        
        x_recon, x_smoothed = run_mlem_inversion(y_measured, h_matrix, num_iterations=50)
        
        os.makedirs('outputs/diagnostics', exist_ok=True)
        plt.figure(figsize=(7, 4.5))
        plt.plot(epsilon_prime, x_true, label="True Initial Spectrum $x(\\epsilon')$ (Dashed)", color="black", linestyle="--", alpha=0.7)
        plt.plot(epsilon_prime, x_recon, label="ML-EM (Raw)", color="red", linestyle=":", alpha=0.4)
        plt.plot(epsilon_prime, x_smoothed, label="ML-EM (Smoothed)", color="red", linewidth=1.8)
        plt.xlabel("Energy $\\epsilon'$ (MeV)")
        plt.ylabel("N, counts per channel")
        plt.title("ML-EM Reconstruction", fontsize=12, fontweight="bold")
        plt.xlim(0, 10)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig('outputs/diagnostics/degas_reconstruction_performance.png', dpi=300)
        plt.close()
        print("ML-EM deconvolution complete. Saved to outputs/diagnostics/degas_reconstruction_performance.png")
        
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