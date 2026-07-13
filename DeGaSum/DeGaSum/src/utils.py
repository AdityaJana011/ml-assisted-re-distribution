import os
import pandas as pd
import matplotlib.pyplot as plt

def load_and_clean_digitized_data(filename):
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None
    # header=None prevents the first data point from becoming the column name
    df = pd.read_csv(filename, header=None)
    df.columns = ['x', 'y']
    # Eliminate duplicate digitizer hits on the same x-pixel by taking the mean
    df = df.groupby('x').mean().reset_index()
    # Sorting by energy ensures lines connect in the correct sequence
    df = df.sort_values(by='x')
    return df

def plot_and_save_figure_1a(data, out_path='outputs/figures/recon_figure_1a.png', fig_size=(8, 4.5)):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=fig_size)
    
    if data.get('a_3MeV') is not None:
        ax.plot(data['a_3MeV']['x'], data['a_3MeV']['y'], label='3 MeV', color='black', linestyle='-')
    if data.get('a_6MeV') is not None:
        ax.plot(data['a_6MeV']['x'], data['a_6MeV']['y'], label='6 MeV', color='black', linestyle='--')
    if data.get('a_8MeV') is not None:
        ax.plot(data['a_8MeV']['x'], data['a_8MeV']['y'], label='8 MeV', color='black', linestyle=':')
        
    ax.set_ylabel('Counts (normalized)')
    ax.set_xlabel('Energy, MeV')
    ax.set_title('(a)', loc='left', fontsize=12, fontweight='bold')
    ax.legend(frameon=False, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(0, 10)
    ax.tick_params(direction='in', top=True, right=True, labelsize=11)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_and_save_figure_1b(data, out_path='outputs/figures/recon_figure_1b.png', fig_size=(8, 4.5)):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=fig_size)
    
    if data.get('b_test') is not None:
        # Downsample the coordinates to plot clear, distinct dots (every 12th point)
        b_test_dots = data['b_test'].iloc[::12]
        ax.plot(b_test_dots['x'], b_test_dots['y'], marker='.', linestyle='None', 
                 markersize=4, color='black', label='Test spectrum convolved with DRF')
                 
    if data.get('b_recon') is not None:
        ax.plot(data['b_recon']['x'], data['b_recon']['y'], linestyle='-', 
                 color='black', label='Reconstructed spectrum convolved with DRF')
                 
    ax.set_ylabel('N, counts per channel')
    ax.set_xlabel('Energy, MeV')
    ax.set_title('(b)', loc='left', fontsize=12, fontweight='bold')
    ax.legend(frameon=False, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(0, 10)
    ax.tick_params(direction='in', top=True, right=True, labelsize=11)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_and_save_figure_1c(data, out_path='outputs/figures/recon_figure_1c.png', fig_size=(8, 4.5)):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=fig_size)
    
    if data.get('c_test') is not None:
        ax.plot(data['c_test']['x'], data['c_test']['y'], label='Test spectrum', 
                 color='black', linestyle='--')
                 
    if data.get('c_recon') is not None:
        ax.plot(data['c_recon']['x'], data['c_recon']['y'], label='Reconstructed spectrum', 
                 color='black', linestyle='-')
                 
    ax.set_ylabel('N, counts per channel')
    ax.set_xlabel('Energy, MeV')
    ax.set_title('(c)', loc='left', fontsize=12, fontweight='bold')
    ax.legend(frameon=False, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(0, 10)
    ax.tick_params(direction='in', top=True, right=True, labelsize=11)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")
