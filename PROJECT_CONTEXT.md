# PROJECT_CONTEXT.md - DeGaSum Technical Overview

## Project Purpose
DeGaSum (Deconvolution of Gamma-Ray Spectra) is a computational physics system for reconstructing true gamma-ray energy spectra from blurred, noisy detector measurements in thermonuclear fusion plasmas. The project implements both classical iterative reconstruction (Maximum Likelihood Expectation Maximization) and deep learning approaches (CNNs, Transformers) to solve the inverse problem of deconvolving detector measurements for fast ion and runaway electron diagnostics in tokamak devices like ITER.

**FACT**: Project implements gamma-ray spectrum deconvolution using classical ML-EM and deep learning approaches.

## Scientific Problem
Gamma-ray detectors in fusion plasmas record blurred and noise-contaminated energy spectra due to the detector's Response Function (DRF). This forward process is modeled by the Fredholm integral equation of the first kind:

```
y(ε) = ∫₀^∞ x(ε')h(ε, ε')dε' + n(ε)
```

Where:
- **ε**: Measured energy deposited inside the detector (channels)
- **ε'**: Initial true energy of incoming gamma quanta from the plasma
- **x(ε')**: True initial spectrum of gamma quanta (target for deconvolution)
- **h(ε, ε')**: Detector Response Function (DRF) matrix
- **n(ε)**: Statistical Poisson measurement noise
- **y(ε)**: Final noisy spectrum measured by the detector (blurred input)

**FACT**: Inverse problem formulation based on Fredholm integral equation of first kind.

## Inverse Problem Formulation
The deconvolution problem is mathematically ill-posed: small changes in measured data y can cause large changes in reconstructed x. The project addresses this through:

1. **Classical approach**: Maximum Likelihood Expectation Maximization (ML-EM) with stabilization
2. **Deep learning approach**: Neural networks (CNN, Transformer) trained on synthetic data
3. **Physics-informed approach**: Combining ML with physical constraints via loss functions

**FACT**: Problem is ill-posed, requiring regularization through classical or ML methods.

## Meaning of Measured y
- **y**: Detector-measured spectrum (blurred, noisy)
- **Shape**: [6100] for production (measured energy channels)
- **Units**: Counts per channel (Poisson statistics)
- **Content**: Convolution of true spectrum with detector response + Poisson noise
- **Entry point**: Raw detector measurements from ADITYA-U LaBr3 detector

**FACT**: y is measured detector output, typically 6100 channels for production.

## Meaning of Reconstructed x
- **x**: True incident gamma-ray spectrum (deconvolved)
- **Shape**: [599] for production (true energy channels)
- **Units**: Counts per channel (non-negative)
- **Content**: Original plasma gamma-ray emission spectrum
- **Goal**: Recover x from measured y using knowledge of detector response H

**FACT**: x is the true spectrum to be recovered, typically 599 channels for production.

## Detector Response Matrix H
- **Physical meaning**: H[i,j] = probability that a gamma-ray with true energy ε'_j is measured in detector channel ε_i
- **Properties**: Column-normalized (each column sums to 1.0 for probability conservation)
- **Production shape**: 6100×599 (measured channels × true energy channels)
- **Energy ranges**: Emeas: 0.5-6099.5 keV, Etrue: 20-6000 keV
- **Physical basis**: Monte Carlo simulations of detector physics (Compton scattering, photoelectric effect)

**FACT**: H encodes detector physics, 6100×599 for production, column-normalized.

## Production Data Resolution
- **Production resolution**: 6100×599 (6100 measured channels, 599 true energy channels)
- **Energy grid**: ~1 keV bin width for measured data
- **Total range**: 0.5-6099.5 keV measured, 20-6000 keV true
- **Rationale**: Matches real ADITYA-U LaBr3 detector resolution
- **PROJECT DECISION**: 100×100 representation is deprecated for production

**FACT**: Production uses 6100×599 resolution based on real detector characteristics.

## Real Calibration Data
- **Eu-152**: Europium-152 calibration source (multiple gamma lines)
- **Co-60**: Cobalt-60 calibration source (two prominent gamma lines)
- **Ba-133**: Barium-133 calibration source (multiple gamma lines)
- **Purpose**: Validate deconvolution accuracy against known physical peaks
- **Requirements**: Peak-center accuracy ≤10 keV relative to chemical certificate values

**FACT**: Real calibration sources (Eu-152, Co-60, Ba-133) used for validation.

## Current MLEM Baseline
- **Status**: Classical MLEM is current production/reference baseline
- **Preferred implementation**: WITHOUT distorting narrowed-DRF trick
- **Validation**: Demonstrated strongest results for real pre-computed detector data
- **Algorithm**: Richardson-Lucy iteration with multiplicative updates
- **Stabilization**: Non-negativity, periodic smoothing, semi-convergence tracking

**PROJECT DECISION**: Classical MLEM without narrowed-DRF trick is the preferred baseline.

## ML Research Directions
### 1D CNN Approach
- **Architecture**: 6-layer 1D CNN with batch normalization
- **Training**: Scale-invariant area-normalized spectra
- **Loss**: Pure MSE (master branch) or physics-informed (experimental branches)
- **Status**: Experimental, requires validation against MLEM baseline

### Transformer Approach
- **Architecture**: Bidirectional transformer encoder with positional encoding
- **Features**: Multi-head attention, Softplus output for positivity
- **Visualization**: Attention head heatmaps for explainability
- **Status**: Research/experimental, not production-ready

### Physics-Informed Neural Networks (PINN)
- **Concept**: Encode physical constraints into training objective
- **Loss components**: Supervised MSE + physics residual + smoothness + positivity
- **Forward model**: H · x_pred ≈ y_obs enforced during training
- **Status**: Research direction, validation against classical baseline required

**FACT**: Three ML research directions exist: CNN, Transformer, PINN.

## End-to-End Architecture
```
1. Data Entry:
   - Raw detector measurements (y)
   - Response matrix H (response_matrix.csv)
   - Calibration source data (Eu-152, Co-60)

2. Forward Model (Physics):
   - H: 6100×599 detector response matrix
   - y_ideal = H @ x_true (forward convolution)
   - y_measured = Poisson(y_ideal) (add noise)

3. Deconvolution Methods:
   - Classical: MLEM iteration on y_measured using H
   - ML: Neural network trained to map y → x
   - PINN: ML with physics constraints

4. Validation:
   - Peak-center accuracy vs calibration certificates
   - Reduced χ² against physical response model
   - Physics constraint satisfaction

5. Output:
   - Reconstructed spectrum x
   - Uncertainty estimates (when available)
   - Diagnostic plots and metrics
```

**FACT**: Architecture follows forward model → deconvolution → validation pipeline.

## Data Flow
```
Physical Calibration Data (Eu-152, Co-60)
    ↓
Detector Measurements (y: 6100 channels)
    ↓
Response Matrix H (6100×599 from response_matrix.csv)
    ↓
Deconvolution (MLEM or ML)
    ↓
Reconstructed Spectrum (x: 599 channels)
    ↓
Validation (peak accuracy, χ², physics constraints)
```

**FACT**: Data flows from calibration measurements through response matrix to reconstruction.

## Important Datasets
### Input Data
- **data/raw/response_matrix.csv**: 6100×599 supervisor matrix (MANDATORY)
- **data/raw/graph_a(3MeV).csv**: Digitized 3 MeV DRF curve
- **data/raw/graph_a(6MeV).csv**: Digitized 6 MeV DRF curve
- **data/raw/graph_a(8MeV).csv**: Digitized 8 MeV DRF curve
- **data/raw/graph_b*.csv**: Reference validation plots
- **data/raw/graph_c*.csv**: Reference validation plots

### Generated Data
- **data/processed/detector_response_matrix.npy**: Processed H matrix
- **data/processed/emeas_grid_keV.npy**: Measured energy grid (6100 channels)
- **data/processed/etrue_grid_keV.npy**: True energy grid (599 channels)
- **data/processed/degas_ml_training_data.npz**: Synthetic training data (experimental)

**FACT**: Mandatory input is response_matrix.csv; other files support validation and training.

## Important Outputs/Results
### Validation Results
- **v2_real_H_benchmark/ucurve_semiconvergence.png**: MLEM semi-convergence analysis
- **v2_real_H_benchmark/drf_overlay.png**: DRF validation overlay
- **v2_real_H_benchmark/binsize_tradeoff.png**: Bin size optimization
- **v2_real_H_benchmark/firstpass_check.png**: Initial validation check

### Diagnostic Outputs
- **outputs/diagnostics/degas_reconstruction_performance.png**: MLEM reconstruction performance
- **outputs/diagnostics/Sample_pair.png**: Training data sample visualization
- **outputs/figures/recon_figure_*.png**: Paper figure reproductions

**FACT**: Validation results exist in v2_real_H_benchmark directory.

## Branch Structure
### master
- **Purpose**: Physics-compliant integration branch
- **Status**: Not necessarily best scientific implementation
- **Content**: Baseline 100×100 implementation, clean pipeline
- **Role**: Integration target for validated features

### origin/aditya
- **Purpose**: Contains production pipeline v2_real_H_benchmark
- **Content**: Real calibration data processing, validation notebooks
- **Status**: Represents intended production direction

### trial/-1d-cnn-tricks
- **Purpose**: Classical MLEM improvements and high-resolution support
- **Features**: Enhanced MLEM tricks, supervisor matrix support, uncertainty propagation
- **Status**: Experimental classical algorithm improvements

### trial/-PINN
- **Purpose**: Physics-informed CNN research
- **Features**: PINN loss function, evaluation benchmarking
- **Status**: Experimental ML approach

### trial/-transformer
- **Purpose**: Transformer architecture research
- **Features**: Transformer model, attention visualization, comprehensive benchmarking
- **Status**: Most advanced experimental ML approach

**FACT**: Five branches exist with different purposes: master (integration), aditya (production), trial branches (experimental).

## Production vs Experimental Code
### Production Code
- **Location**: v2_real_H_benchmark (aditya branch)
- **Resolution**: 6100×599 only
- **Validation**: Real calibration data (Eu-152, Co-60)
- **Baseline**: Classical MLEM without narrowed-DRF trick
- **Status**: Intended production direction

### Experimental Code
- **Location**: trial/-* branches
- **Resolution**: Mixed 100×100 and 6100×599
- **Validation**: Mixed synthetic and real data
- **Purpose**: Research and development
- **Status**: NOT production-ready without validation

**PROJECT DECISION**: v2_real_H_benchmark represents intended production direction.

## Validation Methodology
### Classical MLEM Validation
1. Forward check: H @ x_true ≈ y_measured
2. Self-consistency: MLEM on synthetic data recovers x_true
3. Real calibration: Peak accuracy vs Eu-152, Co-60 certificates
4. Physics metrics: Reduced χ² ≈ 1.0, non-negativity, smoothness

### ML Approach Validation
1. ML metrics: MSE, accuracy on test data
2. Physics metrics: Forward model consistency, reduced χ²
3. Calibration validation: Peak accuracy on real sources
4. Baseline comparison: Must match or exceed classical MLEM
5. Physics constraints: Non-negativity, probability conservation

**FACT**: Validation requires both ML metrics and physics validation.

## Current Known Limitations
### Resolution Mismatch
- **Issue**: Master branch uses 100×100, production requires 6100×599
- **Impact**: Experimental 100×100 results may not generalize to production
- **Status**: Resolution standardization needed

### ML Validation Gap
- **Issue**: ML approaches not yet validated on real calibration data
- **Impact**: Cannot claim production readiness
- **Status**: Requires Eu-152, Co-60 validation

### Uncertainty Quantification
- **Issue**: Limited uncertainty estimation for ML approaches
- **Impact**: Diagnostic results lack error bars
- **Status**: Bayesian methods, ensembles under investigation

### Performance Targets
- **Issue**: Runtime not yet optimized for inter-shot constraints
- **Impact**: May not meet minutes-level processing requirement
- **Status: Performance evaluation needed

**INFERENCE**: Resolution mismatch and ML validation gap are current limitations.

## External Dependencies
### Core Libraries
- **numpy**: Numerical operations and array manipulation
- **scipy**: Interpolation (interp1d) and signal processing
- **pandas**: CSV data handling and cleaning
- **torch**: Deep learning framework (PyTorch)
- **matplotlib**: Scientific plotting and visualization
- **scikit-learn**: Train/test splitting

### Hardware Acceleration
- **Apple Silicon GPU**: MPS backend support
- **NVIDIA CUDA**: Fallback GPU support
- **CPU**: Final fallback option

### Data Dependencies
- **response_matrix.csv**: 19MB supervisor matrix (MANDATORY)
- **Digitized curves**: From research papers (Shevelev et al. 2013)

**FACT**: Dependencies include standard scientific Python stack and mandatory CSV file.

## Expected Execution Environment
- **Primary**: Post-discharge/inter-shot diagnostic analysis
- **Timing**: Minutes-level processing (not microsecond real-time)
- **Hardware**: Workstation with GPU acceleration preferred
- **Data**: Local access to response_matrix.csv and calibration data
- **Context**: Tokamak diagnostic workflow, not plasma control

**PROJECT DECISION**: Deployment context is post-discharge analysis, not real-time control.

## Important Entry Points
### Pipeline Execution
```bash
python main.py --step all          # Full pipeline
python main.py --step drf          # DRF construction
python main.py --step data_gen     # Data generation
python main.py --step train        # Model training
python main.py --step mlem         # Classical MLEM
python main.py --step plot         # Paper figures
```

### Model Selection
```bash
python main.py --step train --model cnn
python main.py --step train --model transformer
```

### Two-Layer Model
```bash
python main.py --step all --two-layer
```

**FACT**: Entry points support stepwise execution and model selection.

## Physics Requirements Summary
1. **Non-negativity**: x ≥ 0, y ≥ 0 (counts cannot be negative)
2. **Probability conservation**: Column normalization in H (Σᵢ H[i,j] = 1.0)
3. **Forward model consistency**: H · x ≈ y must hold physically
4. **Detector response agreement**: Reconstructions must agree with known H
5. **No unphysical artifacts**: Avoid negative ringing, unphysical oscillations
6. **Calibration agreement**: Match known peak positions (Eu-152, Co-60)
7. **Statistical consistency**: Reduced χ² ≈ 1.0 against physical model

**FACT**: Seven physics requirements must be satisfied for valid reconstructions.

## Uncertainty Requirements
- **Importance**: Diagnostic results require uncertainty for physics use
- **Classical MLEM**: Monte Carlo uncertainty propagation exists in experimental branches
- **ML approaches**: Bayesian dropout, ensembles, or similar methods needed
- **Status**: Production-ready uncertainty estimation not yet implemented
- **Requirement**: Any production approach must include validated uncertainty quantification

**FACT**: Uncertainty estimation is required but not yet production-ready for ML approaches.

## Transformer/PINN Research Status
- **Classification**: Research directions, NOT production replacements
- **Purpose**: Determine if learned methods can outperform classical baseline
- **Validation criteria**: Must demonstrate physical validity against MLEM baseline
- **Attention visualization**: Research/explainability tool, not production requirement
- **Current status**: Experimental, requires real calibration data validation

**PROJECT DECISION**: ML approaches are research until validated against classical baseline.

## Current Production Status
- **Classical MLEM**: Production baseline, validated on real data
- **ML approaches**: Experimental, not production-ready
- **Resolution**: Transitioning from 100×100 (experimental) to 6100×599 (production)
- **Validation**: Classical baseline established, ML validation pending
- **Integration**: Master branch for physics-compliant features only

**INFERENCE**: Production currently relies on classical MLEM; ML approaches require validation.

## Key Technical Constraints
1. **Resolution**: Must use 6100×599 for production work
2. **Physics**: Cannot relax physics constraints for ML performance
3. **Validation**: Must test on real calibration data, not just synthetic
4. **Baseline**: Must compare against classical MLEM
5. **Data**: response_matrix.csv is mandatory for production
6. **Uncertainty**: Required for production diagnostic use

**FACT**: Six key technical constraints govern production implementation.

## File Path Conventions
- **Input data**: `data/raw/` (CSV files)
- **Processed data**: `data/processed/` (numpy arrays)
- **Source code**: `src/` (Python modules)
- **Models**: `models/` (PyTorch weights)
- **Outputs**: `outputs/` (plots and figures)
- **Documentation**: `notes/` (project documentation)
- **Production**: `v2_real_H_benchmark/` (aditya branch)

**FACT**: Standard directory structure with clear separation of concerns.

## Code Organization Principles
- **Modular design**: Separate modules for DRF, data generation, MLEM, training
- **Physics-first**: Physical constraints encoded in core algorithms
- **Reproducibility**: Deterministic results where possible (random seeds)
- **Validation**: Continuous validation against physical requirements
- **Documentation**: Clear separation of production vs experimental code

**INFERENCE**: Code organization emphasizes modularity, physics, and reproducibility.

## Remaining Uncertainties
1. **ML validation**: Extent of ML validation on real calibration data unclear
2. **Performance targets**: Specific runtime requirements for inter-shot processing
3. **Integration strategy**: Timeline for integrating validated features into master
4. **Uncertainty methods**: Which uncertainty quantification approach will be standardized
5. **Production deployment**: Specific hardware and deployment environment details

**UNCERTAIN**: ML validation extent, performance requirements, integration timeline unclear.

## Next Engineering Priorities
1. **Resolution standardization**: Migrate all production work to 6100×599
2. **ML validation**: Validate CNN/Transformer approaches on Eu-152, Co-60 data
3. **Uncertainty implementation**: Production-ready uncertainty quantification
4. **Performance optimization**: Ensure runtime meets inter-shot constraints
5. **Integration planning**: Define criteria and process for master integration

**INFERENCE**: Resolution standardization and ML validation are immediate priorities.