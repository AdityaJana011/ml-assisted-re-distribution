# AGENTS.md - AI Coding Agent Instructions

## Project Purpose
DeGaSum (Deconvolution of Gamma-Ray Spectra) reconstructs true gamma-ray energy spectra from blurred, noisy detector measurements in thermonuclear fusion plasmas. The system solves the inverse problem of deconvolving detector measurements to recover the original incident gamma-ray spectra for fast ion and runaway electron diagnostics in tokamak devices like ITER.

## Repository Architecture
- **main.py**: Central CLI orchestrator for pipeline execution
- **src/**: Core implementation modules
- **data/raw/**: Input data including mandatory `response_matrix.csv`
- **data/processed/**: Generated matrices and training data
- **models/**: Trained model weights
- **outputs/**: Diagnostic plots and validation results
- **v2_real_H_benchmark/**: Production pipeline (aditya branch) - INTENDED PRODUCTION DIRECTION

## Important Entry Points
```bash
# Run pipeline steps
python main.py --step all          # Full pipeline
python main.py --step drf          # DRF construction only
python main.py --step data_gen     # Data generation only
python main.py --step train        # Model training only
python main.py --step mlem         # Classical MLEM only
python main.py --step plot         # Paper figure reproduction only
```

## Production Branch Philosophy
- **master**: Treated as "Physics-Compliant" integration branch ONLY
- **v2_real_H_benchmark** (aditya branch): INTENDED production direction
- **trial/-\*** branches: Experimental/research features ONLY
- **DO NOT** assume current master contents represent the best scientific implementation
- **DO NOT** merge experimental features into master without validation

## PRODUCTION DATA RESOLUTION: 6100×599
- **MANDATORY**: Production resolution is 6100 measured channels × 599 true energy channels
- **FORBIDDEN**: Do NOT downsample to 100×100 for production work
- **FORBIDDEN**: Do NOT treat 100×100 as equivalent to production resolution
- Real ADITYA-U LaBr3 data aligns to ~1 keV bins (~6100 channels)
- The 100×100 representation is a toy/experimental concept only

## Classical MLEM as Current Baseline
- **STATUS**: Classical MLEM is the production/reference baseline
- **PREFERRED**: Classical MLEM WITHOUT the distorting narrowed-DRF trick
- **VALIDATION**: Has demonstrated strongest results for real pre-computed detector data
- **REQUIREMENT**: Any new approach must match or exceed this baseline
- **DO NOT** describe Transformer/PINN as production replacements

## Production Performance Targets
- **Reduced χ²**: ~1.0
- **Peak-center accuracy**: ≤10 keV relative to chemical certificate values
- **Runtime**: Evaluated against inter-shot tokamak delays (minutes-level)
- **Tolerance**: 10 keV bounded by E_true 10 keV bin width

## Mandatory Dependencies
- **response_matrix.csv** (19MB): MANDATORY for full-resolution physical reconstruction
- **Location**: `data/raw/response_matrix.csv`
- **Purpose**: Physical forward operator H for decoding measured detector counts
- **Format**: 6100×599 matrix (Emeas_center_keV + 599 Etrue columns)
- **Loading**: Code expects this file for supervisor matrix functionality

## Physics Constraints (CRITICAL)
The project MUST preserve physical consistency:
- **Non-negativity**: Reconstructed spectra/counts must be ≥0
- **Probability conservation**: Column normalization in response matrices
- **Physical forward projection**: H·x_pred ≈ y_obs must hold
- **Detector response consistency**: Agreement with H matrix
- **No unphysical ringing**: Avoid negative artifacts
- **Calibration peak agreement**: Match known calibration peaks
- **Reduced χ²**: Must be appropriate against physical response model

**For PINN/physics-informed approaches**: Physics MUST be encoded into training objective:
- Incorporate forward model: H · x_pred ≈ y_obs
- Include non-negativity constraints
- Include appropriate regularization
- **DO NOT** remove physics constraints to improve ML metrics

## Validation Requirements Before Master Integration
An approach can ONLY be considered for master integration when:
1. It successfully reconstructs real 6100×599 physical calibration spectra (Eu-152, Co-60)
2. Accuracy matches or exceeds established classical MLEM baseline
3. Satisfies all physics constraints listed above
4. Meets production performance targets
5. Has been validated against real detector data (not synthetic)

## Rules for Working with Experimental Branches
- **trial/-1d-cnn-tricks**: Classical MLEM improvements, high-resolution matrices
- **trial/-PINN**: Physics-informed CNN research
- **trial/-transformer**: Transformer architecture + attention research
- **Keep experimental work on branches**: Do not merge to master without validation
- **Document experimental status**: Clearly mark research vs production code
- **Maintain baseline**: Always compare against classical MLEM
- **Physics-first**: Never sacrifice physics for ML performance

## Important Commands (Verified from Repository)
```bash
# Generate full-resolution DRF
python main.py --step drf

# Generate training data with two-layer model
python main.py --step data_gen --two-layer

# Train specific model architecture
python main.py --step train --model cnn
python main.py --step train --model transformer

# Run classical MLEM
python main.py --step mlem

# Reproduce paper validation figures
python main.py --step plot
```

## Important Files/Directories
- **data/raw/response_matrix.csv**: MANDATORY 6100×599 supervisor matrix
- **data/raw/graph_*.csv**: Digitized DRF curves (3, 6, 8 MeV) and reference plots
- **src/drf.py**: Response matrix construction (includes supervisor loading)
- **src/mlem.py**: Classical MLEM implementation
- **src/model.py**: CNN and Transformer architectures
- **src/train.py**: Training pipeline with physics-informed loss
- **src/losses.py**: Physics-informed loss function (experimental branches)
- **v2_real_H_benchmark/**: Production validation pipeline (aditya branch)

## Known Dangerous Assumptions
- **100×100 is production**: FALSE - 100×100 is toy/experimental only
- **Current master is best implementation**: FALSE - master is integration branch only
- **ML beats classical automatically**: FALSE - must validate against MLEM baseline
- **Supervisor CSV is optional**: FALSE - mandatory for production resolution
- **Physics constraints can be relaxed**: FALSE - physics is non-negotiable
- **Synthetic data validates production**: FALSE - must test on real calibration spectra

## Instructions: Inspect Before Replacing
- **ALWAYS** inspect existing MLEM implementation before modifying
- **ALWAYS** understand response matrix structure before changing physics
- **ALWAYS** verify data resolution before processing
- **ALWAYS** check physics constraints before removing them
- **NEVER** replace classical MLEM without benchmarking comparison
- **NEVER** modify scientific code without understanding the forward model

## Instructions: Data Resolution
- **NEVER** silently downsample physical data from 6100×599 to 100×100
- **ALWAYS** use full 6100×599 resolution for production work
- **ALWAYS** preserve energy grid information (emeas_grid_keV.npy, etrue_grid_keV.npy)
- **NEVER** assume 100×100 results generalize to full resolution
- **ALWAYS** validate that resolution matches experimental requirements

## Instructions: Physics Constraints
- **NEVER** remove physics constraints for convenience
- **ALWAYS** maintain non-negativity in reconstructions
- **ALWAYS** preserve probability conservation in response matrices
- **NEVER** disable physics-informed loss components for better ML metrics
- **ALWAYS** verify forward model consistency: H·x ≈ y
- **ALWAYS** check reduced χ² against physical response model

## Instructions: Production vs Experimental Code
- **Master branch**: Physics-compliant integration ONLY
- **Experimental branches**: Research features separate from production
- **CLEARLY mark** experimental status in code comments
- **DO NOT** claim experimental approaches are production-ready
- **VALIDATE** against classical MLEM before any production claims
- **MAINTAIN** baseline classical implementation for comparison

## Instructions: Reproducibility
- **ALWAYS** preserve scientific benchmark reproducibility
- **NEVER** modify classical MLEM without documenting impact on benchmarks
- **ALWAYS** keep validation plots and metrics
- **MAINTAIN** traceability between code changes and scientific results
- **DOCUMENT** any changes to physics parameters or constraints

## Instructions: Comparison Requirements
- **ALWAYS** compare new approaches against established MLEM baseline
- **USE**: Same datasets, same metrics, same validation criteria
- **REPORT**: Both ML metrics AND physics validation metrics
- **NEVER** claim improvement without physics validation
- **BENCHMARK**: on real calibration data, not just synthetic
- **DOCUMENT**: comparison methodology clearly

## Uncertainty Requirements
- **IMPORTANT**: Uncertainty estimation is required for eventual ML diagnostic use
- **Classical MLEM**: Document uncertainty/covariance methodology if present
- **ML approaches**: Investigate Bayesian dropout, ensembles, or defensible methods
- **DO NOT** claim uncertainty method is production-ready without validation
- **DIAGNOSTIC results without uncertainty are insufficient for physics use**

## Transformer/PINN Research Status
- **STATUS**: Research directions, NOT production replacements
- **PURPOSE**: Determine if learned methods can outperform classical baseline
- **ATTENTION visualization**: Research/explainability, NOT production requirement
- **EXPLAINABILITY**: Helps physicists understand and validate learned reconstructions
- **CRITERIA**: Must demonstrate physical validity against classical baseline
- **DO NOT** promote to production based on ML loss alone

## Deployment Context
- **PRIMARY**: Post-discharge/inter-shot diagnostic analysis
- **TIMING**: Processing occurs over inter-shot delays (minutes, not microseconds)
- **OPTIMIZATION**: Do NOT optimize for microsecond inference unless explicitly directed
- **FOCUS**: Physical accuracy and validation over raw speed

## Current Verified Repository State
- **Branches**: master, trial/-1d-cnn-tricks, trial/-PINN, trial/-transformer, origin/aditya
- **Supervisor matrix**: `data/raw/response_matrix.csv` (6100×599) - VERIFIED
- **Production pipeline**: `v2_real_H_benchmark/` (aditya branch) - VERIFIED
- **Calibration data**: Eu-152, Co-60 benchmarks exist in v2_real_H_benchmark
- **No AGENTS.md existed**: This is the initial agent documentation
- **No skills mechanism**: No reusable agent workflows currently established

## Critical Reminders
1. **6100×599 is production resolution** - never downsample for convenience
2. **Classical MLEM is baseline** - validate everything against it
3. **Physics constraints are mandatory** - never remove them
4. **response_matrix.csv is required** - production cannot work without it
5. **Master is integration branch** - not necessarily best implementation
6. **Experimental branches require validation** - before any master integration
7. **Real calibration data validation** - synthetic is insufficient for production claims