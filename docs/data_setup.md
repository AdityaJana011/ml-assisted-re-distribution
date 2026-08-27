# Data Setup

Large data files are intentionally not tracked in git.

To run the current real-response validation pipeline, place the supplied response matrix at:

```text
data/response_matrix.csv
```

Expected matrix convention:

- First column: measured-energy bin centers in keV.
- Remaining headers: true-energy labels like `Etrue_20p000000_keV`.
- Matrix shape after loading: `6100 x 599`.
- Measured-energy grid: `0.5` to `6099.5 keV`.
- True-energy grid: `20` to `6000 keV`.

After placing the file, run:

```bash
python scripts/validate_stage2_real_response.py
```

This writes validation metrics and figures under `results/stage2_real_response/`.
The `results/` directory is generated output and is ignored by git.

