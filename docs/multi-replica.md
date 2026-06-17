# Multi-Replica Analysis

PepIntProt supports analyzing 1–10 independent MD replicas with automatic cross-replica statistics.

---

## Overview

Multiple replicas provide statistical confidence in your results:

- **Independent sampling** of conformational space
- **Reproducibility** assessment across runs
- **Mean ± standard deviation** for all quantitative metrics

---

## Setup

1. Set **Number of replicas** in the sidebar (1–10)
2. Upload **one topology PDB** (shared across all replicas)
3. Upload **one trajectory per replica** (prompted sequentially)

---

## Per-Replica Analysis

Each replica is analyzed independently with:

- Its own set of plots and CSV files
- Its own AI-generated report
- Its own downloadable ZIP archive

Results are stored in separate directories: `replica_1/`, `replica_2/`, etc.

---

## Combined Analysis

After all replicas complete, PepIntProt automatically generates combined statistics:

### Metrics

For each time-series metric (RMSD, Rg, Distance, etc.):

- **Mean** across replicas at each time point
- **Standard deviation** (± std envelope)
- Individual replica traces shown as thin lines

### Plots

Combined plots show:

- Thin colored lines for individual replicas
- Bold line for the mean
- Shaded envelope for ± 1 standard deviation

### Combined AI Report

A separate AI report analyzes the cross-replica data, discussing:

- Overall reproducibility
- Inter-replica variability
- Statistical significance of observations
- Convergence assessment

---

## Output Structure

```
PepIntProt_results/
├── replica_1/
│   ├── rmsd_data.csv
│   ├── rmsf_combined.png
│   ├── AI_Report.pdf
│   └── ...
├── replica_2/
│   └── ...
├── combined/
│   ├── combined_rmsd.png
│   ├── combined_rmsf.png
│   ├── Combined_Report.pdf
│   └── ...
└── master.zip
```

---

## Best Practices

!!! tip "Recommendations"

    - Use at least 3 replicas for meaningful statistics
    - Ensure replicas start from different initial velocities
    - Same topology PDB must work for all replicas
    - Consider using the same simulation length for cleaner comparisons

!!! note "Trajectory Length"
    Replicas with different frame counts are supported — PepIntProt
    handles interpolation for cross-replica alignment.
