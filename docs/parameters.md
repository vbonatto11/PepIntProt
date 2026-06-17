# Parameters Reference

Complete reference of all configurable parameters in PepIntProt.

---

## System Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| System type | Dropdown | Holo (Peptide + Protein) | Analysis mode selection |
| Ligand residue name | Text | LIG | 3-letter residue name (ligand mode only) |
| Number of replicas | Slider | 1 | Independent replicas (1–10) |
| Auto-detect chains | Checkbox | True | Automatic chain identification |
| Detection method | Radio | By chain ID | Chain detection strategy |

---

## Simulation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Simulation time (ns) | Number | 100 | Total simulation time for axis scaling |

---

## Analysis Parameters

### Contact Analysis

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Contact cutoff (Å) | 4.5 | 2.0–10.0 | Max distance for residue contact |

### Free Energy Landscape

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Temperature (K) | 300 | 200–400 | For ΔG = -kT ln(P) calculation |
| Number of bins | 100 | 20–500 | 2D histogram resolution |
| FEL extract | Peptide+Protein or Protein+Ligand | — | Which atoms for RMSD/Rg |

### eRMSF

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Skip (frames/segment) | 10 | 2–100 | Frames per eRMSF segment |
| vmin | 0 | 0–10 | Colorbar minimum (Å) |
| vmax | 5 | 1–20 | Colorbar maximum (Å) |

---

## AI Report Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| AI Provider | Dropdown | LLM provider selection |
| Model | Dropdown | Specific model within provider |
| API Key | Password | Provider API key (not needed for Databricks) |

---

## Output Files

Each analysis generates:

| File Type | Format | Description |
|-----------|--------|-------------|
| Plots | `.png` | 200 dpi publication-quality figures |
| Data | `.csv` | Raw numerical data for each analysis |
| Report | `.pdf` | AI-generated analysis report |
| Archive | `.zip` | All files packaged for download |
