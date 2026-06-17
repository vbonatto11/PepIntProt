# System Modes

PepIntProt supports three system types, each enabling different analyses and chain-detection strategies.

---

## Holo (Peptide + Protein)

For simulations containing a peptide bound to a protein receptor.

### Chain Detection

PepIntProt automatically identifies the peptide and protein chains using one of two methods:

**By Chain ID** (default):

- Reads chain identifiers from the PDB topology
- Smallest chain (fewest residues) = Peptide
- Largest chain = Protein

**By Residue ID Gap**:

- Detects discontinuities in residue numbering
- Useful when chain IDs are missing or incorrect

!!! note "Large Peptides"
    If the "peptide" chain has > 40 residues, PepIntProt labels it as
    "Protein" and calls it a protein–protein interaction instead.

### Manual Selection

Disable auto-detection and specify custom MDAnalysis selection strings:

- Peptide: e.g., `resid 1:25`
- Protein: e.g., `resid 26:350`

---

## Holo (Protein + Ligand)

For simulations containing a protein with a bound small-molecule ligand.

### How It Works

1. You specify the **ligand residue name** (e.g., `LIG`, `MOL`, `JZ4`)
2. PepIntProt separates the system:
   - **Ligand** = all atoms with that residue name
   - **Protein** = all protein atoms excluding the ligand
3. No chain detection is needed — separation is by residue name

### Atom Selections

| Component | Selection | Use |
|-----------|-----------|-----|
| Ligand (RMSF/eRMSF) | Heavy atoms (not H*) | Per-atom fluctuation |
| Protein (RMSF) | CA atoms | Per-residue fluctuation |
| Complex | Ligand heavy + Protein CA | Combined analysis |
| Ligand (all) | All ligand atoms | Energy, distance |
| Protein (all) | All protein atoms | Energy, distance |

---

## APO Protein / APO Peptide

For single-chain simulations without an interaction partner.

- All protein/peptide atoms are treated as one unit
- Interaction-dependent analyses are disabled (Distance, Contact, ProLIF, Interaction Energy)
- Structural analyses remain available (RMSD, RMSF, Rg, PCA, DSSP, eRMSF, FEL)

---

## File Requirements

### Topology PDB

- Single-frame PDB file
- Must contain valid atom names and residue information
- For Holo (Peptide + Protein): chain IDs recommended
- For Holo (Protein + Ligand): ligand must have a unique residue name
- Shared across all replicas

### Trajectory Files

| Format | Extension | Notes |
|--------|-----------|-------|
| DCD | `.dcd` | CHARMM/NAMD format |
| XTC | `.xtc` | GROMACS compressed |
| TRR | `.trr` | GROMACS full precision |
| NetCDF | `.nc` | Amber format |
| PDB | `.pdb` | Multi-frame PDB |

### Optional Files

| File | Purpose |
|------|---------|
| **ProLIF PDB** | PDB with explicit hydrogens and element column for accurate interaction fingerprints |
| **FF Topology** | PRMTOP/TOP file for accurate Coulomb + LJ interaction energy calculations |

---

## Analysis Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Simulation time (ns) | 100 | Total simulation time for x-axis scaling |
| Contact cutoff (Å) | 4.5 | Distance threshold for residue contacts |
| FEL temperature (K) | 300 | Temperature for free energy calculation |
| FEL bins | 100 | Number of bins for 2D histogram |
| eRMSF skip | 10 | Frames per segment for eRMSF |
| eRMSF vmin/vmax | 0 / 5 | Colorbar range for eRMSF heatmap |
