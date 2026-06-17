# Analyses

PepIntProt provides 12 complementary analyses for MD trajectory characterization.
Each analysis generates publication-quality plots (200 dpi PNG) and raw data (CSV).

---

## 1. 3D Visualization

Generates trajectory snapshots and center-of-mass (COM) traces.

**Outputs:**

- 3D COM trace colored by time
- XY projection of COM movement
- Conformational overlay (representative frames)

**Available for:** All system modes

---

## 2. RMSD — Root Mean Square Deviation

Measures structural deviation from the reference frame (frame 0) over time.

**Computed for:**

- Protein (CA atoms)
- Peptide/Ligand (CA or heavy atoms)
- Complex (all selected atoms)

**Interpretation:**

- Plateau = system equilibrated
- Drift = ongoing conformational change
- Jumps = discrete conformational transitions

---

## 3. RMSF — Root Mean Square Fluctuation

Per-residue (or per-atom) positional fluctuation averaged over the trajectory.

**Peptide + Protein mode:**

- Per-residue RMSF using CA atoms for both components
- Boundary line separating peptide from protein

**Protein + Ligand mode:**

- Per-atom RMSF using heavy atoms for the ligand
- Per-residue RMSF using CA atoms for the protein
- X-axis: "Atom Index (heavy atoms | CA)"

**Interpretation:**

- High RMSF = flexible regions
- Low RMSF = rigid/structured regions
- Compare peptide/ligand vs protein flexibility

---

## 4. Radius of Gyration

Measures compactness of each component over time.

**Computed for:**

- Protein
- Peptide/Ligand
- Complex

**Interpretation:**

- Decreasing Rg = compaction/folding
- Increasing Rg = unfolding/dissociation
- Stable Rg = maintained structure

---

## 5. PCA — Principal Component Analysis

Reduces trajectory dimensionality to show dominant motions.

**Method:**

1. Align trajectory to average structure
2. Compute covariance matrix of atomic fluctuations
3. Project onto PC1 and PC2
4. Color by simulation time

**Interpretation:**

- Clustered points = stable conformation
- Spread = conformational sampling
- Transitions = time-colored paths between clusters

---

## 6. DSSP — Secondary Structure

Assigns secondary structure (Helix/Strand/Coil) per residue per frame using mdtraj.

**Outputs:**

- Per-chain heatmap (residue vs time, colored by SS type)
- Stacked area plot (% Helix/Strand/Coil over time)
- Average SS composition bar chart

**Protein + Ligand mode:** Only protein DSSP is computed (ligands have no secondary structure).

---

## 7. Distance — COM Distance

Center-of-mass distance between peptide/ligand and protein over time.

**Available for:** Holo modes only

**Interpretation:**

- Stable distance = maintained binding
- Increasing = dissociation
- Fluctuation = dynamic binding

---

## 8. Contact Analysis

Identifies residues in contact (distance < cutoff) over the trajectory.

**Outputs:**

- Contact frequency bar chart (top residues)
- Contact timeline heatmap (binary: contact/no contact per frame)

**Parameters:** Contact cutoff (default: 4.5 Å)

---

## 9. ProLIF — Interaction Fingerprints

Detailed interaction analysis using the ProLIF library.

**Interaction types detected:**

- Hydrogen bonds (donor/acceptor)
- Hydrophobic contacts
- π-stacking (face-to-face, edge-to-face)
- Salt bridges
- Cation-π interactions

**Outputs:**

- Interaction frequency bar chart
- Interaction timeline heatmap
- Residue–residue interaction map
- Per-residue interaction profiles

!!! tip "Best Results"
    Upload a ProLIF-specific PDB topology with explicit hydrogens and a
    correct element column for reliable H-bond and π-stacking detection.

---

## 10. eRMSF — Ensemble RMSF

Time-resolved RMSF using the ermsfkit library. Shows how flexibility changes over the simulation.

**Method:**

1. Divide trajectory into segments (controlled by `skip` parameter)
2. Compute RMSF within each segment
3. Display as heatmap (residue/atom vs time)

**Outputs:**

- Complex heatmap with boundary line between components
- eRMSF vs traditional RMSF comparison plot
- Top 5 flexible residues over time

---

## 11. Free Energy Landscape (FEL)

2D free energy surface from RMSD and Radius of Gyration.

**Method:**

$$G(x,y) = -k_B T \ln P(x,y)$$

Where P(x,y) is the probability from the 2D histogram of RMSD vs Rg.

**Outputs:**

- Contour plot with energy minima marked
- Representative frame PDB from the lowest-energy basin

**Parameters:** Temperature (K), number of bins

---

## 12. Interaction Energy

Non-bonded interaction energy between peptide/ligand and protein.

**Two modes:**

| Mode | Input | Accuracy |
|------|-------|----------|
| **Accurate** | FF topology (.prmtop/.top) | Full Coulomb + LJ with proper parameters |
| **Simplified** | PDB only | Geometric MM approximation |

**Outputs:**

- Total energy over time
- Coulomb component
- Lennard-Jones component
- Statistics (mean ± std)

**Available for:** Holo modes only
