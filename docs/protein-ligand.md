# Protein–Ligand Mode

This guide covers the special considerations when analyzing protein–ligand MD simulations with PepIntProt.

---

## Overview

The **Holo (Protein + Ligand)** mode is designed for systems containing a protein receptor bound to a small-molecule ligand. Unlike the peptide–protein mode (which uses chain detection), ligand mode separates components by **residue name**.

---

## Setup

### 1. Select System Type

In the sidebar, choose **"Holo (Protein + Ligand)"**.

### 2. Enter Ligand Residue Name

A text input appears where you type the ligand's 3-letter residue name as it appears in your PDB file:

- Common examples: `LIG`, `MOL`, `JZ4`, `ATP`, `GTP`, `NAD`
- Case-sensitive: must match exactly

!!! tip "Finding Your Ligand Residue Name"
    ```bash
    grep "^HETATM" your_topology.pdb | awk '{print $4}' | sort -u
    ```
    Or open the PDB in a molecular viewer and inspect the ligand residue.

### 3. Upload Files

Upload your topology PDB and trajectory as usual. No chain detection is needed — PepIntProt will separate protein from ligand automatically.

---

## How Separation Works

PepIntProt uses these MDAnalysis selection strings:

| Component | Selection |
|-----------|-----------|
| Ligand | `resname {LIG}` |
| Protein | `protein and not resname {LIG}` |

If the ligand residue name is not found in the topology, an error is displayed listing all available residue names.

---

## Analysis Differences

### RMSF — Per-Atom for Ligand

Since a small molecule typically has only 1 residue but many atoms, RMSF is computed **per heavy atom** for the ligand:

- **Ligand**: all heavy atoms (not H*) → one RMSF value per atom
- **Protein**: CA atoms → one RMSF value per residue
- **X-axis**: "Atom Index (heavy atoms | CA)"

This shows which atoms in the ligand are most flexible in the binding pocket.

### DSSP — Protein Only

Ligands have no secondary structure. DSSP is computed exclusively for the protein component. The ligand is automatically excluded by mdtraj's `topology.select("protein")`.

### eRMSF — Per-Atom Heatmap

Like RMSF, eRMSF uses per-atom data for the ligand:

- Labels: `ResName ResID:AtomName` (e.g., `LIG1:C1`, `LIG1:O2`)
- Boundary line: positioned between protein (bottom) and ligand (top) in the heatmap
- Y-axis shows atom-level labels for the ligand portion

### ProLIF — Direct Residue Selection

ProLIF selects the ligand directly by residue name (`resname {LIG}`) rather than using chain-based detection. This ensures correct identification regardless of chain assignments.

### Distance — COM-to-COM

Center-of-mass distance between all ligand atoms and all protein atoms over time. Useful for monitoring binding stability.

### Interaction Energy

Uses all ligand atoms vs all protein atoms for Coulomb and Lennard-Jones calculations.

---

## PDF Report

The AI-generated report automatically identifies the system as a protein–ligand complex and adjusts its language accordingly:

- References "Ligand" instead of "Peptide"
- Discusses binding stability and ligand dynamics
- Report cover title: "Protein-Ligand Interaction Analysis Report"

---

## Best Practices

!!! success "Recommended"

    - Ensure your ligand has a **unique residue name** not shared with protein residues
    - Use a ProLIF topology with hydrogens for better interaction fingerprinting
    - Upload a force field topology (.prmtop) for accurate interaction energies
    - Check that your PDB HETATM records have correct atom names

!!! warning "Common Pitfalls"

    - Ligand residue name typo (check case sensitivity)
    - Ligand sharing a residue name with crystallographic waters (e.g., HOH)
    - Missing HETATM records in topology PDB
    - Ligand split across multiple residue numbers (unusual but possible)

---

## Example Workflow

```
1. System type: Holo (Protein + Ligand)
2. Ligand residue name: JZ4
3. Upload: 3htb_complex.pdb + md_trajectory.xtc
4. Analyses: RMSD, RMSF, Distance, ProLIF, FEL, Interaction Energy
5. AI Model: Groq — LLaMA 4 Scout 17B (free)
6. Run Analysis → Download ZIP
```
