# Quick Start

This guide walks you through your first PepIntProt analysis in under 5 minutes.

---

## What You Need

| File | Format | Description |
|------|--------|-------------|
| **Topology** | `.pdb` | Single-frame PDB with chain IDs |
| **Trajectory** | `.dcd`, `.xtc`, `.trr`, `.nc` | MD trajectory file |

---

## Step-by-Step

### 1. Launch the App

```bash
cd PepIntProt
streamlit run app.py --server.port 8000
```

### 2. Select System Type

In the sidebar, choose your system:

- **Holo (Peptide + Protein)** — for peptide–protein complexes
- **Holo (Protein + Ligand)** — for protein–small molecule complexes
- **APO Protein** — for single protein chain

### 3. Upload Files

- Upload your topology PDB (shared across replicas)
- Upload your trajectory file(s)

### 4. Configure & Run

- Select which analyses to run (default: all applicable)
- Adjust parameters if needed (cutoffs, temperature, etc.)
- Click **🚀 Run Analysis**

### 5. Download Results

- Each replica generates a ZIP with plots (PNG) and data (CSV)
- AI-generated PDF report included if an AI model is selected
- Combined report available for multi-replica runs

---

## Example: Peptide–Protein Complex

```
1. System type: Holo (Peptide + Protein)
2. Upload: complex.pdb + trajectory.xtc
3. Enable: Auto-detect chains ✓
4. Analyses: RMSD, RMSF, PCA, ProLIF, Distance
5. AI Model: Groq — LLaMA 4 Scout (free)
6. Click Run!
```

---

## Example: Protein–Ligand Complex

```
1. System type: Holo (Protein + Ligand)
2. Ligand residue name: LIG
3. Upload: complex.pdb + trajectory.xtc
4. Analyses: RMSD, RMSF, Distance, ProLIF, Interaction Energy
5. AI Model: Groq — LLaMA 4 Scout (free)
6. Click Run!
```

!!! tip "Finding the Ligand Residue Name"
    Open your PDB file and look for HETATM records. The 3-letter code in
    columns 18–20 is your ligand residue name (e.g., LIG, MOL, JZ4, ATP).
