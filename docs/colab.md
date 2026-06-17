# Google Colab

PepIntProt is available as a Google Colab notebook for zero-install, browser-based analysis.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vbonatto11/PepIntProt/blob/main/PepIntProt_Colab.ipynb)

---

## Features

- **Zero installation** — all dependencies installed automatically
- **Hidden code cells** — clean interface showing only configuration forms
- **Google Drive integration** — upload/download via Colab's native interface
- **Free compute** — works on free Colab tier (no GPU required)
- **4 AI providers** — Groq (free), OpenAI, Google Gemini, Anthropic

---

## Notebook Structure

| Cell | Title | Description |
|------|-------|-------------|
| 1 | Introduction | Overview and instructions |
| 2 | Install Dependencies | Automated conda/pip setup (~3 min) |
| 3 | Import Libraries | Load all required modules |
| 4 | Upload Files | System type selection + file upload |
| 5 | Configure Parameters | Analysis settings and selection |
| 6 | Run Analysis | Execute all selected analyses |
| 7 | Generate AI Report | Optional LLM-powered report |
| 8 | Download Results | Package and download ZIP |

---

## Workflow

### 1. Open the Notebook

Click the "Open in Colab" badge above, or upload `PepIntProt_Colab.ipynb` manually.

### 2. Install Dependencies (Cell 2)

Run the first code cell. This installs:

- MDAnalysis, mdtraj via conda-forge
- ProLIF, RDKit
- ermsfkit from GitHub
- All other pip dependencies

!!! note "Installation Time"
    First run takes ~3 minutes. Subsequent runs in the same session are instant.

### 3. Select System Type (Cell 4)

Choose from the dropdown:

- **Holo (Peptide + Protein)**
- **Holo (Protein + Ligand)** — prompts for ligand residue name
- **APO Protein**
- **APO Peptide**

### 4. Upload Files (Cell 4)

Colab's native upload dialog appears. Upload:

- Topology PDB
- Trajectory file (DCD/XTC/TRR/NC)

### 5. Configure & Run (Cells 5–6)

Adjust parameters in the form interface, then run the analysis cell.

### 6. Download (Cell 8)

Results are packaged into a ZIP and auto-downloaded.

---

## Limitations vs Streamlit App

| Feature | Streamlit | Colab |
|---------|:---------:|:-----:|
| Multi-replica | ✓ (1–10) | Single replica |
| AI providers | 6 | 4 |
| Ollama (local) | ✓ | — |
| Databricks models | ✓ | — |
| Interactive widgets | ✓ | Forms only |
| Persistent downloads | ✓ | Session-based |
| PDF report | Full | Simplified |

---

## Troubleshooting

!!! warning "Session Timeout"
    Free Colab sessions disconnect after ~90 min idle. Save results before stepping away.

!!! tip "Large Trajectories"
    For trajectories > 1 GB, upload to Google Drive first, then mount and
    reference the file path directly.
