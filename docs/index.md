# PepIntProt (PIP) v4.0

<p align="center">
  <img src="assets/logov1.png" alt="PepIntProt Logo" width="600">
</p>

**Peptide–Protein & Protein–Ligand Interaction Profiler from Molecular Dynamics Simulations**

---

## What is PepIntProt?

PepIntProt (PIP) is a comprehensive, interactive tool for analyzing Molecular Dynamics (MD) trajectories. It supports three system types:

- **Peptide–Protein complexes** — auto chain detection, interaction fingerprints, binding analysis
- **Protein–Ligand complexes** — small-molecule binding analysis with per-atom RMSF
- **APO simulations** — single-chain protein or peptide structural dynamics

PepIntProt provides **12 publication-ready analyses** with automated AI-powered PDF reports, all accessible through an intuitive Streamlit web interface or Google Colab notebook.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **3 System Modes** | Holo (Peptide+Protein), Holo (Protein+Ligand), APO |
| **12 Analyses** | RMSD, RMSF, Rg, PCA, DSSP, Distance, Contact, ProLIF, eRMSF, FEL, Interaction Energy, 3D Viz |
| **Multi-Replica** | 1–10 replicas with cross-replica statistics (mean ± std) |
| **AI Reports** | 6 providers, 20+ models including free options (Groq) |
| **Publication-Ready** | 200 dpi plots, CSV exports, PDF reports with vision-based analysis |
| **Flexible Deployment** | Local Streamlit, Google Colab, Databricks Apps |

---

## Quick Links

- [Installation Guide](getting-started.md) — Get up and running in minutes
- [Usage Guide](usage.md) — Learn about system modes and workflow
- [All Analyses](analyses.md) — Detailed description of each analysis
- [Protein–Ligand Mode](protein-ligand.md) — Special guide for small-molecule systems
- [Google Colab](colab.md) — Zero-install browser-based analysis

---

## Supported Platforms

=== "Local (Streamlit)"

    Run the full-featured app locally with all 12 analyses, multi-replica support,
    and 6 AI providers including local Ollama models.

    ```bash
    streamlit run app.py --server.port 8000
    ```

=== "Google Colab"

    Zero-install analysis directly in your browser. Supports Groq (free),
    OpenAI, Google Gemini, and Anthropic for AI reports.

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vbonatto11/PepIntProt/blob/main/PepIntProt_Colab.ipynb)

=== "Databricks Apps"

    Deploy as a managed Databricks App with workspace-integrated AI models
    (Claude via Databricks Foundation Models).

---

## Citation

If you use PepIntProt in your research, please cite:

```bibtex
@software{pepintprot2025,
  title = {PepIntProt (PIP): Peptide-Protein Interaction Profiler
           from Molecular Dynamics Simulations},
  year = {2025},
  url = {https://github.com/vbonatto11/PepIntProt}
}
```
