# Changelog

All notable changes to PepIntProt are documented here.

---

## v4.0 (2025)

### Added

- **Protein–Ligand mode** — new "Holo (Protein + Ligand)" system type
    - Separates protein and ligand by residue name
    - Per-atom RMSF/eRMSF for ligand heavy atoms
    - Ligand-aware ProLIF fingerprinting
    - Protein-only DSSP (ligand excluded)
    - Dedicated AI report prompt for protein–ligand systems
    - PDF cover title adapts to system type
- **Multi-replica support** — analyze 1–10 independent replicas
    - Per-replica results with individual reports
    - Combined mean ± std statistics
    - Cross-replica comparison plots
    - Combined AI report discussing reproducibility
- **6 AI providers** — Databricks, Anthropic, Groq (free), OpenAI, Google Gemini, Ollama
- **Vision support** — plot images embedded in AI prompts for capable models
- **eRMSF analysis** — ensemble RMSF using ermsfkit
- **3D Visualization** — COM traces and conformational overlays
- **Free Energy Landscape** — 2D FEL with representative frame extraction
- **Interaction Energy** — Coulomb + LJ with parmed support
- **Persistent downloads** — @st.fragment buttons that don't trigger page rerun
- **Unicode PDF** — DejaVu Sans font for full character support
- **Google Colab notebook** — zero-install browser analysis

### Changed

- Restructured codebase into single `app.py` (~3500 lines)
- Improved chain detection with residue ID gap method
- Better ProLIF error handling with 3-attempt inference strategy
- eRMSF boundary line correctly positioned for each system type

### Fixed

- RMSF array length mismatch in ligand mode (per-atom vs per-residue)
- eRMSF boundary line position for protein-ligand (ligand after protein in topology)
- DSSP chain detection respects ligand mode (no "ligand secondary structure")
- PDF cover title now dynamic per system type

---

## v3.0 (2024)

### Added

- Initial public release
- Peptide–protein interaction analysis
- APO mode
- Basic AI report generation
- Streamlit web interface

---

## Roadmap

- [ ] Hydrogen bond lifetime analysis
- [ ] Water bridge detection
- [ ] MM-GBSA/MM-PBSA integration
- [ ] Trajectory clustering (GROMOS, K-means)
- [ ] Markov State Model (MSM) analysis
- [ ] Allosteric pathway detection
