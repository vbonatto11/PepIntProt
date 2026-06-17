# FAQ

Frequently asked questions about PepIntProt.

---

## General

??? question "What trajectory formats are supported?"
    PepIntProt supports DCD, XTC, TRR, NetCDF (.nc), and multi-frame PDB files.
    Any format readable by MDAnalysis should work.

??? question "Do I need a GPU?"
    No. All analyses run on CPU. The Google Colab free tier is sufficient.

??? question "How long does analysis take?"
    Depends on trajectory size and selected analyses. Typical:
    
    - 1000 frames, basic analyses: ~30 seconds
    - 5000 frames, all analyses: ~3–5 minutes
    - 10000 frames, all analyses + ProLIF: ~10–15 minutes

??? question "Can I use GROMACS .gro files as topology?"
    No. PepIntProt requires a PDB file as topology. Convert with:
    ```bash
    gmx editconf -f structure.gro -o structure.pdb
    ```

---

## System Setup

??? question "How do I find my ligand's residue name?"
    Open your PDB file and look for HETATM records:
    ```bash
    grep "^HETATM" topology.pdb | awk '{print $4}' | sort -u
    ```
    Common names: LIG, MOL, UNK, DRG, or specific codes like JZ4, ATP.

??? question "What if my ligand has multiple residue numbers?"
    This is unusual but can happen with multi-fragment ligands. PepIntProt
    selects ALL atoms matching the residue name, regardless of residue number.

??? question "Chain detection fails — what do I do?"
    Try switching from "By chain ID" to "By resid gap" detection method.
    Or disable auto-detection and specify manual selection strings.

??? question "My peptide is detected as protein (> 40 residues)"
    PepIntProt automatically labels chains > 40 residues as "Protein".
    If this is incorrect, use manual chain selection.

---

## Analyses

??? question "Why is DSSP empty for my ligand?"
    Ligands (small molecules) have no secondary structure. In Protein+Ligand
    mode, DSSP is computed only for the protein component.

??? question "ProLIF shows no interactions — why?"
    Common causes:
    
    1. Missing hydrogens in the topology PDB
    2. Missing or incorrect element column in PDB
    3. Ligand too far from protein in all frames
    
    Solution: Upload a ProLIF-specific PDB with explicit H atoms.

??? question "Interaction energy values seem wrong"
    Without a force field topology (.prmtop/.top), PepIntProt uses a
    simplified geometric approximation. Upload a proper FF topology for
    accurate Coulomb + LJ calculations.

---

## AI Reports

??? question "Which free AI model should I use?"
    **Groq — LLaMA 4 Scout 17B** is the best free option. It has vision
    support and produces high-quality reports. Get a free key at
    [console.groq.com](https://console.groq.com).

??? question "Report generation fails with timeout"
    Some models may time out on very large result sets. Try:
    
    1. A faster model (LLaMA 3.1 8B Instant or Claude Haiku)
    2. Reducing the number of analyses
    3. Checking your API key is valid

??? question "Can I use my own local LLM?"
    Yes! Select Ollama as the provider and ensure `ollama serve` is running
    locally with your preferred model pulled.

---

## Output & Export

??? question "Where are my results saved?"
    Results are saved to a temporary directory and packaged into a ZIP
    for download. In Colab, they're at `/content/PepIntProt_results/`.

??? question "Can I get higher resolution plots?"
    Plots are saved at 200 dpi by default. The CSV data files allow you
    to regenerate plots at any resolution with your own styling.

??? question "PDF has missing characters"
    PepIntProt uses DejaVu Sans font for full Unicode support. If you see
    missing characters, ensure the font is available in your environment.
