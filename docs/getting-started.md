# Installation

## Requirements

- Python 3.10–3.12
- conda (recommended for MDAnalysis and RDKit)

---

## Local Installation (Recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/vbonatto11/PepIntProt.git
cd PepIntProt
```

### 2. Create Conda Environment

```bash
conda create -n pip_env python=3.12
conda activate pip_env
```

### 3. Install Conda Dependencies

MDAnalysis, mdtraj, RDKit, and ProLIF install best via conda-forge:

```bash
conda install -c conda-forge mdanalysis mdtraj rdkit prolif
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install eRMSF (from GitHub)

```bash
pip install git+https://github.com/pablo-arantes/ermsfkit.git
```

### 6. Run the App

```bash
streamlit run app.py --server.port 8000
```

The app will open in your browser at `http://localhost:8000`.

---

## Google Colab (Zero Install)

No installation required — open the notebook and run:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vbonatto11/PepIntProt/blob/main/PepIntProt_Colab.ipynb)

All dependencies are installed automatically in the first cell.

---

## Databricks Apps

### Prerequisites

- Databricks workspace with Apps enabled
- The `app.yaml` handles conda/pip environment setup automatically

### Deployment

1. Clone the repository into your workspace:
   ```
   /Shared/PepIntProt/
   ```

2. Deploy as a Databricks App:
   ```bash
   databricks apps create pepintprot --source-code-path /Shared/PepIntProt
   ```

3. The app will be available at your workspace's Apps URL.

---

## Dependencies Overview

| Category | Packages |
|----------|----------|
| **Core** | streamlit, numpy, pandas, matplotlib, seaborn, scipy |
| **MD Analysis** | MDAnalysis, mdtraj, prolif, ermsfkit |
| **Chemistry** | RDKit (for ProLIF) |
| **Force Field** | parmed (optional, for accurate interaction energy) |
| **PDF** | fpdf2 |
| **AI/LLM** | anthropic, groq, google-generativeai, openai, databricks-sdk[openai] |

---

## Troubleshooting

!!! warning "Common Issues"

    **MDAnalysis won't install via pip**: Use conda-forge instead.
    
    **ProLIF import errors**: Ensure RDKit is installed via conda-forge, not pip.
    
    **ermsfkit not found**: Install directly from GitHub (see step 5 above).
    
    **Streamlit port in use**: Change the port: `--server.port 8001`

!!! tip "Verifying Installation"

    ```python
    import MDAnalysis
    import mdtraj
    import prolif
    import eRMSF
    print("All dependencies OK!")
    ```
