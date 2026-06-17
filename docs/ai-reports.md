# AI Reports

PepIntProt generates professional PDF analysis reports using large language models.

---

## How It Works

1. All analysis metrics are collected (RMSD mean/std, top flexible residues, etc.)
2. A detailed prompt is constructed describing the system and results
3. The LLM generates a structured markdown report
4. The report is rendered to PDF with embedded plots (vision-capable models)

---

## Supported Providers

### Databricks (Workspace)

No API key required — uses your Databricks workspace credentials.

| Model | Vision | Notes |
|-------|:------:|-------|
| Claude Opus 4.7 | ✓ | Best quality |
| Claude Opus 4.6 | ✓ | |
| Claude Sonnet 4.5 | ✓ | Good speed/quality balance |
| Claude Haiku 4.5 | ✓ | Fastest |
| Qwen3-Next 80B | — | Free tier |

### Groq (Free)

Get a free API key at [console.groq.com](https://console.groq.com).

| Model | Vision | Notes |
|-------|:------:|-------|
| LLaMA 4 Scout 17B | ✓ | Best free option |
| LLaMA 3.3 70B | — | High quality |
| LLaMA 3.1 8B Instant | — | Fastest |
| Gemma 2 9B | — | Lightweight |

### OpenAI

| Model | Vision | Notes |
|-------|:------:|-------|
| GPT-4.1 | ✓ | Latest |
| GPT-4.1 Mini | ✓ | Cost-efficient |
| GPT-4o | ✓ | |
| o3 Mini | — | Reasoning |

### Google Gemini

| Model | Vision | Notes |
|-------|:------:|-------|
| Gemini 2.5 Pro | ✓ | Most capable |
| Gemini 2.5 Flash | ✓ | Fast |
| Gemini 2.0 Flash | ✓ | Fastest |

### Anthropic

| Model | Vision | Notes |
|-------|:------:|-------|
| Claude Opus 4 | ✓ | Best quality |
| Claude Sonnet 4 | ✓ | Balanced |

### Ollama (Local)

Run models locally — no internet required.

| Model | Vision | Notes |
|-------|:------:|-------|
| Qwen3.5 7B | — | Lightweight |
| LLaMA 3.3 70B | — | High quality |
| DeepSeek R1 32B | — | Reasoning |
| Gemma 3 27B | ✓ | Vision support |

!!! note "Ollama Setup"
    Start the Ollama server before running: `ollama serve`
    
    Not available in Google Colab or Databricks Apps.

---

## Report Structure

The generated report includes:

1. **Executive Summary** — Key findings in 2–3 paragraphs
2. **System Overview** — Simulation parameters and setup
3. **Per-Analysis Interpretation** — Detailed discussion of each analysis
4. **Binding Characterization** — (Holo modes) interaction assessment
5. **Conclusions & Recommendations** — Summary and suggested follow-ups

---

## Vision Support

Models with vision capability (✓) can analyze the actual plot images:

- Plots are encoded as base64 and sent with the prompt
- The model can reference specific visual features
- Results in more accurate and detailed interpretations

Models without vision receive only the numerical metrics.

---

## PDF Rendering

Reports are rendered to PDF with:

- **DejaVu Sans** font for full Unicode support
- Markdown-aware formatting (headers, bold, lists, tables)
- Embedded analysis plots at key sections
- Professional cover page with system information

### Cover Titles by System Type

| System | Cover Title |
|--------|-------------|
| Peptide + Protein | Peptide-Protein Interaction Analysis Report |
| Protein + Ligand | Protein-Ligand Interaction Analysis Report |
| Protein + Protein | Protein-Protein Interaction Analysis Report |
| APO | APO Protein/Peptide Structural Analysis Report |
