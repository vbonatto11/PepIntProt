# Databricks Apps

PepIntProt can be deployed as a managed Databricks App with workspace-integrated AI models.

---

## Advantages

- **No API keys needed** — uses workspace-level access to Databricks Foundation Models
- **Managed deployment** — automatic scaling and environment management
- **Claude models** — access to Claude Opus/Sonnet/Haiku via Databricks endpoints
- **Team sharing** — accessible to all workspace members

---

## Deployment

### Prerequisites

- Databricks workspace with Apps enabled
- Workspace access to Foundation Model APIs

### Steps

1. **Clone the repository** into your workspace:
   ```
   /Shared/PepIntProt/
   ```

2. **Verify `app.yaml`** exists with the correct configuration:
   ```yaml
   command:
     - streamlit
     - run
     - app.py
     - --server.port
     - "8000"
   ```

3. **Deploy**:
   ```bash
   databricks apps create pepintprot --source-code-path /Shared/PepIntProt
   ```

4. **Access** the app at your workspace's Apps URL.

---

## Environment Setup

The `app.yaml` configures:

- Python 3.12 via miniforge
- conda-forge packages: MDAnalysis, mdtraj, RDKit, ProLIF
- pip packages from `requirements.txt`
- ermsfkit from GitHub

---

## AI Models (Databricks)

When running as a Databricks App, workspace-level models are available without API keys:

| Model | Vision | Speed |
|-------|:------:|-------|
| Claude Opus 4.7 | ✓ | Slow (best quality) |
| Claude Opus 4.6 | ✓ | Slow |
| Claude Sonnet 4.5 | ✓ | Medium |
| Claude Haiku 4.5 | ✓ | Fast |
| Qwen3-Next 80B | — | Medium (free tier) |

These models are accessed via the Databricks SDK and don't count against external API quotas.
