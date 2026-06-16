"""PepIntProt (PIP) v4.0 — MD Trajectory Analysis App

Upload trajectory + topology files, select analyses, download results as ZIP.

Analyses: RMSD, RMSF, Rg, PCA, DSSP, Distance, Contact, ProLIF,
eRMSF, and 3-D trajectory visualisation.

GitHub ermsfkit: github.com/pablo-arantes/ermsfkit
"""

import streamlit as st
import os, sys, glob, warnings, io, zipfile, time, tempfile, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from statistics import mean, stdev
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, BoundaryNorm


import base64
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False
try:
    import anthropic as _anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
try:
    from openai import OpenAI as _OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
try:
    from groq import Groq as _Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
try:
    import google.generativeai as _genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
import requests as _requests  # for Ollama local HTTP API

from scipy.ndimage import gaussian_filter as _gaussian_filter

try:
    import parmed as _parmed
    HAS_PARMED = True
except ImportError:
    HAS_PARMED = False

warnings.filterwarnings("ignore")

# ── Ensure miniforge site-packages are on sys.path ─────────────────
for sp in glob.glob("/tmp/miniforge/lib/python*/site-packages"):
    if sp not in sys.path:
        sys.path.insert(0, sp)

import MDAnalysis as mda
from MDAnalysis.analysis import rms, align, distances
from MDAnalysis.analysis.rms import RMSF as MDA_RMSF
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.topology.guessers import guess_types, guess_masses

# ====================================================================
# PDF FONT SETUP (Unicode-compatible DejaVu Sans via matplotlib)
# ====================================================================
import matplotlib as _mpl_font
_FONT_DIR = os.path.join(os.path.dirname(_mpl_font.__file__), "mpl-data", "fonts", "ttf")
_FONT_NAME = "DejaVu"  # Used throughout PDF generation

def _register_fonts(pdf_obj):
    """Register DejaVu Sans Unicode fonts with an FPDF instance."""
    pdf_obj.add_font(_FONT_NAME, "", os.path.join(_FONT_DIR, "DejaVuSans.ttf"))
    pdf_obj.add_font(_FONT_NAME, "B", os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf"))
    pdf_obj.add_font(_FONT_NAME, "I", os.path.join(_FONT_DIR, "DejaVuSans-Oblique.ttf"))
    pdf_obj.add_font(_FONT_NAME, "BI", os.path.join(_FONT_DIR, "DejaVuSans-BoldOblique.ttf"))

# ====================================================================
# PAGE CONFIG
# ====================================================================
st.set_page_config(
    page_title="PepIntProt (PIP)",
    page_icon="logov1.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logo & Title ──
_LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logov1.png")
_HAS_LOGO = os.path.exists(_LOGO_PATH)

# Show logo in sidebar navigation
if _HAS_LOGO:
    st.logo(_LOGO_PATH)

# Main title with logo
if _HAS_LOGO:
    _col_logo, _col_title = st.columns([0.06, 0.94])
    with _col_logo:
        st.image(_LOGO_PATH, width=72)
    with _col_title:
        st.title("PepIntProt (PIP) v4.0")
        st.caption("Peptide\u2013Protein Interaction Analysis from Molecular Dynamics Simulations")
else:
    st.title("\U0001f9ec PepIntProt (PIP) v4.0")
    st.caption("Peptide\u2013Protein Interaction Analysis from Molecular Dynamics Simulations")


# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def _save_upload(uploaded_file, directory):
    """Save a Streamlit UploadedFile to disk and return the path."""
    path = os.path.join(directory, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def detect_chains(u):
    chain_info = []
    try:
        chain_ids = sorted(set(cid for cid in u.atoms.chainIDs if cid.strip()))
        if len(chain_ids) >= 2:
            for cid in chain_ids:
                atoms = u.select_atoms(f"chainID {cid}")
                prot = atoms.select_atoms("protein or resname ACE NME")
                if len(prot) > 0:
                    resids = sorted(set(prot.resids))
                    chain_info.append(dict(id=cid, sel_prefix="chainID",
                        n_residues=len(resids), resids=resids,
                        first_resid=min(resids), last_resid=max(resids)))
    except Exception:
        pass
    if len(chain_info) < 2:
        chain_info = []
        prot = u.select_atoms("protein or resname ACE NME")
        resids = sorted(set(prot.resids))
        chains, cur = [], [resids[0]]
        for i in range(1, len(resids)):
            if resids[i] - resids[i-1] > 1:
                chains.append(cur); cur = [resids[i]]
            else:
                cur.append(resids[i])
        chains.append(cur)
        for idx, cr in enumerate(chains):
            chain_info.append(dict(id=str(idx+1), sel_prefix="resid",
                n_residues=len(cr), resids=cr,
                first_resid=min(cr), last_resid=max(cr)))
    if len(chain_info) < 2:
        raise ValueError("Need >= 2 chains. Use manual selections.")
    chain_info.sort(key=lambda x: x["n_residues"])
    pep, prot = chain_info[0], chain_info[-1]
    pf = pep["sel_prefix"]
    if pf == "chainID":
        return f"chainID {pep['id']}", f"chainID {prot['id']}", pep, prot
    return (f"resid {pep['first_resid']}:{pep['last_resid']}",
            f"resid {prot['first_resid']}:{prot['last_resid']}", pep, prot)


def make_time_array(n_frames, simulation_time_ns):
    return np.linspace(0, simulation_time_ns, n_frames)


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def save_and_show(fig, label, out_dir=None):
    img = fig_to_bytes(fig)
    st.image(img, use_container_width=True)
    if out_dir:
        path = os.path.join(out_dir, f"{label}.png")
        with open(path, "wb") as f:
            f.write(img)




# ====================================================================
# LLM CALL HELPER  (Databricks Foundation Model API or Anthropic API)
# ====================================================================

def _build_openai_messages(prompt, supports_vision, image_paths):
    """Build OpenAI-compatible messages list with optional vision content."""
    import os as _os
    if supports_vision and image_paths:
        content_parts = []
        for img_path in (image_paths or []):
            if _os.path.exists(img_path):
                with open(img_path, "rb") as f_img:
                    b64 = base64.standard_b64encode(f_img.read()).decode("utf-8")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })
                content_parts.append({
                    "type": "text",
                    "text": f"[Above image: {_os.path.basename(img_path)}]"
                })
        content_parts.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content_parts}]
    else:
        return [{"role": "user", "content": prompt}]


def call_llm(prompt, model_config, api_key="", image_paths=None, max_tokens=4096):
    """Unified LLM call. Returns generated text.

    Supports:
    - Databricks Foundation Model API (OpenAI-compatible, workspace auth)
    - Anthropic API (direct, requires api_key)
    - Groq (OpenAI-compatible, requires api_key)
    - OpenAI / ChatGPT (direct, requires api_key)
    - Google Gemini (requires api_key)
    - Ollama (local HTTP, no key needed)
    """
    provider = model_config["provider"]
    endpoint = model_config["endpoint"]
    supports_vision = model_config.get("supports_vision", False)

    if provider == "databricks":
        # ── Databricks Foundation Model API (OpenAI-compatible) ──
        import os as _os
        client = None

        # Strategy 1: databricks-sdk (auto-auth in Databricks Apps)
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            client = w.serving_endpoints.get_open_ai_client()
        except Exception:
            pass

        # Strategy 2: Manual OpenAI with env vars
        if client is None:
            if not HAS_OPENAI:
                raise ImportError(
                    "Neither databricks-sdk nor openai package installed. "
                    "Run: pip install databricks-sdk[openai]")
            db_token = _os.environ.get("DATABRICKS_TOKEN", "")
            db_host = _os.environ.get("DATABRICKS_HOST", "")
            if not db_host or not db_token:
                raise ValueError(
                    "Cannot authenticate to Databricks. Ensure "
                    "DATABRICKS_HOST and DATABRICKS_TOKEN are set, "
                    "or install databricks-sdk for automatic auth.")
            client = _OpenAI(
                api_key=db_token,
                base_url=f"{db_host.rstrip('/')}/serving-endpoints",
            )

        messages = _build_openai_messages(prompt, supports_vision, image_paths)
        response = client.chat.completions.create(
            model=endpoint, max_tokens=max_tokens, messages=messages)
        return response.choices[0].message.content

    elif provider == "anthropic":
        # ── Anthropic API (direct) ──
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic package not installed. "
                              "Run: pip install anthropic")
        if not api_key:
            raise ValueError("Anthropic API key is required.")

        client = _anthropic.Anthropic(api_key=api_key)

        # Build message content with vision support
        msg_content = []
        if supports_vision and image_paths:
            import os as _os
            for img_path in (image_paths or []):
                if _os.path.exists(img_path):
                    with open(img_path, "rb") as f_img:
                        b64 = base64.standard_b64encode(f_img.read()).decode("utf-8")
                    msg_content.append({"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": b64}})
                    msg_content.append({"type": "text",
                        "text": f"[Above image: {_os.path.basename(img_path)}]"})
        msg_content.append({"type": "text", "text": prompt})

        response = client.messages.create(
            model=endpoint, max_tokens=max_tokens,
            messages=[{"role": "user", "content": msg_content}])
        return response.content[0].text

    elif provider == "groq":
        # ── Groq Cloud (OpenAI-compatible) ──
        if not HAS_GROQ:
            raise ImportError("groq package not installed. Run: pip install groq")
        if not api_key:
            raise ValueError("Groq API key is required.")
        client = _Groq(api_key=api_key)
        messages = _build_openai_messages(prompt, supports_vision, image_paths)
        response = client.chat.completions.create(
            model=endpoint, max_tokens=max_tokens, messages=messages)
        return response.choices[0].message.content

    elif provider == "openai":
        # ── OpenAI / ChatGPT (direct) ──
        if not HAS_OPENAI:
            raise ImportError("openai package not installed. Run: pip install openai")
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        client = _OpenAI(api_key=api_key)
        messages = _build_openai_messages(prompt, supports_vision, image_paths)
        response = client.chat.completions.create(
            model=endpoint, max_tokens=max_tokens, messages=messages)
        return response.choices[0].message.content

    elif provider == "gemini":
        # ── Google Gemini ──
        if not HAS_GENAI:
            raise ImportError("google-generativeai package not installed. "
                              "Run: pip install google-generativeai")
        if not api_key:
            raise ValueError("Google Gemini API key is required.")
        _genai.configure(api_key=api_key)
        model = _genai.GenerativeModel(endpoint)
        # Build content parts for Gemini (vision support)
        parts = []
        if supports_vision and image_paths:
            import os as _os
            for img_path in (image_paths or []):
                if _os.path.exists(img_path):
                    with open(img_path, "rb") as f_img:
                        img_bytes = f_img.read()
                    parts.append({"mime_type": "image/png", "data": img_bytes})
                    parts.append(f"[Above image: {_os.path.basename(img_path)}]")
        parts.append(prompt)
        response = model.generate_content(
            parts, generation_config={"max_output_tokens": max_tokens})
        return response.text

    elif provider == "ollama":
        # ── Ollama (local HTTP API) ──
        ollama_url = model_config.get("ollama_url", "http://localhost:11434")
        import os as _os
        # Build images list for vision models
        images_b64 = []
        if supports_vision and image_paths:
            for img_path in (image_paths or []):
                if _os.path.exists(img_path):
                    with open(img_path, "rb") as f_img:
                        images_b64.append(
                            base64.standard_b64encode(f_img.read()).decode("utf-8"))
        payload = {"model": endpoint, "prompt": prompt, "stream": False,
                   "options": {"num_predict": max_tokens}}
        if images_b64:
            payload["images"] = images_b64
        resp = _requests.post(f"{ollama_url}/api/generate", json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json().get("response", "")

    else:
        raise ValueError(f"Unknown provider: {provider}")


def _sanitize_for_pdf(text):
    """Minimal sanitization for PDF text. DejaVu Sans handles most Unicode.
    Only replace chars that cause layout issues in fpdf2."""
    replacements = {
        # Dashes and quotes
        '\u2013': '-', '\u2014': '--', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2026': '...',
        # Math operators and symbols
        '\u00b1': '+/-', '\u00d7': 'x', '\u2248': '~',
        '\u2264': '<=', '\u2265': '>=', '\u2260': '!=',
        '\u2212': '-', '\u2211': 'sum', '\u221a': 'sqrt',
        '\u221e': 'inf', '\u2261': '==',
        # Greek letters (uppercase)
        '\u0394': 'Delta', '\u03a3': 'Sigma', '\u03a9': 'Omega',
        '\u0393': 'Gamma', '\u03a0': 'Pi', '\u039b': 'Lambda',
        # Greek letters (lowercase)
        '\u03b1': 'alpha', '\u03b2': 'beta', '\u03b3': 'gamma',
        '\u03b4': 'delta', '\u03b5': 'epsilon', '\u03b6': 'zeta',
        '\u03b7': 'eta', '\u03b8': 'theta', '\u03ba': 'kappa',
        '\u03bb': 'lambda', '\u03bc': 'mu', '\u03bd': 'nu',
        '\u03c0': 'pi', '\u03c1': 'rho', '\u03c3': 'sigma',
        '\u03c4': 'tau', '\u03c6': 'phi', '\u03c8': 'psi',
        '\u03c9': 'omega', '\u03c7': 'chi',
        # Angstrom, degree, arrows
        '\u00c5': 'A', '\u00e5': 'a', '\u00b0': 'deg',
        '\u2192': '->', '\u2190': '<-', '\u2194': '<->',
        '\u2191': '^', '\u2193': 'v',
        # Subscripts and superscripts
        '\u2080': '0', '\u2081': '1', '\u2082': '2', '\u2083': '3',
        '\u2084': '4', '\u2085': '5', '\u2086': '6', '\u2087': '7',
        '\u2088': '8', '\u2089': '9',
        '\u2070': '0', '\u00b9': '1', '\u00b2': '2', '\u00b3': '3',
        '\u2074': '4', '\u2075': '5', '\u2076': '6', '\u2077': '7',
        '\u2078': '8', '\u2079': '9',
        '\u207a': '+', '\u207b': '-', '\u208a': '+', '\u208b': '-',
        # Misc
        '\u2022': '*', '\u2023': '>', '\u25cf': '*', '\u25cb': 'o',
        '\u00b7': '.', '\u2032': "'", '\u2033': '"',
        '\u00bc': '1/4', '\u00bd': '1/2', '\u00be': '3/4',
        '\u00a9': '(c)', '\u00ae': '(R)',
        '\u2019': "'",  # right single quotation
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text




def _render_markdown_para(pdf_obj, text, font_size=10):
    """Render a paragraph with inline **bold** and *italic* to PDF."""
    import re
    # Split on **bold** markers
    parts = re.split(r'(\*\*.*?\*\*)', text)
    line_parts = []
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            line_parts.append(("B", part[2:-2]))
        else:
            # Check for *italic*
            sub_parts = re.split(r'(\*.*?\*)', part)
            for sp in sub_parts:
                if sp.startswith("*") and sp.endswith("*") and len(sp) > 2:
                    line_parts.append(("I", sp[1:-1]))
                else:
                    line_parts.append(("", sp))

    # If it's simple (no formatting), just use multi_cell
    if len(line_parts) == 1 and line_parts[0][0] == "":
        pdf_obj.set_font(_FONT_NAME, "", font_size)
        pdf_obj.multi_cell(0, 5, line_parts[0][1])
        return

    # For mixed formatting, render word by word using write()
    # First, join everything and use multi_cell with stripped markdown
    plain = text.replace("**", "").replace("*", "")
    pdf_obj.set_font(_FONT_NAME, "", font_size)
    pdf_obj.multi_cell(0, 5, plain)



def _write_markdown_text(pdf_obj, text, font_size=10, line_height=5):
    """Write text with inline **bold** and *italic* using fpdf2 write()."""
    import re
    pdf_obj.set_font(_FONT_NAME, "", font_size)
    parts = re.split(r'(\*\*[^*]+\*\*)', text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            pdf_obj.set_font(_FONT_NAME, "B", font_size)
            pdf_obj.write(line_height, part[2:-2])
            pdf_obj.set_font(_FONT_NAME, "", font_size)
        else:
            sub_parts = re.split(r'(\*[^*]+\*)', part)
            for sp in sub_parts:
                if sp.startswith("*") and sp.endswith("*") and len(sp) > 2:
                    pdf_obj.set_font(_FONT_NAME, "I", font_size)
                    pdf_obj.write(line_height, sp[1:-1])
                    pdf_obj.set_font(_FONT_NAME, "", font_size)
                else:
                    pdf_obj.write(line_height, sp)


def _add_text_to_pdf(pdf_obj, text, font_size=10):
    """Robust markdown-to-PDF renderer with bold/italic, tables, and bullets."""
    text = _sanitize_for_pdf(text)
    line_h = 5
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            pdf_obj.ln(3)
            continue
        try:
            if "|" in para and para.strip().startswith("|"):
                _rows = [r.strip() for r in para.split("\n") if r.strip()]
                _rows = [r for r in _rows if not all(c in "|-: " for c in r)]
                if _rows:
                    _cols = [c.strip() for c in _rows[0].split("|") if c.strip()]
                    n_cols = max(len(_cols), 1)
                    avail_w = pdf_obj.w - pdf_obj.l_margin - pdf_obj.r_margin
                    col_w = avail_w / n_cols
                    if col_w < 20 or n_cols > 8:
                        pdf_obj.set_font(_FONT_NAME, "", font_size - 1)
                        for row in _rows:
                            pdf_obj.multi_cell(0, line_h, row.replace("|", " | ").replace("**", "").strip())
                        pdf_obj.ln(3)
                    else:
                        max_chars = max(int(col_w / 2.2), 5)
                        for ri, row in enumerate(_rows):
                            cells = [c.strip() for c in row.split("|") if c.strip()]
                            if ri == 0:
                                pdf_obj.set_font(_FONT_NAME, "B", font_size - 1)
                            else:
                                pdf_obj.set_font(_FONT_NAME, "", font_size - 1)
                            for cell_text in cells[:n_cols]:
                                pdf_obj.cell(col_w, 6, cell_text.replace("**", "")[:max_chars], border=1)
                            pdf_obj.ln()
                        pdf_obj.ln(3)
                continue
            if para.lstrip().startswith("- ") or para.lstrip().startswith("* "):
                for bline in para.split("\n"):
                    bline = bline.strip()
                    if not bline:
                        continue
                    if bline.startswith("- ") or bline.startswith("* "):
                        bline = bline[2:]
                    if not bline:
                        continue
                    pdf_obj.set_font(_FONT_NAME, "", font_size)
                    pdf_obj.set_x(pdf_obj.l_margin)
                    pdf_obj.write(line_h, "  " + chr(8226) + " ")
                    _write_markdown_text(pdf_obj, bline, font_size, line_h)
                    pdf_obj.ln(line_h)
                pdf_obj.ln(2)
                continue
            _write_markdown_text(pdf_obj, para, font_size, line_h)
            pdf_obj.ln(line_h + 3)
        except Exception:
            try:
                pdf_obj.set_font(_FONT_NAME, "", font_size)
                safe = ''.join(c if ord(c) < 128 else '?' for c in
                               para.replace("**", "").replace("*", "").replace("|", " "))
                pdf_obj.multi_cell(0, line_h, safe)
                pdf_obj.ln(3)
            except Exception:
                pdf_obj.ln(5)


# ====================================================================
# SIDEBAR CONFIGURATION
# ====================================================================
with st.sidebar:
    st.header("\u2699\ufe0f Configuration")

    # ── System Type (must be defined before conditional uploads) ──
    st.subheader("System Type")
    system_type = st.selectbox(
        "Simulation system",
        ["Holo (Peptide + Protein)", "Holo (Protein + Ligand)", "APO Protein", "APO Peptide"],
        index=0,
        help="Holo (Peptide + Protein): peptide\u2013protein complex. "
             "Holo (Protein + Ligand): protein\u2013small molecule complex. "
             "APO: single-chain simulation (protein-only or peptide-only). "
             "APO mode disables interaction analyses.")
    is_holo = system_type.startswith("Holo")
    is_holo_ligand = system_type == "Holo (Protein + Ligand)"

    # ── Ligand residue name (only for Protein + Ligand mode) ──
    if is_holo_ligand:
        ligand_resname = st.text_input(
            "Ligand residue name",
            value="LIG",
            help="Enter the 3-letter residue name of the ligand as it appears "
                 "in your topology file (e.g., LIG, UNK, MOL, JZ4). "
                 "The app will separate protein and ligand based on this name.")
    else:
        ligand_resname = ""

    # ── Replicas ──
    st.subheader("\U0001f501 Replicas")
    n_replicas = st.number_input("Number of replicas", min_value=1, max_value=10,
                                  value=1, step=1,
                                  help="Run the same analyses on multiple independent MD replicas. "
                                       "If > 1, a combined analysis (mean \u00b1 std) is generated.")

    # ── Upload Files ──
    st.subheader("\U0001f4c1 Upload Files")
    top_up = st.file_uploader(
        "Topology PDB (shared across replicas)",
        type=["pdb"],
    )

    # Per-replica trajectory uploads
    traj_uploads = []
    for _ri in range(n_replicas):
        _label = f"Trajectory — Replica {_ri+1}" if n_replicas > 1 else "Trajectory (DCD / XTC / TRR / NC / PDB)"
        traj_uploads.append(st.file_uploader(
            _label,
            type=["dcd", "xtc", "trr", "nc", "pdb"],
            key=f"traj_rep_{_ri}",
        ))

    if is_holo:
        plf_up = st.file_uploader(
            "ProLIF topology \u2014 optional (PDB / MOL2 / TPR / TOP / PRMTOP / PSF)",
            type=["pdb", "mol2", "tpr", "top", "prmtop", "psf"],
        )
        ff_up = st.file_uploader(
            "FF Topology \u2014 optional for Interaction Energy (PRMTOP / TOP / PDB)",
            type=["prmtop", "top", "pdb"],
            help="Force field topology with charges & LJ params (.prmtop/.top). "
                 "PDB files are accepted but lack FF parameters \u2014 the app will "
                 "auto-detect and fall back to simplified MM if needed."
        )
    else:
        plf_up = None
        ff_up = None

    st.subheader("Simulation")
    simulation_time_ns = st.number_input(
        "Total simulation time (ns)", value=500.0, min_value=0.1, step=50.0)

    if is_holo and not is_holo_ligand:
        st.subheader("Chain detection")
        auto_detect = st.checkbox("Auto-detect chains", value=True)
        if not auto_detect:
            manual_pep = st.text_input("Peptide selection", value="chainID A")
            manual_prot = st.text_input("Protein selection", value="chainID B")
    elif is_holo_ligand:
        auto_detect = False
    else:
        auto_detect = True

    if is_holo:
        st.subheader("Contact analysis")
        contact_cutoff = st.slider("Contact cutoff (\u00c5)", 3.0, 10.0, 5.0, 0.5)
    else:
        contact_cutoff = 5.0

    st.subheader("eRMSF")
    ermsf_skip = st.slider("Frames per segment", 2, 50, 10)
    ermsf_vmin = st.number_input("Heatmap vmin (\u00c5)", value=0.0)
    ermsf_vmax = st.number_input("Heatmap vmax (\u00c5)", value=4.0)

    st.subheader("Free Energy Landscape")
    fel_temperature = st.number_input("Temperature (K)", value=300.0, min_value=1.0, step=10.0)
    fel_bins = st.slider("FEL bins", 30, 100, 50, 5)
    if is_holo:
        _fel_options = (["Protein + Ligand", "All atoms"] if is_holo_ligand
                        else ["Peptide + Protein", "All atoms"])
        fel_extract_selection = st.selectbox(
            "Atoms to extract (representative frame)",
            _fel_options,
            index=0)
    else:
        fel_extract_selection = "All atoms"

    if is_holo:
        st.subheader("ProLIF")
        prolif_top_n = st.slider("Top N interactions", 5, 40, 20)
        prolif_frame_skip = st.select_slider(
            "Frame skip (speed up)",
            options=[1, 2, 5, 10, 20, 50],
            value=1,
            help="Analyze every Nth frame. Skip=10 on 1000 frames \u2192 100 frames analyzed")
    else:
        prolif_top_n = 20
        prolif_frame_skip = 1

    st.subheader("\U0001f4dd AI Report")

    AVAILABLE_MODELS = {
        # ── Databricks (workspace auth, no key needed) ──
        "Databricks \u2014 Claude Opus 4.7": {
            "provider": "databricks", "endpoint": "databricks-claude-opus-4-7",
            "supports_vision": True, "label": "Claude Opus 4.7 (Databricks)"},
        "Databricks \u2014 Claude Opus 4.6": {
            "provider": "databricks", "endpoint": "databricks-claude-opus-4-6",
            "supports_vision": True, "label": "Claude Opus 4.6 (Databricks)"},
        "Databricks \u2014 Claude Sonnet 4.5": {
            "provider": "databricks", "endpoint": "databricks-claude-sonnet-4-5",
            "supports_vision": True, "label": "Claude Sonnet 4.5 (Databricks)"},
        "Databricks \u2014 Claude Haiku 4.5": {
            "provider": "databricks", "endpoint": "databricks-claude-haiku-4-5",
            "supports_vision": True, "label": "Claude Haiku 4.5 (Databricks)"},
        "Databricks \u2014 Qwen3-Next 80B": {
            "provider": "databricks", "endpoint": "databricks-qwen3-next-80b-a3b-instruct",
            "supports_vision": False, "label": "Qwen3-Next 80B (Databricks, free tier)"},
        # ── Anthropic API (requires key) ──
        "Anthropic API \u2014 Claude Opus 4": {
            "provider": "anthropic", "endpoint": "claude-opus-4-20250514",
            "supports_vision": True, "label": "Claude Opus 4 (Anthropic API)"},
        "Anthropic API \u2014 Claude Sonnet 4": {
            "provider": "anthropic", "endpoint": "claude-sonnet-4-20250514",
            "supports_vision": True, "label": "Claude Sonnet 4 (Anthropic API)"},
        # ── Groq (free, requires key) ──
        "Groq \u2014 LLaMA 4 Scout": {
            "provider": "groq", "endpoint": "meta-llama/llama-4-scout-17b-16e-instruct",
            "supports_vision": True, "label": "LLaMA 4 Scout 17B (Groq, free)"},
        "Groq \u2014 LLaMA 3.3 70B": {
            "provider": "groq", "endpoint": "llama-3.3-70b-versatile",
            "supports_vision": False, "label": "LLaMA 3.3 70B (Groq, free)"},
        "Groq \u2014 LLaMA 3.1 8B": {
            "provider": "groq", "endpoint": "llama-3.1-8b-instant",
            "supports_vision": False, "label": "LLaMA 3.1 8B Instant (Groq, free)"},
        "Groq \u2014 Gemma 2 9B": {
            "provider": "groq", "endpoint": "gemma2-9b-it",
            "supports_vision": False, "label": "Gemma 2 9B (Groq, free)"},
        # ── OpenAI (requires key) ──
        "OpenAI \u2014 GPT-4.1": {
            "provider": "openai", "endpoint": "gpt-4.1",
            "supports_vision": True, "label": "GPT-4.1 (OpenAI)"},
        "OpenAI \u2014 GPT-4.1 Mini": {
            "provider": "openai", "endpoint": "gpt-4.1-mini",
            "supports_vision": True, "label": "GPT-4.1 Mini (OpenAI)"},
        "OpenAI \u2014 GPT-4o": {
            "provider": "openai", "endpoint": "gpt-4o",
            "supports_vision": True, "label": "GPT-4o (OpenAI)"},
        "OpenAI \u2014 o3 Mini": {
            "provider": "openai", "endpoint": "o3-mini",
            "supports_vision": False, "label": "o3 Mini (OpenAI)"},
        # ── Google Gemini (requires key) ──
        "Gemini \u2014 2.5 Pro": {
            "provider": "gemini", "endpoint": "gemini-2.5-pro",
            "supports_vision": True, "label": "Gemini 2.5 Pro (Google)"},
        "Gemini \u2014 2.5 Flash": {
            "provider": "gemini", "endpoint": "gemini-2.5-flash",
            "supports_vision": True, "label": "Gemini 2.5 Flash (Google)"},
        "Gemini \u2014 2.0 Flash": {
            "provider": "gemini", "endpoint": "gemini-2.0-flash",
            "supports_vision": True, "label": "Gemini 2.0 Flash (Google)"},
        # ── Ollama (local, no key needed) ──
        "Ollama \u2014 Local Model": {
            "provider": "ollama", "endpoint": "",
            "supports_vision": False, "label": "Local Model (Ollama)"},
    }

    selected_model_name = st.selectbox(
        "AI Model for Report",
        list(AVAILABLE_MODELS.keys()),
        index=0,
        help="Databricks: workspace auth (no key). Groq: free API key. "
             "OpenAI/Gemini/Anthropic: paid API key. Ollama: local (no key).")
    selected_model = AVAILABLE_MODELS[selected_model_name]

    llm_api_key = ""
    _provider = selected_model["provider"]
    if _provider == "anthropic":
        llm_api_key = st.text_input("Anthropic API Key", type="password",
            help="Required for direct Anthropic API calls. "
                 "Get yours at console.anthropic.com")
    elif _provider == "groq":
        llm_api_key = st.text_input("Groq API Key", type="password",
            help="Free API key from console.groq.com")
    elif _provider == "openai":
        llm_api_key = st.text_input("OpenAI API Key", type="password",
            help="Get yours at platform.openai.com/api-keys")
    elif _provider == "gemini":
        llm_api_key = st.text_input("Google Gemini API Key", type="password",
            help="Get yours at aistudio.google.com/apikey")
    elif _provider == "ollama":
        _ollama_model = st.text_input(
            "Model name",
            value="",
            placeholder="e.g. gemma4:latest, qwen3.5:latest, llama3:8b",
            help="Type the exact model name as shown by \'ollama list\' on your machine.")
        if _ollama_model.strip():
            selected_model["endpoint"] = _ollama_model.strip()
            selected_model["label"] = f"{_ollama_model.strip()} (Ollama local)"
        _ollama_url = st.text_input("Ollama URL", value="http://localhost:11434",
            help="URL of your local Ollama server (default is fine for most setups)")
        selected_model["ollama_url"] = _ollama_url
        st.caption("\u2699\ufe0f Ollama runs locally \u2014 no API key needed. "
                   "Run \'ollama list\' in your terminal to see available models.")
    else:
        st.caption("\u2705 Using Databricks workspace authentication \u2014 no API key needed.")

    generate_report = st.checkbox("Generate PDF report with AI", value=True,
        help="The selected model analyzes your results and generates a professional PDF report")

    _HOLO_ONLY = {"Distance", "Contact", "ProLIF", "Interaction Energy"}
    _ALL_ANALYSES = ["3D Visualization", "RMSD", "RMSF", "Radius of Gyration",
         "PCA", "DSSP", "Distance", "Contact", "ProLIF", "eRMSF",
         "Free Energy Landscape", "Interaction Energy"]
    if is_holo:
        available_analyses = _ALL_ANALYSES
        default_analyses = ["RMSD", "RMSF", "Radius of Gyration", "PCA",
                 "DSSP", "Distance", "Contact", "eRMSF",
                 "3D Visualization", "ProLIF",
                 "Free Energy Landscape", "Interaction Energy"]
    else:
        available_analyses = [a for a in _ALL_ANALYSES if a not in _HOLO_ONLY]
        default_analyses = list(available_analyses)
        st.info("\U0001f4a1 APO mode: Distance, Contact, ProLIF, and Interaction "
                "Energy are disabled (require peptide\u2013protein complex).")

    analyses = st.multiselect(
        "Analyses to run",
        available_analyses,
        default=default_analyses,
    )

    run_btn = st.button("\U0001f680 Run Analysis", type="primary", use_container_width=True)

    st.divider()
    if st.button("\U0001f9f9 Clear Cache", use_container_width=True,
                 help="Remove all temporary files from previous runs and reset the session."):
        # Remove all pepintprot_ temp dirs from /tmp
        for _d in glob.glob("/tmp/pepintprot_*"):
            shutil.rmtree(_d, ignore_errors=True)
        # Also clear the one stored in session_state
        _prev = st.session_state.pop("tmp_dir", None)
        if _prev and os.path.isdir(_prev):
            shutil.rmtree(_prev, ignore_errors=True)
        # Clear all result keys
        for _k in list(st.session_state.keys()):
            if _k.startswith(("zip_data_", "zip_name_", "pdf_data_", "pdf_name_",
                               "all_zip_keys", "all_pdf_keys", "result_images")):
                del st.session_state[_k]
        st.session_state["analysis_done"] = False
        st.success("\u2705 Cache cleared successfully.")
        st.rerun()



# ====================================================================
# SINGLE-REPLICA ANALYSIS FUNCTION
# ====================================================================
def _run_single_replica(traj_path, top_path, plf_path, ff_path, out_dir, rep_label=""):
    """Run all selected analyses for one replica.

    Accesses sidebar variables (analyses, is_holo, system_type, simulation_time_ns,
    contact_cutoff, ermsf_skip, etc.) from enclosing module scope via closure.

    Returns: (report_data, numeric_data)
    """
    numeric = {}

    prog = st.progress(0, text=f"Loading trajectory{rep_label}...")


    # Load universe
    u = mda.Universe(top_path, traj_path)
    if np.all(u.atoms.masses == 0):
        u.atoms.types = guess_types(u.atoms.names)
        u.atoms.masses = guess_masses(u.atoms.types)

    n_frames = len(u.trajectory)
    n_atoms = len(u.atoms)
    time_array = make_time_array(n_frames, simulation_time_ns)
    sim_ns = time_array[-1]
    numeric["_time_array"] = time_array

    # Detect chains (Holo) or use single chain (APO)
    if is_holo and is_holo_ligand:
        # Protein + Ligand mode: separate by residue name
        _lig_atoms = u.select_atoms(f"resname {ligand_resname}")
        if len(_lig_atoms) == 0:
            st.error(f"No atoms found with resname '{ligand_resname}'. "
                     f"Available residue names: {', '.join(sorted(set(u.atoms.resnames))[:30])}")
            st.stop()
        _prot_atoms = u.select_atoms(f"protein and not resname {ligand_resname}")
        if len(_prot_atoms) == 0:
            st.error("No protein atoms found after excluding the ligand.")
            st.stop()
        peptide_sel = f"resname {ligand_resname}"
        protein_sel = f"protein and not resname {ligand_resname}"
        _lig_resids = sorted(set(_lig_atoms.resids))
        _prot_resids = sorted(set(_prot_atoms.resids))
        peptide_info = dict(n_residues=len(_lig_resids),
            first_resid=min(_lig_resids), last_resid=max(_lig_resids))
        protein_info = dict(n_residues=len(_prot_resids),
            first_resid=min(_prot_resids), last_resid=max(_prot_resids))
    elif is_holo:
        if auto_detect:
            peptide_sel, protein_sel, peptide_info, protein_info = detect_chains(u)
        else:
            peptide_sel, protein_sel = manual_pep, manual_prot
            pep_a = u.select_atoms(peptide_sel)
            prot_a = u.select_atoms(protein_sel)
            peptide_info = dict(n_residues=len(set(pep_a.resids)),
                first_resid=int(min(pep_a.resids)), last_resid=int(max(pep_a.resids)))
            protein_info = dict(n_residues=len(set(prot_a.resids)),
                first_resid=int(min(prot_a.resids)), last_resid=int(max(prot_a.resids)))
    else:
        # APO mode: single chain — select all protein/peptide atoms
        _apo_atoms = u.select_atoms("protein or resname ACE NME")
        _apo_resids = sorted(set(_apo_atoms.resids))
        _apo_sel = f"resid {min(_apo_resids)}:{max(_apo_resids)}"
        _apo_info = dict(n_residues=len(_apo_resids),
            first_resid=min(_apo_resids), last_resid=max(_apo_resids))
        peptide_sel = _apo_sel
        protein_sel = _apo_sel
        peptide_info = _apo_info
        protein_info = _apo_info

    if is_holo_ligand:
        # Ligand mode: use all heavy atoms for the ligand (no CA backbone)
        pep_ca = f"({peptide_sel}) and not name H*"  # heavy atoms as proxy for CA
        prot_ca = f"({protein_sel}) and name CA"
        cx_ca = f"(({peptide_sel}) and not name H*) or (({protein_sel}) and name CA)"
        pep_all = f"({peptide_sel})"
        prot_all = f"({protein_sel})"
    else:
        pep_ca = f"({peptide_sel}) and name CA"
        prot_ca = f"({protein_sel}) and name CA"
        cx_ca = f"(({peptide_sel}) or ({protein_sel})) and name CA"
        pep_all = f"({peptide_sel}) and (protein or resname ACE NME)"
        prot_all = f"({protein_sel}) and (protein or resname ACE NME)"

    # ── Display labels ──
    if is_holo_ligand:
        _pep_label = "Ligand"
        _prot_label = "Protein"
        _system_interaction = "Protein–Ligand"
    elif is_holo and peptide_info["n_residues"] > 40:
        _pep_label = "Protein"
        _prot_label = "Protein Target"
        _system_interaction = "Protein–Protein Target"
    elif is_holo:
        _pep_label = "Peptide"
        _prot_label = "Protein Target"
        _system_interaction = "Peptide–Protein Target"
    else:
        _apo_lbl = "Protein" if system_type == "APO Protein" else "Peptide"
        _pep_label = _apo_lbl
        _prot_label = _apo_lbl
        _system_interaction = f"APO {_apo_lbl}"

    # Info bar
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Atoms", n_atoms)
    c2.metric("Frames", n_frames)
    c3.metric("Simulation", f"{sim_ns:.0f} ns")
    c4.metric("Files uploaded", f"{2 + (1 if plf_path else 0) + (1 if ff_path else 0)}")

    if is_holo:
        st.info(f"**{_pep_label}**: {peptide_sel} ({peptide_info['n_residues']} res)  |  "
                f"**{_prot_label}**: {protein_sel} ({protein_info['n_residues']} res)")
    else:
        _apo_label = "Protein" if system_type == "APO Protein" else "Peptide"
        st.info(f"**APO {_apo_label}**: {peptide_sel} ({peptide_info['n_residues']} residues)")

    total_analyses = len(analyses)
    report_data = {"system": {
        "n_atoms": n_atoms, "n_frames": n_frames, "simulation_ns": sim_ns,
        "system_type": system_type,
        "peptide_sel": peptide_sel, "protein_sel": protein_sel,
        "peptide_n_res": peptide_info["n_residues"],
        "protein_n_res": protein_info["n_residues"],
        "ligand_resname": ligand_resname if is_holo_ligand else None},
        "analyses": {}, "plots": {}}

    completed = 0

    # ================================================================
    # 3D TRAJECTORY VISUALIZATION
    # ================================================================
    if "3D Visualization" in analyses:
        prog.progress(completed / total_analyses, text="3D Trajectory Visualization...")
        with st.expander("\U0001f310 3D Trajectory Visualization", expanded=True):
            prot_viz = u.select_atoms(prot_ca)
            pep_viz = u.select_atoms(pep_ca)

            # ---- Snapshots ----
            snap_fracs = [0, 0.25, 0.5, 0.75, 1.0]
            snap_frames = [min(int(f*(n_frames-1)), n_frames-1) for f in snap_fracs]
            all_p, all_q = [], []
            for fi in snap_frames:
                u.trajectory[fi]
                all_p.append(prot_viz.positions.copy())
                all_q.append(pep_viz.positions.copy())
            coords = np.vstack(all_p + all_q)
            pad = 5
            xl = (coords[:,0].min()-pad, coords[:,0].max()+pad)
            yl = (coords[:,1].min()-pad, coords[:,1].max()+pad)
            zl = (coords[:,2].min()-pad, coords[:,2].max()+pad)

            fig = plt.figure(figsize=(22, 5))
            if is_holo:
                _3d_title = f"{_system_interaction} \u2014 Trajectory Snapshots (CA)"
            else:
                _apo_lbl = "Protein" if system_type == "APO Protein" else "Peptide"
                _3d_title = f"APO {_apo_lbl} \u2014 Trajectory Snapshots (CA)"
            fig.suptitle(_3d_title, fontsize=14, fontweight="bold", y=1.04)
            for i, (fi, pp, qq) in enumerate(zip(snap_frames, all_p, all_q)):
                ax = fig.add_subplot(1, 5, i+1, projection="3d")
                ax.plot(pp[:,0], pp[:,1], pp[:,2], c="#2196F3", lw=1, alpha=.5)
                ax.scatter(pp[:,0], pp[:,1], pp[:,2], c="#2196F3", s=4, alpha=.5)
                ax.plot(qq[:,0], qq[:,1], qq[:,2], c="#E91E63", lw=2, alpha=.9)
                ax.scatter(qq[:,0], qq[:,1], qq[:,2], c="#E91E63", s=30, alpha=1,
                           edgecolors="k", linewidths=.3)
                ax.set_xlim(xl); ax.set_ylim(yl); ax.set_zlim(zl)
                ax.set_title(f"{time_array[fi]:.0f} ns", fontweight="bold")
                ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                for pn in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                    pn.fill = False; pn.set_edgecolor("lightgrey")
                ax.view_init(elev=20, azim=45+i*10)
            handles = [
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
                           markersize=8, label=f"{_prot_label} (CA)"),
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#E91E63",
                           markersize=8, label=f"{_pep_label} (CA)"),
            ]
            fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
                       frameon=False, bbox_to_anchor=(0.5, -0.04))
            fig.tight_layout()
            save_and_show(fig, "trajectory_snapshots", out_dir)

            # ---- COM trace (actual 3D + XY, matching notebook) ----
            pep_com = np.zeros((n_frames, 3))
            prot_com = np.zeros((n_frames, 3))
            for i, _ in enumerate(u.trajectory):
                pep_com[i] = pep_viz.center_of_mass()
                prot_com[i] = prot_viz.center_of_mass()

            u.trajectory[0]
            prot_pos_first = prot_viz.positions.copy()
            u.trajectory[-1]
            prot_pos_last = prot_viz.positions.copy()

            cmap_com = cm.plasma
            colors_com = cmap_com(np.linspace(0, 1, n_frames))

            fig = plt.figure(figsize=(16, 6))
            fig.suptitle(f"{_pep_label} Center-of-Mass Trajectory (colored by time)",
                         fontsize=14, fontweight="bold")

            # 3D view
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax1.scatter(prot_pos_first[:,0], prot_pos_first[:,1], prot_pos_first[:,2],
                        c="#BBDEFB", s=3, alpha=0.25, label="Protein (t=0)")
            for j in range(n_frames - 1):
                ax1.plot(pep_com[j:j+2, 0], pep_com[j:j+2, 1], pep_com[j:j+2, 2],
                         color=colors_com[j], linewidth=1.2, alpha=0.8)
            ax1.scatter(*pep_com[0], c="lime", s=80, edgecolors="k", zorder=5,
                        label="Start (0 ns)")
            ax1.scatter(*pep_com[-1], c="red", s=80, edgecolors="k", zorder=5,
                        label=f"End ({sim_ns:.0f} ns)")
            ax1.set_xlabel("X (\u00c5)", fontsize=9)
            ax1.set_ylabel("Y (\u00c5)", fontsize=9)
            ax1.set_zlabel("Z (\u00c5)", fontsize=9)
            ax1.legend(fontsize=8, loc="upper left")
            ax1.view_init(elev=25, azim=60)
            for pn in [ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane]:
                pn.fill = False

            # 2D XY projection
            ax2 = fig.add_subplot(1, 2, 2)
            sc = ax2.scatter(pep_com[:,0], pep_com[:,1],
                             c=time_array, cmap="plasma", s=4, alpha=0.7)
            ax2.scatter(prot_com[0,0], prot_com[0,1],
                        c="#2196F3", s=100, marker="s", edgecolors="k",
                        label=f"{_prot_label} COM (t=0)", zorder=5)
            ax2.scatter(pep_com[0,0], pep_com[0,1],
                        c="lime", s=60, edgecolors="k", label=f"{_pep_label} start", zorder=5)
            ax2.scatter(pep_com[-1,0], pep_com[-1,1],
                        c="red", s=60, edgecolors="k", label=f"{_pep_label} end", zorder=5)
            cbar = plt.colorbar(sc, ax=ax2, label="Time (ns)", pad=0.02)
            ax2.set_xlabel("X (\u00c5)", fontsize=11, fontweight="bold")
            ax2.set_ylabel("Y (\u00c5)", fontsize=11, fontweight="bold")
            ax2.set_title("XY Projection", fontsize=12, fontweight="bold")
            ax2.legend(fontsize=8, loc="best")
            ax2.set_aspect("equal")

            fig.tight_layout()
            save_and_show(fig, "trajectory_peptide_com", out_dir)

            # ---- Multi-frame overlay ----
            ovl_frames = np.linspace(0, n_frames - 1, 10, dtype=int)
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.set_title(f"{_pep_label} Conformational Ensemble on {_prot_label} Surface\n"
                         "(10 evenly spaced frames, CA atoms)",
                         fontsize=13, fontweight="bold")
            u.trajectory[0]
            prot_p0 = prot_viz.positions.copy()
            ax.plot(prot_p0[:,0], prot_p0[:,1], prot_p0[:,2],
                    c="#2196F3", lw=1.2, alpha=0.4)
            ax.scatter(prot_p0[:,0], prot_p0[:,1], prot_p0[:,2],
                       c="#2196F3", s=6, alpha=0.3)
            cmap_ovl = cm.magma
            ovl_colors = cmap_ovl(np.linspace(0.15, 0.9, len(ovl_frames)))
            for idx, fi in enumerate(ovl_frames):
                u.trajectory[fi]
                pep_pos = pep_viz.positions.copy()
                t_ns = time_array[fi]
                ax.plot(pep_pos[:,0], pep_pos[:,1], pep_pos[:,2],
                        color=ovl_colors[idx], lw=2, alpha=0.8)
                ax.scatter(pep_pos[:,0], pep_pos[:,1], pep_pos[:,2],
                           color=ovl_colors[idx], s=20, alpha=0.9)
                if idx == 0 or idx == len(ovl_frames) - 1:
                    mid = len(pep_pos) // 2
                    ax.text(pep_pos[mid,0], pep_pos[mid,1], pep_pos[mid,2],
                            f"  {t_ns:.0f} ns", fontsize=8,
                            color=ovl_colors[idx], fontweight="bold")
            sm = plt.cm.ScalarMappable(cmap=cmap_ovl,
                                       norm=plt.Normalize(0, sim_ns))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08, label="Time (ns)")
            ax.view_init(elev=20, azim=55)
            for pn in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                pn.fill = False; pn.set_edgecolor("lightgrey")
            ax.set_xlabel("X (\u00c5)", fontsize=9)
            ax.set_ylabel("Y (\u00c5)", fontsize=9)
            ax.set_zlabel("Z (\u00c5)", fontsize=9)
            fig.tight_layout()
            save_and_show(fig, "trajectory_peptide_overlay", out_dir)

            u.trajectory[0]

            disp = np.linalg.norm(pep_com[-1] - pep_com[0])
            drift = np.max(np.linalg.norm(pep_com - pep_com[0], axis=1))
            fluct = np.std(np.linalg.norm(pep_com - pep_com.mean(axis=0), axis=1))
            st.markdown(f"**COM displacement**: {disp:.2f} \u00c5 &nbsp;|&nbsp; "
                        f"**Max drift**: {drift:.2f} \u00c5 &nbsp;|&nbsp; "
                        f"**Fluctuation \u03c3**: {fluct:.2f} \u00c5")
            report_data["analyses"]["3D Visualization"] = {
                "com_displacement": f"{disp:.2f}", "max_drift": f"{drift:.2f}",
                "fluctuation_std": f"{fluct:.2f}"}
            report_data["plots"]["3D Visualization"] = [
                "trajectory_snapshots.png", "trajectory_peptide_com.png",
                "trajectory_peptide_overlay.png"]
        completed += 1

    # ================================================================
    # RMSD
    # ================================================================
    if "RMSD" in analyses:
        prog.progress(completed / total_analyses, text="Computing RMSD...")
        with st.expander("\U0001f4c9 RMSD Analysis", expanded=True):
            rmsd_data = {}
            if is_holo:
                _rmsd_items = [(_prot_label, prot_ca, "#2196F3"),
                               (_pep_label, pep_ca, "#E91E63"),
                               ("Complex", cx_ca, "#4CAF50")]
            else:
                _apo_lbl = "Protein" if system_type == "APO Protein" else "Peptide"
                _apo_col = "#2196F3" if system_type == "APO Protein" else "#E91E63"
                _rmsd_items = [(_apo_lbl, cx_ca, _apo_col)]
            for lbl, sel, col in _rmsd_items:
                R = rms.RMSD(u, u, select=sel, ref_frame=0)
                R.run()
                rmsd_data[lbl] = {"values": R.results.rmsd[:, 2], "color": col}

            _n_rmsd = len(rmsd_data)
            fig, axes = plt.subplots(_n_rmsd, 1, figsize=(10, max(4, _n_rmsd*3.3)),
                                     sharex=True, squeeze=False)
            axes = axes.flatten()
            for ax, (lbl, d) in zip(axes, rmsd_data.items()):
                ax.plot(time_array, d["values"], alpha=.7, color=d["color"], lw=.8)
                ax.set_ylabel("RMSD (\u00c5)"); ax.set_title(lbl, fontweight="bold")
                ax.set_xlim(0, sim_ns)
            axes[-1].set_xlabel("Time (ns)")
            fig.tight_layout()
            save_and_show(fig, "rmsd_all", out_dir)

            fig, axes = plt.subplots(1, _n_rmsd, figsize=(max(5, _n_rmsd*4.7), 4),
                                     squeeze=False)
            axes = axes.flatten()
            for ax, (lbl, d) in zip(axes, rmsd_data.items()):
                sns.kdeplot(d["values"], color=d["color"], fill=True, alpha=.3, ax=ax)
                ax.set_xlabel("RMSD (\u00c5)"); ax.set_title(lbl, fontweight="bold")
                ax.set_yticks([]); ax.set_ylabel("")
            fig.tight_layout()
            save_and_show(fig, "rmsd_distributions", out_dir)

            cols = st.columns(min(3, _n_rmsd))
            for c, (lbl, d) in zip(cols, rmsd_data.items()):
                v = d["values"]
                c.metric(lbl, f"{mean(v):.2f} \u00b1 {stdev(v):.2f} \u00c5")

            df_rmsd = pd.DataFrame({"Time_ns": time_array,
                **{f"RMSD_{k}": v["values"] for k, v in rmsd_data.items()}})
            report_data["analyses"]["RMSD"] = {
                k: f"{mean(v['values']):.2f} +/- {stdev(v['values']):.2f} A"
                for k, v in rmsd_data.items()}
            report_data["plots"]["RMSD"] = ["rmsd_all.png", "rmsd_distributions.png"]
            df_rmsd.to_csv(os.path.join(out_dir, "rmsd_data.csv"), index=False)
            numeric["rmsd"] = {lbl: d["values"].copy() for lbl, d in rmsd_data.items()}
            numeric["rmsd_colors"] = {lbl: d["color"] for lbl, d in rmsd_data.items()}

        completed += 1

    # ================================================================
    # RMSF  (with fill_between shadow)
    # ================================================================
    if "RMSF" in analyses:
        prog.progress(completed / total_analyses, text="Computing RMSF...")
        with st.expander("\U0001f4c8 RMSF Analysis", expanded=True):
            avg_s = align.AverageStructure(u, u, select=cx_ca, ref_frame=0).run()
            ref_u = avg_s.results.universe
            align.AlignTraj(u, ref_u, select=cx_ca, in_memory=True).run()

            all_ca_atoms = u.select_atoms(cx_ca)
            R_all = MDA_RMSF(all_ca_atoms).run()
            rmsf_values = R_all.results.rmsf
            residue_indices = np.arange(len(rmsf_values))

            if is_holo:
                pep_atoms = u.select_atoms(pep_ca)
                prot_atoms = u.select_atoms(prot_ca)

                if is_holo_ligand:
                    # Ligand mode: multiple heavy atoms per residue
                    # Build per-atom split (each heavy atom gets its own RMSF point)
                    pep_atom_indices = set(pep_atoms.indices)
                    _lig_mask = np.array([a.index in pep_atom_indices for a in all_ca_atoms])
                    rmsf_pep = rmsf_values[_lig_mask]
                    rmsf_prot = rmsf_values[~_lig_mask]
                    boundary = int(_lig_mask.sum())
                    _chain_labels = ["Ligand" if m else "Protein" for m in _lig_mask]
                else:
                    # Standard holo: one CA per residue
                    peptide_resids = sorted(set(pep_atoms.resids))
                    protein_resids = sorted(set(prot_atoms.resids))
                    resid_to_idx = {resid: i for i, resid in enumerate(all_ca_atoms.resids)}
                    rmsf_pep = [rmsf_values[resid_to_idx[r]] for r in peptide_resids if r in resid_to_idx]
                    rmsf_prot = [rmsf_values[resid_to_idx[r]] for r in protein_resids if r in resid_to_idx]
                    boundary = len(rmsf_pep)
                    _chain_labels = ["Peptide"] * len(rmsf_pep) + ["Protein"] * len(rmsf_prot)

                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(residue_indices[:boundary], rmsf_pep,
                        color="#E91E63", lw=1.2, alpha=0.9, label=f"{_pep_label}")
                ax.plot(residue_indices[boundary:boundary+len(rmsf_prot)], rmsf_prot,
                        color="#2196F3", lw=1.2, alpha=0.9, label=f"{_prot_label}")
                ax.fill_between(residue_indices[:boundary], rmsf_pep,
                                alpha=0.15, color="#E91E63")
                ax.fill_between(residue_indices[boundary:boundary+len(rmsf_prot)], rmsf_prot,
                                alpha=0.15, color="#2196F3")
                ymin, ymax = ax.get_ylim()
                ax.axvline(x=boundary - 0.5, color="black", ls="--", lw=1.5, alpha=0.8)
                ax.text(boundary - 1, ymax * 0.95, f"{_pep_label}", ha="right", fontsize=10,
                        fontweight="bold", color="#E91E63")
                ax.text(boundary + 1, ymax * 0.95, f"{_prot_label}", ha="left", fontsize=10,
                        fontweight="bold", color="#2196F3")
                ax.set_title(f"RMSF: {_pep_label} & {_prot_label}",
                             fontsize=14, fontweight="bold")
            else:
                _apo_lbl = "Protein" if system_type == "APO Protein" else "Peptide"
                _apo_col = "#2196F3" if system_type == "APO Protein" else "#E91E63"
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(residue_indices, rmsf_values,
                        color=_apo_col, lw=1.2, alpha=0.9, label=f"APO {_apo_lbl}")
                ax.fill_between(residue_indices, rmsf_values, alpha=0.15, color=_apo_col)
                ax.set_title(f"RMSF: APO {_apo_lbl}", fontsize=14, fontweight="bold")
                _chain_labels = [_apo_lbl] * len(rmsf_values)

            _rmsf_xlabel = "Atom Index (heavy atoms | CA)" if is_holo_ligand else "Residue Index"
            ax.set_xlabel(_rmsf_xlabel, fontsize=13, fontweight="bold")
            ax.set_ylabel("RMSF ($\\AA$)", fontsize=13, fontweight="bold")
            ax.set_xlim(0, len(rmsf_values) - 1)
            ax.tick_params(labelsize=11)
            ax.legend(loc="upper right", fontsize=11, frameon=False)
            fig.tight_layout()
            save_and_show(fig, "rmsf_combined", out_dir)

            rmsf_df = pd.DataFrame({
                "Residue_Index": residue_indices[:len(rmsf_values)],
                "Resid": list(all_ca_atoms.resids),
                "Resname": list(all_ca_atoms.resnames),
                "RMSF": rmsf_values,
                "Chain": _chain_labels,
            })
            report_data["analyses"]["RMSF"] = {
                "top_flexible": ", ".join(
                    f"{rmsf_df.iloc[i]['Resname']}{int(rmsf_df.iloc[i]['Resid'])} ({rmsf_df.iloc[i]['RMSF']:.2f} A)"
                    for i in rmsf_df.nlargest(5, 'RMSF').index)}
            report_data["plots"]["RMSF"] = ["rmsf_combined.png"]
            rmsf_df.to_csv(os.path.join(out_dir, "rmsf_data.csv"), index=False)
            numeric["rmsf"] = rmsf_values.copy()
            numeric["rmsf_resids"] = list(all_ca_atoms.resids)

        completed += 1

    # ================================================================
    # RADIUS OF GYRATION
    # ================================================================
    if "Radius of Gyration" in analyses:
        prog.progress(completed / total_analyses, text="Computing Rg...")
        with st.expander("\U0001f535 Radius of Gyration", expanded=True):
            rg_data = {}
            if is_holo:
                _rg_items = [(_prot_label, prot_all, "#2196F3"),
                             (_pep_label, pep_all, "#E91E63"),
                             ("Complex", cx_ca, "#4CAF50")]
            else:
                _apo_lbl = "Protein" if system_type == "APO Protein" else "Peptide"
                _apo_col = "#2196F3" if system_type == "APO Protein" else "#E91E63"
                _rg_items = [(_apo_lbl, cx_ca, _apo_col)]
            for lbl, sel, col in _rg_items:
                atoms = u.select_atoms(sel)
                vals = np.array([atoms.radius_of_gyration() for _ in u.trajectory])
                rg_data[lbl] = {"values": vals, "color": col}

            _n_rg = len(rg_data)
            fig, axes = plt.subplots(_n_rg, 1, figsize=(10, max(4, _n_rg*3.3)),
                                     sharex=True, squeeze=False)
            axes = axes.flatten()
            for ax, (lbl, d) in zip(axes, rg_data.items()):
                ax.plot(time_array, d["values"], alpha=.7, color=d["color"], lw=.8)
                ax.set_ylabel("Rg (\u00c5)"); ax.set_title(lbl, fontweight="bold")
                ax.set_xlim(0, sim_ns)
            axes[-1].set_xlabel("Time (ns)")
            fig.tight_layout()
            save_and_show(fig, "rg_all", out_dir)

            cols = st.columns(min(3, _n_rg))
            for c, (lbl, d) in zip(cols, rg_data.items()):
                v = d["values"]
                c.metric(lbl, f"{mean(v):.2f} \u00b1 {stdev(v):.2f} \u00c5")

            df_rg = pd.DataFrame({"Time_ns": time_array,
                **{f"Rg_{k}": v["values"] for k, v in rg_data.items()}})
            report_data["analyses"]["Radius of Gyration"] = {
                k: f"{mean(v['values']):.2f} +/- {stdev(v['values']):.2f} A"
                for k, v in rg_data.items()}
            report_data["plots"]["Radius of Gyration"] = ["rg_all.png"]
            df_rg.to_csv(os.path.join(out_dir, "rg_data.csv"), index=False)
            numeric["rg"] = {lbl: d["values"].copy() for lbl, d in rg_data.items()}
            numeric["rg_colors"] = {lbl: d["color"] for lbl, d in rg_data.items()}

        completed += 1

    # ================================================================
    # PCA
    # ================================================================
    if "PCA" in analyses:
        prog.progress(completed / total_analyses, text="Running PCA...")
        with st.expander("\U0001f3af PCA Analysis", expanded=True):
            from MDAnalysis.analysis.pca import PCA as MDA_PCA
            if is_holo:
                _pca_items = [(_prot_label, prot_ca, "#2196F3"),
                              (_pep_label, pep_ca, "#E91E63"),
                              ("Complex", cx_ca, "#4CAF50")]
            else:
                _apo_lbl = "Protein" if system_type == "APO Protein" else "Peptide"
                _apo_col = "#2196F3" if system_type == "APO Protein" else "#E91E63"
                _pca_items = [(_apo_lbl, cx_ca, _apo_col)]
            for lbl, sel, col in _pca_items:
                atoms = u.select_atoms(sel)
                pca = MDA_PCA(u, select=sel).run()
                transformed = pca.transform(atoms, n_components=3)
                var = pca.results.variance
                var_ratio = var / var.sum() * 100

                fig, ax = plt.subplots(figsize=(8, 6))
                sc = ax.scatter(transformed[:, 0], transformed[:, 1],
                                c=time_array, cmap="plasma", s=8, alpha=.7)
                plt.colorbar(sc, ax=ax, label="Time (ns)")
                ax.set_xlabel(f"PC1 ({var_ratio[0]:.1f}%)")
                ax.set_ylabel(f"PC2 ({var_ratio[1]:.1f}%)")
                ax.set_title(f"PCA \u2014 {lbl}", fontweight="bold")
                fig.tight_layout()
                save_and_show(fig, f"pca_{lbl.lower()}", out_dir)

                df_pca = pd.DataFrame(transformed[:, :3],
                                      columns=["PC1", "PC2", "PC3"])
                df_pca.insert(0, "Time_ns", time_array)
                df_pca.to_csv(os.path.join(out_dir, f"pca_{lbl.lower()}.csv"),
                              index=False)
        # PCA metrics for report
        if "PCA" in analyses:
            _pca_labels = [lbl.lower() for lbl, _, _ in _pca_items]
            report_data["analyses"]["PCA"] = f"PC1-PC2 scatter plots generated for {', '.join(l.title() for l in _pca_labels)}"
            report_data["plots"]["PCA"] = [f"pca_{l}.png" for l in _pca_labels]

        completed += 1

    # ================================================================
    # DSSP  (with colorbar legend: Helix/Strand/Coil)
    # ================================================================
    if "DSSP" in analyses:
        prog.progress(completed / total_analyses, text="Computing DSSP...")
        with st.expander("\U0001f9e9 Secondary Structure (DSSP)", expanded=True):
            try:
                import mdtraj
                traj_full = mdtraj.load(traj_path, top=top_path)
                # Filter to protein-only atoms (exclude ions, water, etc.)
                _prot_atom_idx = traj_full.topology.select("protein")
                traj = traj_full.atom_slice(_prot_atom_idx)
                dssp_all = mdtraj.compute_dssp(traj, simplified=True)

                ss_map = {"H": 0, "E": 1, "C": 2, "NA": 3}
                ss_labels = ["Helix", "Strand", "Coil", "N/A"]
                ss_colors = ["#E91E63", "#2196F3", "#BDBDBD", "#EEEEEE"]
                dssp_num = np.vectorize(lambda x: ss_map.get(x, 3))(dssp_all)

                cmap_ss = ListedColormap(ss_colors)
                bounds_ss = [-0.5, 0.5, 1.5, 2.5, 3.5]
                norm_ss = BoundaryNorm(bounds_ss, cmap_ss.N)

                # Detect protein chains (ions already removed)
                mt_chains = [c for c in traj.topology.chains if c.n_residues > 0]
                mt_chains.sort(key=lambda c: c.n_residues)
                if is_holo_ligand:
                    # Ligand has no secondary structure; only compute protein DSSP
                    pep_chain = None
                    prot_chain = mt_chains[-1]
                elif is_holo and len(mt_chains) >= 2:
                    pep_chain = mt_chains[0]
                    prot_chain = mt_chains[-1]
                else:
                    # APO: use the single (or largest) protein chain
                    if system_type == "APO Peptide":
                        pep_chain = mt_chains[-1]
                        prot_chain = None
                    else:
                        pep_chain = None
                        prot_chain = mt_chains[-1]

                pep_idx = [r.index for r in pep_chain.residues] if pep_chain else []
                prot_idx = [r.index for r in prot_chain.residues] if prot_chain else []
                pep_res_labels = [f"{r.name}{r.resSeq}" for r in pep_chain.residues] if pep_chain else []
                prot_res_labels = [f"{r.name}{r.resSeq}" for r in prot_chain.residues] if prot_chain else []

                # ---- Peptide DSSP heatmap ----
                if pep_idx:
                    fig, ax = plt.subplots(figsize=(14, max(3, len(pep_idx)*0.25)))
                    im = ax.imshow(dssp_num[:, pep_idx].T, aspect="auto",
                                   cmap=cmap_ss, norm=norm_ss,
                                   interpolation="nearest", origin="lower",
                                   extent=[0, sim_ns, -0.5, len(pep_idx)-0.5])
                    ax.set_yticks(range(len(pep_res_labels)))
                    ax.set_yticklabels(pep_res_labels, fontsize=8)
                    ax.set_xlabel("Time (ns)", fontsize=12, fontweight="bold")
                    ax.set_ylabel("Residue", fontsize=12, fontweight="bold")
                    _pep_dssp_title = f"Secondary Structure Evolution \u2014 {_pep_label}" if is_holo else "Secondary Structure Evolution \u2014 APO Peptide"
                    ax.set_title(_pep_dssp_title, fontsize=13, fontweight="bold")
                    ax.tick_params(labelsize=10)
                    cbar = plt.colorbar(im, ax=ax, shrink=0.5, ticks=[0, 1, 2, 3])
                    cbar.set_ticklabels(ss_labels)
                    fig.tight_layout()
                    save_and_show(fig, "secondary_structure_peptide", out_dir)

                # ---- Protein DSSP heatmap ----
                if prot_idx:
                    fig, ax = plt.subplots(figsize=(14, max(6, len(prot_idx)*0.08)))
                    im = ax.imshow(dssp_num[:, prot_idx].T, aspect="auto",
                                   cmap=cmap_ss, norm=norm_ss,
                                   interpolation="nearest", origin="lower",
                                   extent=[0, sim_ns, -0.5, len(prot_idx)-0.5])
                    n_pr = len(prot_res_labels)
                    tick_step = max(1, n_pr // 30)
                    ax.set_yticks(range(0, n_pr, tick_step))
                    ax.set_yticklabels(prot_res_labels[::tick_step], fontsize=7)
                    ax.set_xlabel("Time (ns)", fontsize=12, fontweight="bold")
                    ax.set_ylabel("Residue", fontsize=12, fontweight="bold")
                    _prot_dssp_title = f"Secondary Structure Evolution \u2014 {_prot_label}" if is_holo else "Secondary Structure Evolution \u2014 APO Protein"
                    ax.set_title(_prot_dssp_title, fontsize=13, fontweight="bold")
                    ax.tick_params(labelsize=10)
                    cbar = plt.colorbar(im, ax=ax, shrink=0.4, ticks=[0, 1, 2, 3])
                    cbar.set_ticklabels(ss_labels)
                    fig.tight_layout()
                    save_and_show(fig, "secondary_structure_protein", out_dir)

                # ---- SS composition over time (stacked area) ----
                _dssp_domains = []
                if pep_idx:
                    _dssp_domains.append((pep_idx, f"{_pep_label}" if is_holo else "APO Peptide"))
                if prot_idx:
                    _dssp_domains.append((prot_idx, f"{_prot_label}" if is_holo else "APO Protein"))
                _n_dom = len(_dssp_domains)
                if _n_dom > 0:
                    fig, axes_ss = plt.subplots(_n_dom, 1, figsize=(12, _n_dom * 4),
                                                sharex=True, squeeze=False)
                    axes_ss = axes_ss.flatten()
                    for ax, (d_idx, label) in zip(axes_ss, _dssp_domains):
                        data_ss = dssp_num[:, d_idx]
                        n_r = data_ss.shape[1]
                        h_frac = np.sum(data_ss == 0, axis=1) / n_r * 100
                        e_frac = np.sum(data_ss == 1, axis=1) / n_r * 100
                        c_frac = np.sum(data_ss == 2, axis=1) / n_r * 100
                        ax.stackplot(time_array, h_frac, e_frac, c_frac,
                                     labels=["Helix", "Strand", "Coil"],
                                     colors=["#E91E63", "#2196F3", "#BDBDBD"],
                                     alpha=0.7)
                        ax.set_ylabel("SS Content (%)", fontsize=11, fontweight="bold")
                        ax.set_title(label, fontsize=12, fontweight="bold")
                        ax.set_xlim(0, sim_ns); ax.set_ylim(0, 100)
                        ax.tick_params(labelsize=10)
                        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5),
                                  fontsize=9, frameon=False)
                    axes_ss[-1].set_xlabel("Time (ns)", fontsize=12, fontweight="bold")
                    fig.tight_layout()
                    save_and_show(fig, "secondary_structure_composition", out_dir)

                # ---- Average SS bar chart ----
                if _n_dom > 0:
                    fig, axes_avg = plt.subplots(1, _n_dom, figsize=(max(5, _n_dom * 5), 4),
                                                 squeeze=False)
                    axes_avg = axes_avg.flatten()
                    for ax, (d_idx, label) in zip(axes_avg, _dssp_domains):
                        data_ss = dssp_num[:, d_idx]
                        total_el = data_ss.size
                        avg_h = np.sum(data_ss == 0) / total_el * 100
                        avg_e = np.sum(data_ss == 1) / total_el * 100
                        avg_c = np.sum(data_ss == 2) / total_el * 100
                        bars = ax.bar(["Helix", "Strand", "Coil"],
                                      [avg_h, avg_e, avg_c],
                                      color=["#E91E63", "#2196F3", "#BDBDBD"],
                                      edgecolor="gray", linewidth=0.5)
                        ax.set_ylabel("Average Content (%)",
                                      fontsize=11, fontweight="bold")
                        ax.set_title(label, fontsize=12, fontweight="bold")
                        ax.tick_params(labelsize=10)
                        for bar, val in zip(bars, [avg_h, avg_e, avg_c]):
                            ax.text(bar.get_x() + bar.get_width()/2,
                                    bar.get_height() + 1,
                                    f"{val:.1f}%", ha="center",
                                    fontsize=10, fontweight="bold")
                    fig.tight_layout()
                    save_and_show(fig, "secondary_structure_average", out_dir)

                df_dssp = pd.DataFrame(dssp_all)
                df_dssp.insert(0, "Time_ns", time_array)
                numeric["dssp_num"] = dssp_num.copy()
                numeric["dssp_pep_idx"] = pep_idx
                numeric["dssp_prot_idx"] = prot_idx
                report_data["analyses"]["DSSP"] = {}
                for d_idx, label_ds in [(pep_idx, "Peptide"), (prot_idx, "Protein")]:
                    if d_idx:
                        data_ss = dssp_num[:, d_idx]
                        total_el = data_ss.size
                        report_data["analyses"]["DSSP"][label_ds] = (
                            f"Helix {np.sum(data_ss==0)/total_el*100:.1f}%, "
                            f"Strand {np.sum(data_ss==1)/total_el*100:.1f}%, "
                            f"Coil {np.sum(data_ss==2)/total_el*100:.1f}%")
                report_data["plots"]["DSSP"] = [
                    "secondary_structure_peptide.png",
                    "secondary_structure_protein.png",
                    "secondary_structure_composition.png",
                    "secondary_structure_average.png"]
                df_dssp.to_csv(os.path.join(out_dir, "dssp_data.csv"), index=False)
            except Exception as e:
                st.error(f"DSSP failed: {e}")
        completed += 1

    # ================================================================
    # DISTANCE
    # ================================================================
    if "Distance" in analyses:
        prog.progress(completed / total_analyses, text="Computing distances...")
        with st.expander("\U0001f4cf Distance Analysis", expanded=True):
            pep_grp = u.select_atoms(pep_all)
            prot_grp = u.select_atoms(prot_all)
            dists = np.array([np.linalg.norm(
                pep_grp.center_of_mass() - prot_grp.center_of_mass())
                for _ in u.trajectory])

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time_array, dists, color="#9C27B0", lw=.8, alpha=.7)
            ax.set_xlabel("Time (ns)"); ax.set_ylabel("Distance (\u00c5)")
            ax.set_title(f"{_system_interaction} COM Distance", fontweight="bold")
            ax.set_xlim(0, sim_ns)
            fig.tight_layout()
            save_and_show(fig, "distance_com", out_dir)

            st.metric("Mean distance",
                      f"{mean(dists):.2f} \u00b1 {stdev(dists):.2f} \u00c5")

            pd.DataFrame({"Time_ns": time_array, "COM_dist": dists}).to_csv(
                os.path.join(out_dir, "distance_data.csv"), index=False)
            report_data["analyses"]["Distance"] = {
                "mean": f"{mean(dists):.2f} +/- {stdev(dists):.2f} A"}
            report_data["plots"]["Distance"] = ["distance_com.png"]
            numeric["distance"] = dists.copy()

        completed += 1

    # ================================================================
    # CONTACT  (with timeline heatmap)
    # ================================================================
    if "Contact" in analyses:
        prog.progress(completed / total_analyses, text="Computing contacts...")
        with st.expander("\U0001f91d Contact Analysis", expanded=True):
            contact_prot = u.select_atoms(f"({protein_sel}) and name CA")
            pep_residues = u.select_atoms(pep_all).residues
            contact_records = []
            for fi, _ in enumerate(u.trajectory):
                prot_pos = contact_prot.positions
                for res in pep_residues:
                    d = np.min(distance_array(res.atoms.positions, prot_pos))
                    if d < contact_cutoff:
                        contact_records.append({
                            "frame": fi, "time_ns": time_array[fi],
                            "resid": res.resid, "resname": res.resname,
                            "min_dist": d,
                            "label": f"{res.resname}{res.resid}"})

            if contact_records:
                df_cont = pd.DataFrame(contact_records)

                # ---- Frequency bar chart ----
                freq = df_cont.groupby(["resid", "resname"]).agg(
                    contact_frames=("frame", "count"),
                    mean_dist=("min_dist", "mean")
                ).reset_index()
                freq["contact_fraction"] = freq["contact_frames"] / n_frames * 100
                freq["label"] = freq["resname"] + freq["resid"].astype(str)
                freq = freq.sort_values("contact_fraction", ascending=False)

                fig, ax = plt.subplots(figsize=(12, 5))
                ax.bar(range(len(freq)), freq["contact_fraction"],
                       color="#E91E63", alpha=0.7, edgecolor="#C2185B", lw=0.5)
                ax.set_xticks(range(len(freq)))
                ax.set_xticklabels(freq["label"], rotation=45, ha="right", fontsize=9)
                ax.set_xlabel(f"{_pep_label} Residue", fontsize=12, fontweight="bold")
                ax.set_ylabel("Contact Occupancy (%)", fontsize=12, fontweight="bold")
                ax.set_title(f"{_pep_label} Residues in Contact with {_prot_label} "
                             f"(<{contact_cutoff} \u00c5)",
                             fontsize=13, fontweight="bold")
                ax.tick_params(labelsize=10)
                fig.tight_layout()
                save_and_show(fig, "contact_frequency", out_dir)

                # ---- Contact timeline heatmap (binary matrix) ----
                top_resids = freq.head(15)["resid"].values
                top_labels = freq.head(15)["label"].values

                contact_matrix = np.zeros((len(top_resids), n_frames))
                for i, resid in enumerate(top_resids):
                    frames_in_contact = df_cont[
                        df_cont["resid"] == resid]["frame"].values
                    contact_matrix[i, frames_in_contact] = 1

                fig, ax = plt.subplots(figsize=(14, 6))
                im = ax.imshow(contact_matrix, aspect="auto", cmap="RdPu",
                               interpolation="nearest", origin="lower",
                               extent=[0, sim_ns, -0.5, len(top_resids)-0.5])
                ax.set_yticks(range(len(top_labels)))
                ax.set_yticklabels(top_labels, fontsize=10)
                ax.set_xlabel("Time (ns)", fontsize=12, fontweight="bold")
                ax.set_ylabel(f"{_pep_label} Residue", fontsize=12, fontweight="bold")
                ax.set_title(f"Contact Timeline (<{contact_cutoff} \u00c5)",
                             fontsize=13, fontweight="bold")
                ax.tick_params(labelsize=10)
                cbar = plt.colorbar(im, ax=ax, shrink=0.5)
                cbar.set_ticks([0, 1])
                cbar.set_ticklabels(["No Contact", "Contact"])
                fig.tight_layout()
                save_and_show(fig, "contact_timeline", out_dir)

                st.metric("Unique residues in contact",
                          df_cont["resid"].nunique())
                st.metric("Total contact events", len(df_cont))

                report_data["analyses"]["Contact"] = {
                    "unique_residues": df_cont["resid"].nunique(),
                    "total_events": len(df_cont),
                    "top_contacts": ", ".join(freq.head(5)["label"].values)}
                report_data["plots"]["Contact"] = [
                    "contact_frequency.png", "contact_timeline.png"]
                df_cont.to_csv(os.path.join(out_dir, "contact_data.csv"),
                               index=False)
                freq.to_csv(os.path.join(out_dir, "contact_frequency.csv"),
                            index=False)
            else:
                st.warning("No contacts found.")
            numeric["contact_freq"] = freq.copy() if contact_records else None

        completed += 1

    # ================================================================
    # PROLIF  (tries main topology if no ProLIF PDB uploaded)
    # ================================================================
    if "ProLIF" in analyses:
        prog.progress(completed / total_analyses, text="Computing ProLIF fingerprints...")
        with st.expander("\U0001f9f2 ProLIF Interaction Fingerprints", expanded=True):
            try:
                import prolif as plf

                # Decide which topology to use for ProLIF
                plf_topology_path = plf_path if plf_path else top_path
                if plf_path:
                    st.write("Using uploaded ProLIF topology (PDB with H + elements).")
                else:
                    st.warning("No ProLIF topology uploaded \u2014 using main topology PDB. "
                               "For best results (H-bonds, pi-stacking), upload a PDB "
                               "with explicit hydrogens and correct element column.")

                u_plf = mda.Universe(plf_topology_path, traj_path)

                # Check element quality
                try:
                    elems = u_plf.atoms.elements
                    unique_elems = set(elems)
                    n_H = sum(1 for e in elems if e == "H")
                    st.write(f"ProLIF topology: {len(u_plf.atoms)} atoms, "
                             f"{n_H} hydrogens, "
                             f"elements: {sorted(unique_elems)}")
                    if len(unique_elems) <= 2:
                        st.warning("Few unique elements detected \u2014 element column "
                                   "may be garbled. Interaction detection may be limited.")
                except Exception:
                    st.write(f"ProLIF topology: {len(u_plf.atoms)} atoms")

                # Detect chains / ligand for ProLIF
                if is_holo_ligand:
                    # Protein + Ligand mode: select by residue name
                    pep_plf = u_plf.select_atoms(f"resname {ligand_resname}")
                    prot_plf = u_plf.select_atoms(
                        f"protein and not resname {ligand_resname}")
                    if len(pep_plf) == 0:
                        st.error(f"ProLIF: No atoms found with resname "
                                 f"'{ligand_resname}' in ProLIF topology.")
                        raise ValueError("Ligand not found in ProLIF topology")
                    st.write(f"Ligand ({ligand_resname}): "
                             f"{len(pep_plf)} atoms | "
                             f"Protein: {len(prot_plf)} atoms")
                else:
                    # Peptide + Protein mode: detect chains by resid gaps
                    all_prot_plf = u_plf.select_atoms("protein or resname ACE NME")
                    all_resids_plf = sorted(set(all_prot_plf.resids))
                    chains_plf, cur = [], [all_resids_plf[0]]
                    for i in range(1, len(all_resids_plf)):
                        if all_resids_plf[i] - all_resids_plf[i-1] > 1:
                            chains_plf.append(cur); cur = [all_resids_plf[i]]
                        else:
                            cur.append(all_resids_plf[i])
                    chains_plf.append(cur)

                    if len(chains_plf) >= 2:
                        chains_plf.sort(key=len)
                        pep_r_plf = chains_plf[0]
                        prot_r_plf = chains_plf[-1]
                    else:
                        n_pep = peptide_info["n_residues"]
                        pep_r_plf = list(range(1, n_pep + 1))
                        prot_r_plf = list(range(n_pep + 1, len(all_resids_plf) + 1))

                    pep_sel_plf = (f"resid {min(pep_r_plf)}:{max(pep_r_plf)} "
                                   f"and (protein or resname ACE NME)")
                    prot_sel_plf = (f"resid {min(prot_r_plf)}:{max(prot_r_plf)} "
                                    f"and (protein or resname ACE NME)")
                    pep_plf = u_plf.select_atoms(pep_sel_plf)
                    prot_plf = u_plf.select_atoms(prot_sel_plf)

                    st.write(f"Peptide: resid {min(pep_r_plf)}-{max(pep_r_plf)} "
                             f"({len(pep_plf)} atoms) | "
                             f"Protein: resid {min(prot_r_plf)}-{max(prot_r_plf)} "
                             f"({len(prot_plf)} atoms)")

                # ── Strategy 1: Pre-filter protein to interface residues ──
                # Use MDAnalysis 'byres around' (KD-tree optimized) to keep only
                # protein residues near the peptide/ligand. This reduces the atom
                # group BEFORE ProLIF, cutting RDKit molecule conversion overhead.
                _n_prot_res = len(set(prot_plf.resids))
                if _n_prot_res > 500:
                    st.info(f"Large system ({_n_prot_res} protein residues). "
                            f"Pre-filtering to interface residues...")
                    u_plf.trajectory[0]
                    # Build selection string for the pre-filter query
                    if is_holo_ligand:
                        _prot_sel_for_filter = f"protein and not resname {ligand_resname}"
                    else:
                        _prot_sel_for_filter = prot_sel_plf
                    _interface_prot = u_plf.select_atoms(
                        f"byres (({_prot_sel_for_filter}) and around 7.0 group pep)",
                        pep=pep_plf)
                    _n_interface = len(set(_interface_prot.resids))
                    if _n_interface > 0:
                        prot_plf = _interface_prot
                        st.write(f"Interface pre-filter: {_n_interface} protein residues "
                                 f"within 7 A (was {_n_prot_res}). "
                                 f"Protein atoms: {len(prot_plf)}")
                    else:
                        st.warning("No interface residues found within 7 A. "
                                   "Using all protein residues.")

                interactions = ["HBDonor", "HBAcceptor", "Hydrophobic",
                                "PiStacking", "PiCation", "CationPi",
                                "Anionic", "Cationic", "VdWContact"]
                # vicinity_cutoff: ProLIF per-frame filter.
                # Small systems (<= 500 res): use default (6.0A).
                # Large systems (> 500 res): tighter 5.7A for performance.
                _vcutoff = 5.7 if _n_prot_res > 500 else 6.0
                fp = plf.Fingerprint(interactions=interactions, vicinity_cutoff=_vcutoff)

                # Frame skipping for faster analysis
                traj_slice = u_plf.trajectory[::prolif_frame_skip]
                n_plf_frames = len(traj_slice)
                st.write(f"Analyzing {n_plf_frames} frames "
                         f"(skip={prolif_frame_skip}, total={len(u_plf.trajectory)})")

                # ── Run ProLIF in background thread (avoids gateway timeout) ──
                # The thread runs fp.run() while the main loop updates the UI
                # every 3s, keeping the browser/WebSocket connection alive.
                import threading as _threading

                _prolif_status = st.empty()
                _prolif_result = {"ok": False, "attempt": 0, "error": "", "fp": None}

                def _prolif_worker():
                    _fp = plf.Fingerprint(interactions=interactions, vicinity_cutoff=_vcutoff)
                    for _att, _kw in enumerate([
                        {},
                        {"converter_kwargs": [{"force": True}, {"force": True}]},
                        {"converter_kwargs": [{"inferrer": None, "force": True},
                                              {"inferrer": None, "force": True}]},
                    ]):
                        try:
                            _fp.run(traj_slice, pep_plf, prot_plf, **_kw)
                            _prolif_result["ok"] = True
                            _prolif_result["attempt"] = _att + 1
                            _prolif_result["fp"] = _fp
                            return
                        except Exception as _e:
                            _prolif_result["error"] = (
                                f"{type(_e).__name__}: {str(_e)[:120]}")
                            _fp = plf.Fingerprint(interactions=interactions,
                                                  vicinity_cutoff=_vcutoff)
                    _prolif_result["ok"] = False

                _thread = _threading.Thread(target=_prolif_worker, daemon=True)
                _thread.start()

                # Poll loop: update UI every 3s to keep connection alive
                _t0 = time.time()
                while _thread.is_alive():
                    _elapsed = time.time() - _t0
                    _mins = int(_elapsed) // 60
                    _secs = int(_elapsed) % 60
                    _prolif_status.info(
                        f"ProLIF computing... ({_mins}m {_secs:02d}s elapsed). "
                        f"Large systems may take several minutes.")
                    time.sleep(3)
                _prolif_status.empty()

                ok = _prolif_result["ok"]
                if ok:
                    fp = _prolif_result["fp"]
                    st.write(f"\u2713 Success (attempt {_prolif_result['attempt']})")
                else:
                    st.write(f"\u2717 All attempts failed: {_prolif_result['error']}")

                if ok:
                    fp_df = fp.to_dataframe()
                    occ = (fp_df.sum() / len(fp_df)) * 100
                    occ = occ.sort_values(ascending=False)
                    top = occ.head(prolif_top_n)

                    st.write(f"Fingerprint: {fp_df.shape[0]} frames, "
                             f"{len(fp_df.columns)} unique interaction pairs")
                    itypes = set(c[2] for c in fp_df.columns
                                 if isinstance(c, tuple))
                    st.write("Types: " + ", ".join(sorted(itypes)))

                    if len(top) > 0:
                        labels_f = [f"{c[0]}-{c[1]}\n({c[2]})"
                                    if isinstance(c, tuple) else str(c)
                                    for c in top.index]
                        fig, ax = plt.subplots(
                            figsize=(14, max(4, len(top)*0.5)))
                        ax.barh(range(len(top)), top.values,
                                color=plt.cm.Set2(
                                    np.linspace(0, 1, len(top))))
                        ax.set_yticks(range(len(labels_f)))
                        ax.set_yticklabels(labels_f, fontsize=8)
                        ax.set_xlabel("Occupancy (%)")
                        ax.set_title(f"Top {_system_interaction} Interactions (ProLIF)",
                                     fontweight="bold")
                        ax.invert_yaxis()
                        fig.tight_layout()
                        save_and_show(fig, "prolif_interaction_frequency", out_dir)

                    # Interaction types bar
                    type_occ = {}
                    for col in fp_df.columns:
                        itype = col[2] if isinstance(col, tuple) else str(col)
                        type_occ[itype] = type_occ.get(itype, 0) + fp_df[col].sum()
                    type_df = pd.DataFrame([
                        {"Interaction": k, "Total": v,
                         "Avg_per_Frame": v / len(fp_df)}
                        for k, v in type_occ.items()
                    ]).sort_values("Total", ascending=False)
                    if len(type_df) > 0:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.bar(type_df["Interaction"],
                               type_df["Avg_per_Frame"],
                               color=plt.cm.Pastel1(
                                   np.linspace(0, 1, len(type_df))),
                               edgecolor="gray", lw=0.5)
                        ax.set_xlabel("Interaction Type"); ax.set_ylabel("Avg/Frame")
                        ax.set_title("Interaction Types Summary", fontweight="bold")
                        plt.xticks(rotation=45, ha="right")
                        fig.tight_layout()
                        save_and_show(fig, "prolif_interaction_types", out_dir)

                    # Timeline heatmap
                    top_cols = occ.head(25).index.tolist()
                    top_ifp = fp_df[top_cols].astype(int)
                    tl_labels = [f"{c[0]}-{c[1]} ({c[2]})"
                                 if isinstance(c, tuple) else str(c)
                                 for c in top_cols]
                    if tl_labels:
                        import matplotlib as _mpl_limit
                        _mpl_limit.rcParams['figure.max_open_warning'] = 50
                        # Limit timeline to avoid decompression bomb error
                        _max_tl_frames = min(top_ifp.shape[0], 5000)
                        _tl_skip = max(1, top_ifp.shape[0] // _max_tl_frames)
                        _tl_data = top_ifp.iloc[::_tl_skip]
                        fig, ax = plt.subplots(
                            figsize=(16, max(4, min(len(tl_labels), 25)*0.5)))
                        im = ax.imshow(_tl_data.T.values, aspect="auto",
                                       cmap="YlOrRd",
                                       interpolation="nearest", origin="lower",
                                       extent=[0, sim_ns, -0.5,
                                               len(tl_labels)-0.5])
                        ax.set_yticks(range(len(tl_labels)))
                        ax.set_yticklabels(tl_labels, fontsize=7)
                        ax.set_xlabel("Time (ns)")
                        ax.set_ylabel("Interaction")
                        ax.set_title("Interaction Fingerprint Timeline",
                                     fontweight="bold")
                        cbar = plt.colorbar(im, ax=ax, shrink=0.3)
                        cbar.set_ticks([0, 1])
                        cbar.set_ticklabels(["Absent", "Present"])
                        fig.tight_layout()
                        save_and_show(fig, "prolif_interaction_timeline", out_dir)

                    # Residue interaction map (occupancy = % frames with ANY interaction)
                    res_pair_cols = {}
                    for col in fp_df.columns:
                        if isinstance(col, tuple) and len(col) >= 3:
                            key = (str(col[0]), str(col[1]))
                            if key not in res_pair_cols:
                                res_pair_cols[key] = []
                            res_pair_cols[key].append(col)

                    # True occupancy: fraction of frames with at least one interaction
                    res_pair_occ = {}
                    for key, cols in res_pair_cols.items():
                        any_present = fp_df[cols].any(axis=1).sum()
                        res_pair_occ[key] = any_present / len(fp_df) * 100

                    pep_res_plf = sorted(set(k[0] for k in res_pair_occ))
                    prot_res_plf = sorted(set(k[1] for k in res_pair_occ))
                    if pep_res_plf and prot_res_plf:
                        cmat = np.zeros((len(pep_res_plf), len(prot_res_plf)))
                        for (p, q), occ_val in res_pair_occ.items():
                            cmat[pep_res_plf.index(p),
                                 prot_res_plf.index(q)] = occ_val
                        fig, ax = plt.subplots(
                            figsize=(max(8, len(prot_res_plf)*0.6),
                                     max(4, len(pep_res_plf)*0.6)))
                        im = ax.imshow(cmat, aspect="auto", cmap="hot_r",
                                       interpolation="nearest", origin="lower",
                                       vmin=0, vmax=100)
                        ax.set_xticks(range(len(prot_res_plf)))
                        ax.set_xticklabels(prot_res_plf, rotation=90, fontsize=8)
                        ax.set_yticks(range(len(pep_res_plf)))
                        ax.set_yticklabels(pep_res_plf, fontsize=8)
                        ax.set_xlabel(f"{_prot_label} Residue", fontsize=12, fontweight="bold")
                        ax.set_ylabel(f"{_pep_label} Residue", fontsize=12, fontweight="bold")
                        ax.set_title("Residue\u2013Residue Interaction Map (ProLIF)",
                                     fontsize=13, fontweight="bold")
                        cbar = plt.colorbar(im, ax=ax, shrink=0.5)
                        cbar.set_label("Occupancy (%)")
                        fig.tight_layout()
                        save_and_show(fig, "prolif_residue_interaction_map", out_dir)

                    # Per-residue interaction profiles (ANY interaction occupancy)
                    pep_any = {}
                    prot_any = {}
                    for key, cols in res_pair_cols.items():
                        pep_r, prot_r = key
                        if pep_r not in pep_any:
                            pep_any[pep_r] = set()
                        if prot_r not in prot_any:
                            prot_any[prot_r] = set()
                        pep_any[pep_r].update(cols)
                        prot_any[prot_r].update(cols)

                    # Compute per-residue occupancy: % frames with any interaction
                    pep_occ = {r: fp_df[list(cols)].any(axis=1).sum() / len(fp_df) * 100
                               for r, cols in pep_any.items()}
                    prot_occ = {r: fp_df[list(cols)].any(axis=1).sum() / len(fp_df) * 100
                                for r, cols in prot_any.items()}

                    if pep_occ or prot_occ:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                        if pep_occ:
                            ps = sorted(pep_occ.items(), key=lambda x: x[1], reverse=True)
                            pn, po = zip(*ps)
                            ax1.bar(range(len(pn)), po, color="#E91E63", alpha=0.7)
                            ax1.set_xticks(range(len(pn)))
                            ax1.set_xticklabels(pn, rotation=45, ha="right", fontsize=8)
                            ax1.set_xlabel(f"{_pep_label} Residue")
                            ax1.set_ylabel("Interaction Occupancy (%)")
                            ax1.set_title(f"{_pep_label} Residue Profile", fontweight="bold")
                            ax1.set_ylim(0, 105)
                        if prot_occ:
                            qs = sorted(prot_occ.items(), key=lambda x: x[1], reverse=True)[:30]
                            qn, qo = zip(*qs)
                            ax2.bar(range(len(qn)), qo, color="#2196F3", alpha=0.7)
                            ax2.set_xticks(range(len(qn)))
                            ax2.set_xticklabels(qn, rotation=45, ha="right", fontsize=8)
                            ax2.set_xlabel(f"{_prot_label} Residue")
                            ax2.set_ylabel("Interaction Occupancy (%)")
                            ax2.set_title(f"{_prot_label} Residue Profile", fontweight="bold")
                            ax2.set_ylim(0, 105)
                        fig.tight_layout()
                        save_and_show(fig, "prolif_residue_profiles", out_dir)

                    fp_df.to_csv(os.path.join(out_dir, "prolif_fingerprint_data.csv"),
                                 index=False)
                    occ.to_csv(os.path.join(out_dir, "prolif_occupancy.csv"))
                    report_data["analyses"]["ProLIF"] = {
                        "unique_pairs": len(fp_df.columns),
                        "interaction_types": ", ".join(sorted(itypes)),
                        "top_interactions": ", ".join(
                            f"{c[0]}-{c[1]} ({c[2]})" if isinstance(c, tuple) else str(c)
                            for c in occ.head(5).index)}
                    report_data["plots"]["ProLIF"] = [
                        "prolif_interaction_frequency.png",
                        "prolif_interaction_timeline.png",
                        "prolif_residue_interaction_map.png"]
                    type_df.to_csv(os.path.join(out_dir, "prolif_interaction_types.csv"),
                                   index=False)
                else:
                    st.error("ProLIF: all 3 inference attempts failed. "
                             "Upload a PDB with correct element column and H atoms.")
            except Exception as e:
                st.error(f"ProLIF failed: {e}")
            numeric["prolif_occ"] = occ.copy() if ok else None

        completed += 1

    # ================================================================
    # eRMSF  (Complex heatmap only — with boundary line)
    # ================================================================
    if "eRMSF" in analyses:
        prog.progress(completed / total_analyses, text="Computing eRMSF...")
        with st.expander("\U0001f321\ufe0f eRMSF \u2014 Ensemble RMSF (ermsfkit)", expanded=True):
            try:
                from eRMSF import ermsfkit as eRMSF_kit

                # Align to average
                avg = align.AverageStructure(u, u, select=cx_ca, ref_frame=0).run()
                align.AlignTraj(u, avg.results.universe, select=cx_ca,
                                in_memory=True).run()

                n_seg = n_frames // ermsf_skip
                seg_time = np.linspace(0, sim_ns, n_seg)

                ermsf_results = {}
                if is_holo:
                    _ermsf_items = [(_prot_label, prot_ca),
                                    (_pep_label, pep_ca),
                                    ("Complex", cx_ca)]
                else:
                    _apo_lbl = "Protein" if system_type == "APO Protein" else "Peptide"
                    _ermsf_items = [(_apo_lbl, cx_ca), ("Complex", cx_ca)]
                for lbl, sel in _ermsf_items:
                    atoms = u.select_atoms(sel)
                    er = eRMSF_kit(atoms, skip=ermsf_skip, reference_frame=0)
                    er.run()
                    mat = er.results.ermsf
                    if is_holo_ligand and sel in (pep_ca, cx_ca):
                        # Per-atom data: ligand has multiple heavy atoms per residue
                        # Use sequential indices and atom-level labels
                        resids = np.arange(len(atoms))
                        resnames = atoms.names
                        labels_r = [f"{atoms.resnames[i]}{atoms.resids[i]}:{atoms.names[i]}"
                                    for i in range(len(atoms))]
                    else:
                        resids = atoms.residues.resids
                        resnames = atoms.residues.resnames
                        labels_r = [f"{rn}{ri}" for rn, ri in zip(resnames, resids)]
                    ermsf_results[lbl] = dict(
                        matrix=mat, resids=resids,
                        resnames=resnames, labels=labels_r)

                # ---- Complex heatmap (with peptide/protein boundary) ----
                data_cx = ermsf_results["Complex"]
                mat_cx = data_cx["matrix"]
                fig, ax = plt.subplots(figsize=(14, 6))
                im = ax.imshow(mat_cx, cmap="viridis", aspect="auto",
                               vmin=ermsf_vmin, vmax=ermsf_vmax,
                               origin="lower", interpolation="bicubic",
                               extent=[0, sim_ns, -0.5, mat_cx.shape[0]-0.5])
                if is_holo_ligand and _prot_label in ermsf_results:
                    # Ligand appears AFTER protein in topology order
                    n_prot_ermsf = ermsf_results[_prot_label]["matrix"].shape[0]
                    ax.axhline(y=n_prot_ermsf - 0.5, color="white", ls="--",
                               lw=1.5, alpha=0.8)
                    ax.text(sim_ns * 0.02, n_prot_ermsf + 1, f"\u2190 {_pep_label}",
                            color="white", fontsize=9, fontweight="bold", va="bottom")
                    ax.text(sim_ns * 0.02, n_prot_ermsf - 2, f"{_prot_label} \u2192",
                            color="white", fontsize=9, fontweight="bold", va="top")
                elif is_holo and _pep_label in ermsf_results:
                    # Peptide appears BEFORE protein in topology order
                    n_pep_ermsf = ermsf_results[_pep_label]["matrix"].shape[0]
                    ax.axhline(y=n_pep_ermsf - 0.5, color="white", ls="--",
                               lw=1.5, alpha=0.8)
                    ax.text(sim_ns * 0.02, n_pep_ermsf + 1, f"\u2190 {_prot_label}",
                            color="white", fontsize=9, fontweight="bold", va="bottom")
                    ax.text(sim_ns * 0.02, n_pep_ermsf - 2, f"{_pep_label} \u2192",
                            color="white", fontsize=9, fontweight="bold", va="top")
                n_cx = mat_cx.shape[0]
                step_cx = max(1, n_cx // 20)
                ticks_cx = list(range(0, n_cx, step_cx))
                ax.set_yticks(ticks_cx)
                ax.set_yticklabels([data_cx["labels"][i] for i in ticks_cx],
                                   fontsize=7)
                ax.set_xlabel("Time (ns)", fontsize=12, fontweight="bold")
                ax.set_ylabel("Residue", fontsize=12, fontweight="bold")
                if is_holo:
                    _ermsf_title = f"eRMSF \u2014 Full Complex ({_pep_label} + {_prot_label})"
                else:
                    _apo_lbl = "Protein" if system_type == "APO Protein" else "Peptide"
                    _ermsf_title = f"eRMSF \u2014 APO {_apo_lbl}"
                ax.set_title(_ermsf_title, fontsize=14, fontweight="bold")
                cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
                cbar.set_label("RMSF ($\\AA$)", fontsize=12, fontweight="bold")
                cbar.ax.tick_params(labelsize=10)
                fig.tight_layout()
                save_and_show(fig, "ermsf_heatmap_complex", out_dir)

                # ---- eRMSF vs RMSF comparison ----
                if is_holo:
                    _comp_items = [(_prot_label, "#2196F3"), (_pep_label, "#E91E63")]
                else:
                    _apo_lbl = "Protein" if system_type == "APO Protein" else "Peptide"
                    _apo_col = "#2196F3" if system_type == "APO Protein" else "#E91E63"
                    _comp_items = [(_apo_lbl, _apo_col)]
                fig, axes = plt.subplots(len(_comp_items), 1,
                                         figsize=(14, max(4, len(_comp_items)*4)),
                                         squeeze=False)
                axes = axes.flatten()
                for ax, (lbl, col) in zip(axes, _comp_items):
                    d = ermsf_results[lbl]
                    m = d["matrix"].mean(axis=1)
                    s = d["matrix"].std(axis=1)
                    atoms = u.select_atoms(prot_ca if lbl == _prot_label else pep_ca)
                    trad = MDA_RMSF(atoms).run().results.rmsf
                    ax.fill_between(d["resids"], m-s, m+s, alpha=.2, color=col,
                                    label="eRMSF \u00b1 \u03c3")
                    ax.plot(d["resids"], m, color=col, lw=1.5,
                            label=f"eRMSF mean ({m.mean():.2f} \u00c5)")
                    ax.plot(d["resids"], trad, "k--", lw=1, alpha=.7,
                            label=f"Traditional RMSF ({trad.mean():.2f} \u00c5)")
                    ax.set_ylabel("RMSF ($\\AA$)")
                    ax.set_title(f"{lbl} \u2014 eRMSF vs Traditional RMSF",
                                 fontweight="bold")
                    ax.legend(fontsize=9)
                    ax.set_xlim(d["resids"][0], d["resids"][-1])
                    top3 = np.argsort(m)[-3:]
                    for idx in top3:
                        ax.annotate(d["labels"][idx],
                                    xy=(d["resids"][idx], m[idx]),
                                    fontsize=7, fontweight="bold", color=col,
                                    ha="center", va="bottom")
                _ermsf_x_label = "Atom Index" if is_holo_ligand else "Residue ID"
                axes[-1].set_xlabel(_ermsf_x_label)
                fig.tight_layout()
                save_and_show(fig, "ermsf_vs_rmsf_comparison", out_dir)

                # ---- Top 5 flexible residues over time ----
                for lbl, col in _comp_items:
                    d = ermsf_results[lbl]
                    m = d["matrix"].mean(axis=1)
                    top5 = np.argsort(m)[-5:][::-1]
                    fig, ax = plt.subplots(figsize=(12, 4))
                    for idx in top5:
                        ax.plot(seg_time, d["matrix"][idx, :], lw=1.2,
                                alpha=0.8, label=d["labels"][idx])
                    ax.set_xlabel("Time (ns)", fontsize=12, fontweight="bold")
                    ax.set_ylabel("eRMSF ($\\AA$)", fontsize=12, fontweight="bold")
                    ax.set_title(f"{lbl} \u2014 Top 5 Flexible Residues Over Time",
                                 fontweight="bold")
                    ax.legend(fontsize=9, loc="best", ncol=2)
                    ax.set_xlim(0, sim_ns)
                    fig.tight_layout()
                    save_and_show(fig, f"ermsf_top_residues_{lbl.lower()}", out_dir)

                # Save CSV data
                for lbl in ermsf_results:
                    d = ermsf_results[lbl]
                    cols_e = [f"seg_{i+1}" for i in range(d["matrix"].shape[1])]
                    df_e = pd.DataFrame(d["matrix"], columns=cols_e)
                    df_e.insert(0, "resid", d["resids"])
                    df_e.insert(1, "resname", d["resnames"])
                    df_e.insert(2, "label", d["labels"])
                    df_e.insert(3, "ermsf_mean", d["matrix"].mean(axis=1))
                    df_e.insert(4, "ermsf_std", d["matrix"].std(axis=1))
                    df_e.to_csv(os.path.join(out_dir, f"ermsf_data_{lbl.lower()}.csv"),
                                index=False)
                report_data["analyses"]["eRMSF"] = {
                    lbl: f"mean {d['matrix'].mean():.2f} A, max {d['matrix'].max():.2f} A"
                    for lbl, d in ermsf_results.items()}
                report_data["plots"]["eRMSF"] = [
                    "ermsf_heatmap_complex.png",
                    "ermsf_vs_rmsf_comparison.png"]

                st.success("eRMSF complete.")
            except Exception as e:
                st.error(f"eRMSF failed: {e}")
            numeric["ermsf"] = {lbl: d["matrix"].copy() for lbl, d in ermsf_results.items()}
            numeric["ermsf_labels"] = {lbl: d["labels"] for lbl, d in ermsf_results.items()}
            numeric["ermsf_resids"] = {lbl: d["resids"].copy() for lbl, d in ermsf_results.items()}

        completed += 1


    # ================================================================
    # FREE ENERGY LANDSCAPE + REPRESENTATIVE FRAME EXTRACTION
    # ================================================================
    if "Free Energy Landscape" in analyses:
        prog.progress(completed / total_analyses, text="Computing Free Energy Landscape...")
        with st.expander("\U0001f30b Free Energy Landscape", expanded=True):
            try:
                st.write("Computing RMSD and Rg for FEL...")

                # Compute RMSD (complex CA, ref frame 0)
                R_fel = rms.RMSD(u, u, select=cx_ca, ref_frame=0)
                R_fel.run()
                rmsd_fel = R_fel.results.rmsd[:, 2]

                # Compute Rg (complex)
                cx_atoms_fel = u.select_atoms(cx_ca)
                rg_fel = np.array([cx_atoms_fel.radius_of_gyration()
                                   for _ in u.trajectory])

                # ---- Compute FEL ----
                kB = 0.001987  # kcal/(mol*K)
                RT = kB * fel_temperature

                hist, xedges, yedges = np.histogram2d(
                    rmsd_fel, rg_fel, bins=fel_bins)
                hist_prob = hist / np.sum(hist)
                hist_prob[hist_prob == 0] = np.min(
                    hist_prob[hist_prob > 0]) * 0.01
                fel = -RT * np.log(hist_prob)
                fel = fel - np.min(fel)

                # Smooth
                fel_smooth = _gaussian_filter(fel, sigma=1.0)

                # Find global minimum
                min_idx_2d = np.unravel_index(
                    np.argmin(fel_smooth), fel_smooth.shape)
                min_rmsd = xedges[min_idx_2d[0]]
                min_rg = yedges[min_idx_2d[1]]

                # Closest trajectory frame
                dist_to_min = np.sqrt(
                    (rmsd_fel - min_rmsd)**2 + (rg_fel - min_rg)**2)
                min_frame_idx = int(np.argmin(dist_to_min))
                min_time_ns = time_array[min_frame_idx]
                actual_rmsd = rmsd_fel[min_frame_idx]
                actual_rg = rg_fel[min_frame_idx]

                # ---- Plot FEL ----
                fig, ax = plt.subplots(figsize=(8, 6))
                X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
                levels = np.linspace(0, np.percentile(fel_smooth, 95), 20)
                contour = ax.contourf(X, Y, fel_smooth.T,
                                      levels=levels, cmap="viridis")
                ax.contour(X, Y, fel_smooth.T, levels=levels,
                           colors="white", linewidths=0.5, alpha=0.3)
                cbar = plt.colorbar(contour, ax=ax)
                cbar.set_label("Free Energy (kcal/mol)",
                               fontsize=12, fontweight="bold")
                ax.plot(xedges[min_idx_2d[0]], yedges[min_idx_2d[1]],
                        "r*", markersize=20, label="Global Minimum",
                        markeredgecolor="white", markeredgewidth=1)
                ax.plot(actual_rmsd, actual_rg, "yo", markersize=12,
                        label=f"Frame {min_frame_idx} ({min_time_ns:.1f} ns)",
                        markeredgecolor="black", markeredgewidth=1.5)
                ax.set_xlabel("RMSD (\u00c5)", fontsize=12, fontweight="bold")
                ax.set_ylabel("Radius of Gyration (\u00c5)",
                              fontsize=12, fontweight="bold")
                ax.set_title("Free Energy Landscape (RMSD vs Rg)",
                             fontsize=13, fontweight="bold")
                ax.legend(fontsize=9)
                fig.tight_layout()
                save_and_show(fig, "free_energy_landscape", out_dir)

                # ---- Extract representative frame as PDB ----
                st.write(f"Extracting representative frame {min_frame_idx} "
                         f"({min_time_ns:.1f} ns)...")
                u.trajectory[min_frame_idx]
                if fel_extract_selection == "All atoms":
                    extract_atoms = u.atoms
                else:
                    extract_atoms = u.select_atoms(
                        f"({peptide_sel}) or ({protein_sel})")
                pdb_name = f"representative_frame_{min_frame_idx}.pdb"
                pdb_path = os.path.join(out_dir, pdb_name)
                extract_atoms.write(pdb_path)
                u.trajectory[0]  # Reset

                st.success(f"Representative frame extracted: "
                           f"frame {min_frame_idx} at {min_time_ns:.1f} ns")
                st.markdown(
                    f"**FEL Global Minimum**: RMSD = {min_rmsd:.2f} \u00c5, "
                    f"Rg = {min_rg:.2f} \u00c5\n\n"
                    f"**Closest Frame**: #{min_frame_idx} at {min_time_ns:.1f} ns "
                    f"(RMSD = {actual_rmsd:.2f} \u00c5, Rg = {actual_rg:.2f} \u00c5)")

                # Note: FEL is NOT added to report_data (user request)
                report_data["plots"]["Free Energy Landscape"] = [
                    "free_energy_landscape.png"]

                # Save CSV
                pd.DataFrame({
                    "RMSD": rmsd_fel, "Rg": rg_fel,
                    "Time_ns": time_array
                }).to_csv(os.path.join(out_dir, "fel_data.csv"), index=False)

                st.success("Free Energy Landscape complete.")
            except Exception as e:
                st.error(f"FEL failed: {e}")
            numeric["fel_rmsd"] = rmsd_fel.copy()
            numeric["fel_rg"] = rg_fel.copy()

        completed += 1


    # ================================================================
    # INTERACTION ENERGY (Coulomb + Lennard-Jones)
    # ================================================================
    if "Interaction Energy" in analyses:
        prog.progress(completed / total_analyses, text="Computing Interaction Energy...")
        with st.expander("\u26a1 Interaction Energy", expanded=True):
            try:
                pep_ie = u.select_atoms(pep_all)
                prot_ie = u.select_atoms(prot_all)

                if ff_path and HAS_PARMED:
                    # ── Accurate mode: parmed with real FF parameters ──
                    st.write("Using FF topology for accurate charges & LJ params...")
                    parm = _parmed.load_file(ff_path)
                    pep_idx_ie = pep_ie.indices
                    prot_idx_ie = prot_ie.indices

                    charges_pep = np.array([parm.atoms[i].charge for i in pep_idx_ie])
                    charges_prot = np.array([parm.atoms[i].charge for i in prot_idx_ie])
                    eps_pep = np.array([parm.atoms[i].epsilon for i in pep_idx_ie])
                    sig_pep = np.array([parm.atoms[i].sigma for i in pep_idx_ie])
                    eps_prot = np.array([parm.atoms[i].epsilon for i in prot_idx_ie])
                    sig_prot = np.array([parm.atoms[i].sigma for i in prot_idx_ie])

                    eps_ij = np.sqrt(np.outer(eps_pep, eps_prot))
                    sig_ij = (np.add.outer(sig_pep, sig_prot)) / 2.0
                    q_ij = np.outer(charges_pep, charges_prot)
                    COULOMB_CONST = 332.0637
                    cutoff_vdw = 12.0
                    cutoff_elec = 12.0
                    diel = 2.0

                    lie_elec_list, lie_vdw_list = [], []
                    prog_ie = st.progress(0, text="Computing interaction energies...")
                    for fi, ts in enumerate(u.trajectory):
                        dists = distance_array(pep_ie.positions, prot_ie.positions)
                        mask_e = (dists < cutoff_elec) & (dists > 0.1)
                        elec = np.where(mask_e, COULOMB_CONST * q_ij / (diel * dists), 0.0)
                        mask_v = (dists < cutoff_vdw) & (dists > 0.1)
                        r6 = np.where(mask_v, (sig_ij / dists) ** 6, 0.0)
                        vdw = np.where(mask_v, eps_ij * (r6**2 - 2.0 * r6), 0.0)
                        lie_elec_list.append(np.sum(elec))
                        lie_vdw_list.append(np.sum(vdw))
                        if fi % max(1, n_frames // 20) == 0:
                            prog_ie.progress(fi / n_frames)
                    prog_ie.progress(1.0, text="Done!")

                    lie_elec = np.array(lie_elec_list)
                    lie_vdw = np.array(lie_vdw_list)
                    lie_total = lie_elec + lie_vdw
                    energy_unit = "kcal/mol"
                    ie_mode = "FF parameters (parmed)"

                else:
                    # ── Simplified mode: assumed charges ──
                    if ff_path and not HAS_PARMED:
                        st.warning("parmed not installed. Using simplified MM energy.")
                    else:
                        st.info("No FF topology uploaded. Using simplified MM energy "
                                "(approximate). Upload .prmtop/.top for accurate results.")

                    COULOMB_CONST = 332.0637
                    q_approx = 0.5
                    epsilon_lj = 0.5  # kcal/mol
                    sigma_lj = 3.5    # Angstrom
                    cutoff = 12.0

                    lie_elec_list, lie_vdw_list = [], []
                    prog_ie = st.progress(0, text="Computing interaction energies...")
                    for fi, ts in enumerate(u.trajectory):
                        dists = distance_array(pep_ie.positions, prot_ie.positions)
                        mask = (dists < cutoff) & (dists > 0.1)
                        elec = np.where(mask, COULOMB_CONST * q_approx**2 / (2.0 * dists), 0.0)
                        sr6 = np.where(mask, (sigma_lj / dists) ** 6, 0.0)
                        vdw = np.where(mask, 4 * epsilon_lj * (sr6**2 - sr6), 0.0)
                        lie_elec_list.append(np.sum(elec))
                        lie_vdw_list.append(np.sum(vdw))
                        if fi % max(1, n_frames // 20) == 0:
                            prog_ie.progress(fi / n_frames)
                    prog_ie.progress(1.0, text="Done!")

                    lie_elec = np.array(lie_elec_list)
                    lie_vdw = np.array(lie_vdw_list)
                    lie_total = lie_elec + lie_vdw
                    energy_unit = "kcal/mol"
                    ie_mode = "Simplified MM (approximate)"

                # ---- Plot: Energy time series ----
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(time_array, lie_total, alpha=0.7, color="blue",
                        lw=0.8, label="Total")
                ax.plot(time_array, lie_elec, alpha=0.7, color="green",
                        lw=0.8, label="Electrostatic")
                ax.plot(time_array, lie_vdw, alpha=0.7, color="red",
                        lw=0.8, label="van der Waals")
                ax.set_xlabel("Time (ns)", fontsize=12, fontweight="bold")
                ax.set_ylabel(f"Energy ({energy_unit})",
                              fontsize=12, fontweight="bold")
                ax.set_title(f"{_system_interaction} Interaction Energy ({ie_mode})",
                             fontsize=13, fontweight="bold")
                ax.set_xlim(0, sim_ns)
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5),
                          fontsize=10, frameon=False)
                ax.tick_params(labelsize=11)
                fig.tight_layout()
                save_and_show(fig, "interaction_energy", out_dir)

                # ---- Plot: Energy distributions ----
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.kdeplot(lie_total, color="blue", fill=True, alpha=0.2,
                            lw=1.0, label="Total", ax=ax)
                sns.kdeplot(lie_elec, color="green", fill=True, alpha=0.2,
                            lw=1.0, label="Electrostatic", ax=ax)
                sns.kdeplot(lie_vdw, color="red", fill=True, alpha=0.2,
                            lw=1.0, label="van der Waals", ax=ax)
                ax.set_xlabel(f"Energy ({energy_unit})",
                              fontsize=11, fontweight="bold")
                ax.set_yticks([]); ax.set_ylabel("")
                ax.legend(fontsize=10, frameon=False)
                for spine in ["top", "right", "left"]:
                    ax.spines[spine].set_visible(False)
                fig.tight_layout()
                save_and_show(fig, "interaction_energy_dist", out_dir)

                # Metrics
                c_ie1, c_ie2, c_ie3 = st.columns(3)
                c_ie1.metric("Total", f"{mean(lie_total):.1f} \u00b1 "
                             f"{stdev(lie_total):.1f} {energy_unit}")
                c_ie2.metric("Electrostatic", f"{mean(lie_elec):.1f} \u00b1 "
                             f"{stdev(lie_elec):.1f} {energy_unit}")
                c_ie3.metric("van der Waals", f"{mean(lie_vdw):.1f} \u00b1 "
                             f"{stdev(lie_vdw):.1f} {energy_unit}")
                st.caption(f"Mode: {ie_mode}")

                # Save CSV
                pd.DataFrame({
                    "Time_ns": time_array,
                    "Total_kcal_mol": lie_total,
                    "Electrostatic_kcal_mol": lie_elec,
                    "VdW_kcal_mol": lie_vdw,
                }).to_csv(os.path.join(out_dir, "interaction_energy_data.csv"),
                          index=False)

                report_data["analyses"]["Interaction Energy"] = {
                    "mode": ie_mode,
                    "total": f"{mean(lie_total):.1f} +/- {stdev(lie_total):.1f} {energy_unit}",
                    "electrostatic": f"{mean(lie_elec):.1f} +/- {stdev(lie_elec):.1f} {energy_unit}",
                    "van_der_waals": f"{mean(lie_vdw):.1f} +/- {stdev(lie_vdw):.1f} {energy_unit}",
                }
                report_data["plots"]["Interaction Energy"] = [
                    "interaction_energy.png", "interaction_energy_dist.png"]

                st.success("Interaction Energy complete.")
            except Exception as e:
                st.error(f"Interaction Energy failed: {e}")
            numeric["ie_elec"] = lie_elec.copy()
            numeric["ie_vdw"] = lie_vdw.copy()
            numeric["ie_total"] = lie_total.copy()
            numeric["ie_mode"] = ie_mode

        completed += 1

    # ================================================================
    # SUMMARY & ZIP DOWNLOAD
    # ================================================================

    # ================================================================
    # AI-GENERATED PDF REPORT (Claude)
    # ================================================================
    if generate_report and (selected_model["provider"] in ("databricks", "ollama") or llm_api_key):
        prog.progress(completed / max(total_analyses, 1), text="Generating AI report...")
        with st.expander("\U0001f4dd AI-Generated PDF Report", expanded=True):
            try:
                if selected_model["provider"] == "ollama" and not selected_model.get("endpoint", "").strip():
                    st.error("\u26a0\ufe0f Please enter a model name in the \'Model name\' field in the sidebar before generating a report.")
                    raise ValueError("Ollama model name not specified.")
                if not HAS_FPDF:
                    st.error("fpdf2 package not installed.")
                    raise ImportError("fpdf2")
                if selected_model["provider"] == "anthropic" and not HAS_ANTHROPIC:
                    st.error("anthropic package not installed.")
                    raise ImportError("anthropic")
                if selected_model["provider"] == "groq" and not HAS_GROQ:
                    st.error("groq package not installed. Run: pip install groq")
                    raise ImportError("groq")
                if selected_model["provider"] == "gemini" and not HAS_GENAI:
                    st.error("google-generativeai package not installed. "
                             "Run: pip install google-generativeai")
                    raise ImportError("google-generativeai")
                if selected_model["provider"] == "databricks":
                    try:
                        from databricks.sdk import WorkspaceClient
                    except ImportError:
                        if not HAS_OPENAI:
                            st.error("Install databricks-sdk or openai: "
                                     "pip install databricks-sdk[openai]")
                            raise ImportError("databricks-sdk or openai")

                # ── Build prompt ──
                metrics_text = []
                metrics_text.append(f"SYSTEM: {report_data['system']['n_atoms']} atoms, "
                    f"{report_data['system']['n_frames']} frames, "
                    f"{report_data['system']['simulation_ns']:.0f} ns simulation")
                metrics_text.append(f"{_pep_label}: {report_data['system']['peptide_sel']} "
                    f"({report_data['system']['peptide_n_res']} residues)")
                metrics_text.append(f"{_prot_label}: {report_data['system']['protein_sel']} "
                    f"({report_data['system']['protein_n_res']} residues)")
                metrics_text.append("")
                for analysis_name, metrics in report_data["analyses"].items():
                    metrics_text.append(f"--- {analysis_name} ---")
                    if isinstance(metrics, dict):
                        for k, v in metrics.items():
                            metrics_text.append(f"  {k}: {v}")
                    else:
                        metrics_text.append(f"  {metrics}")
                    metrics_text.append("")

                # Build system description for prompt
                if is_holo_ligand:
                    _system_desc = (
                        "a protein-ligand complex simulation. "
                        "The system contains a protein and a small-molecule ligand "
                        f"(residue name: {ligand_resname}). Analyses labeled 'Ligand' "
                        "refer to the small molecule, and 'Protein' to the receptor.")
                    _exec_hint = "(2-3 paragraphs: key findings, binding stability, ligand dynamics)"
                elif is_holo:
                    if peptide_info["n_residues"] > 40:
                        _system_desc = "a protein-protein complex simulation (two protein chains)"
                    else:
                        _system_desc = "a protein-peptide complex simulation"
                    _exec_hint = "(2-3 paragraphs: key findings, system stability, binding characteristics)"
                else:
                    _apo_label = "Protein" if system_type == "APO Protein" else "Peptide"
                    _system_desc = f"an APO {_apo_label.lower()} simulation (single chain, no peptide-protein interaction)"
                    _exec_hint = "(2-3 paragraphs: key findings, structural stability, conformational dynamics)"

                prompt = (
                    f"You are a computational biochemist writing a professional molecular "
                    f"dynamics analysis report for {_system_desc}. "
                    "Write a comprehensive report based on these analysis results.\n\n"
                    + "\n".join(metrics_text) +
                    "\n\nGenerate the report with these EXACT section headers (use ## for each):\n"
                    "## EXECUTIVE SUMMARY\n"
                    f"{_exec_hint}\n\n"
                )
                # Map analysis names to descriptive hints for the LLM
                _analysis_hints = {
                    "ProLIF": "ProLIF Protein-Ligand Interaction Fingerprints analysis. "
                        "Discuss the key residue-residue interactions, dominant interaction types "
                        "(H-bonds, hydrophobic, pi-stacking, etc.), occupancy percentages, "
                        "and which peptide residues are most engaged with the protein binding site.",
                    "eRMSF": "ensemble RMSF (eRMSF) analysis using ermsfkit. "
                        "Discuss how residue flexibility evolves over the simulation time segments, "
                        "compare eRMSF to traditional RMSF, identify regions with time-dependent "
                        "flexibility changes, and highlight the most flexible residues.",
                    "3D Visualization": "Discuss the COM displacement, drift, and "
                        "fluctuation values. Comment on structural mobility and positional stability "
                        "of the system over the simulation.",
                    "Interaction Energy": "Peptide-protein interaction energy analysis "
                        "(Coulomb electrostatics + Lennard-Jones van der Waals). "
                        "Discuss the total, electrostatic, and vdW energy components. "
                        "Comment on whether the interaction is favorable (negative energy), "
                        "which component dominates (electrostatic vs vdW), energy stability "
                        "over time, and what the energy magnitudes suggest about binding affinity. "
                        "Note whether accurate FF parameters or simplified MM was used.",
                }
                for analysis_name in report_data["analyses"]:
                    hint = _analysis_hints.get(analysis_name,
                        "2-3 paragraphs of scientific interpretation")
                    prompt += f"## {analysis_name.upper()}\n({hint})\n\n"
                prompt += (
                    "## CONCLUSIONS AND RECOMMENDATIONS\n"
                    "(Key findings, significance, suggestions for further analysis)\n\n"
                    "Use precise language. Reference specific values from the data. "
                    "Write for a scientific audience. "
                    "Use markdown formatting: **bold** for key terms, *italic* for emphasis, "
                    "bullet lists with - for enumerations. "
                    "For tables use simple markdown table format (| col1 | col2 |). "
                    "Avoid using special Unicode characters like Delta, minus sign (use ASCII - instead)."
                )

                # ── Prepare key images for vision models ──
                key_images = ["rmsd_all.png", "rmsf_combined.png",
                              "ermsf_heatmap_complex.png", "contact_timeline.png",
                              "interaction_energy.png"]
                image_paths = [os.path.join(out_dir, img) for img in key_images
                               if os.path.exists(os.path.join(out_dir, img))]

                # ── Call LLM ──
                model_label = selected_model["label"]
                st.write(f"Calling {model_label}...")
                report_text = call_llm(
                    prompt=prompt,
                    model_config=selected_model,
                    api_key=llm_api_key,
                    image_paths=image_paths,
                    max_tokens=8192,
                )
                st.write(f"Report generated successfully with {model_label}!")

                # ── Parse sections (normalize keys to UPPERCASE for reliable matching) ──
                sections = {}
                current_section = "PREAMBLE"
                current_text = []
                for rline in report_text.split("\n"):
                    if rline.startswith("## "):
                        if current_text:
                            sections[current_section] = "\n".join(current_text).strip()
                        current_section = rline[3:].strip().upper()
                        current_text = []
                    else:
                        current_text.append(rline)
                if current_text:
                    sections[current_section] = "\n".join(current_text).strip()

                # ── Generate PDF ──
                st.write("Building PDF...")
                pdf = FPDF()
                _register_fonts(pdf)
                pdf.set_auto_page_break(auto=True, margin=20)

                # Cover page
                pdf.add_page()
                pdf.set_font(_FONT_NAME, "B", 28)
                pdf.ln(20)
                # Logo on cover
                if _HAS_LOGO:
                    try:
                        pdf.image(_LOGO_PATH, x=75, w=60)
                        pdf.ln(5)
                    except Exception:
                        pass
                pdf.cell(0, 15, "PepIntProt (PIP)", ln=True, align="C")
                pdf.set_font(_FONT_NAME, "", 16)
                if is_holo_ligand:
                    _cover_subtitle = "Protein-Ligand Interaction Analysis Report"
                elif is_holo and peptide_info["n_residues"] > 40:
                    _cover_subtitle = "Protein-Protein Interaction Analysis Report"
                elif is_holo:
                    _cover_subtitle = "Peptide-Protein Interaction Analysis Report"
                else:
                    _apo_lbl_cover = "Protein" if system_type == "APO Protein" else "Peptide"
                    _cover_subtitle = f"APO {_apo_lbl_cover} Structural Analysis Report"
                pdf.cell(0, 10, _cover_subtitle, ln=True, align="C")
                pdf.ln(10)
                pdf.set_font(_FONT_NAME, "", 11)
                pdf.cell(0, 7, f"System: {report_data['system']['n_atoms']} atoms, "
                    f"{report_data['system']['n_frames']} frames, "
                    f"{report_data['system']['simulation_ns']:.0f} ns", ln=True, align="C")
                pdf.cell(0, 7, f"{_pep_label}: {report_data['system']['peptide_n_res']} residues | "
                    f"{_prot_label}: {report_data['system']['protein_n_res']} residues",
                    ln=True, align="C")
                pdf.ln(5)
                import datetime
                pdf.cell(0, 7, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    ln=True, align="C")
                pdf.cell(0, 7, f"AI analysis by {selected_model['label']}", ln=True, align="C")

                # Helper to add markdown-aware wrapped text to PDF
                _add_text = _add_text_to_pdf

                # Helper to add image fitted to page
                def _add_image(pdf_obj, img_path, max_w=180):
                    if os.path.exists(img_path):
                        try:
                            pdf_obj.ln(3)
                            pdf_obj.image(img_path, x=15, w=max_w)
                            pdf_obj.ln(5)
                        except Exception:
                            pass

                # Executive Summary
                if "EXECUTIVE SUMMARY" in sections:
                    pdf.add_page()
                    pdf.set_font(_FONT_NAME, "B", 18)
                    pdf.cell(0, 10, "Executive Summary", ln=True)
                    pdf.ln(3)
                    _add_text(pdf, sections["EXECUTIVE SUMMARY"])

                # Analysis sections
                for analysis_name in report_data["analyses"]:
                    pdf.add_page()
                    pdf.set_font(_FONT_NAME, "B", 16)
                    pdf.cell(0, 10, _sanitize_for_pdf(analysis_name), ln=True)
                    pdf.ln(2)

                    # LLM text for this section (with fuzzy fallback)
                    section_key = analysis_name.upper()
                    matched_text = None
                    if section_key in sections:
                        matched_text = sections[section_key]
                    else:
                        # Fuzzy match: check if any section key contains our key or vice versa
                        for sk, sv in sections.items():
                            if section_key in sk or sk in section_key:
                                matched_text = sv
                                break
                        if matched_text is None:
                            # Try matching first word (e.g. "PROLIF" in "PROLIF INTERACTION...")
                            first_word = section_key.split()[0] if section_key else ""
                            for sk, sv in sections.items():
                                if first_word and sk.startswith(first_word):
                                    matched_text = sv
                                    break
                    if matched_text:
                        _add_text(pdf, matched_text)

                    # Embed plots
                    plot_files = report_data["plots"].get(analysis_name, [])
                    for pf in plot_files:
                        _add_image(pdf, os.path.join(out_dir, pf))

                # Conclusions
                if "CONCLUSIONS AND RECOMMENDATIONS" in sections:
                    pdf.add_page()
                    pdf.set_font(_FONT_NAME, "B", 18)
                    pdf.cell(0, 10, "Conclusions and Recommendations", ln=True)
                    pdf.ln(3)
                    _add_text(pdf, sections["CONCLUSIONS AND RECOMMENDATIONS"])

                # Save PDF
                pdf_path = os.path.join(out_dir, "PepIntProt_Report.pdf")
                pdf.output(pdf_path)
                pdf_size = os.path.getsize(pdf_path) / (1024 * 1024)
                st.success(f"PDF report generated ({pdf_size:.1f} MB)")

                # Show executive summary
                if "EXECUTIVE SUMMARY" in sections:
                    st.markdown("**Executive Summary:**")
                    st.markdown(sections["EXECUTIVE SUMMARY"])

                # Full report in expandable section
                with st.expander("\U0001f4d6 Full AI Report (Markdown)", expanded=False):
                    st.markdown(report_text)

                # Store PDF in session_state (persists across reruns)
                with open(pdf_path, "rb") as f_pdf:
                    _pdf_bytes = f_pdf.read()
                    _pdf_key = f"pdf{rep_label.replace(' ', '_')}"
                    st.session_state[f"pdf_data_{_pdf_key}"] = _pdf_bytes
                    st.session_state[f"pdf_name_{_pdf_key}"] = f"PepIntProt_Report{rep_label.replace(' ', '_')}.pdf"
                    if "all_pdf_keys" not in st.session_state:
                        st.session_state["all_pdf_keys"] = []
                    if _pdf_key not in st.session_state["all_pdf_keys"]:
                        st.session_state["all_pdf_keys"].append(_pdf_key)
            except Exception as e:
                st.error(f"Report generation failed: {e}")
                st.info("All plots and CSV data are still available in the ZIP download below.")
    elif generate_report and selected_model["provider"] in ("anthropic", "groq", "openai", "gemini") and not llm_api_key:
        st.warning(f"Enter your {selected_model['provider'].title()} API key in the sidebar to generate the AI report.")

    prog.progress(1.0, text="Done!")
    st.success(f"\u2705 All analyses complete{rep_label}!")

    with st.expander("\U0001f4c2 Generated Files & Download", expanded=True):
        files = sorted(os.listdir(out_dir))
        st.write(f"**{len(files)} files** generated")
        for f in files:
            sz = os.path.getsize(os.path.join(out_dir, f)) / 1024
            st.text(f"  {f}  ({sz:.1f} KB)")

        # Build ZIP and store in session_state (persists across reruns)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(os.path.join(out_dir, f), f)
        _zip_key = f"zip{rep_label.replace(' ', '_')}"
        st.session_state[f"zip_data_{_zip_key}"] = zip_buf.getvalue()
        st.session_state[f"zip_name_{_zip_key}"] = f"PepIntProt_results{rep_label.replace(' ', '_')}.zip"
        if "all_zip_keys" not in st.session_state:
            st.session_state["all_zip_keys"] = []
        if _zip_key not in st.session_state["all_zip_keys"]:
            st.session_state["all_zip_keys"].append(_zip_key)

        # Also store plot images for cached display
        for f in files:
            if f.endswith(".png"):
                _img_store_key = f"result_images{rep_label.replace(' ', '_')}"
                if _img_store_key not in st.session_state:
                    st.session_state[_img_store_key] = {}
                with open(os.path.join(out_dir, f), "rb") as img_f:
                    st.session_state[_img_store_key][f] = img_f.read()

    return report_data, numeric



# ====================================================================
# LOAD TRAJECTORY & RUN
# ====================================================================
if run_btn:
    # Clear previous results from session_state
    for k in list(st.session_state.keys()):
        if k.startswith(("zip_data_", "zip_name_", "pdf_data_", "pdf_name_",
                         "all_zip_keys", "all_pdf_keys", "result_images")):
            del st.session_state[k]
    st.session_state["analysis_done"] = False
    # ── Validate uploads ──
    if top_up is None:
        st.error("Please upload a **topology PDB** file.")
        st.stop()
    for _ri in range(n_replicas):
        if traj_uploads[_ri] is None:
            _lbl = f" for Replica {_ri+1}" if n_replicas > 1 else ""
            st.error(f"Please upload a **trajectory** file{_lbl}.")
            st.stop()
    if not analyses:
        st.error("Please select at least one **analysis to run** in the sidebar.")
        st.stop()

    # ── Temp directory ──
    # Clean up previous run's temp directory if it exists
    _prev_tmp = st.session_state.get("tmp_dir")
    if _prev_tmp and os.path.isdir(_prev_tmp):
        shutil.rmtree(_prev_tmp, ignore_errors=True)
    tmp_dir = tempfile.mkdtemp(prefix="pepintprot_")
    st.session_state["tmp_dir"] = tmp_dir

    # Save shared files
    top_path = _save_upload(top_up, tmp_dir)
    plf_path = _save_upload(plf_up, tmp_dir) if plf_up else None
    ff_path = _save_upload(ff_up, tmp_dir) if ff_up else None

    # ── Create tabs for replicas ──
    all_numeric = []
    all_report_data = []
    all_out_dirs = []

    if n_replicas > 1:
        _tab_names = [f"\U0001f4c1 Replica {i+1}" for i in range(n_replicas)] + ["\U0001f4ca Combined Analysis"]
        _rep_tabs = st.tabs(_tab_names)
    else:
        _rep_tabs = None

    for _rep_i in range(n_replicas):
        # Per-replica trajectory (save with unique name to avoid collisions)
        _traj_orig = traj_uploads[_rep_i]
        _traj_name = f"rep{_rep_i+1}_{_traj_orig.name}" if n_replicas > 1 else _traj_orig.name
        _traj_tmp = os.path.join(tmp_dir, _traj_name)
        with open(_traj_tmp, "wb") as _f:
            _f.write(_traj_orig.getbuffer())
        _traj_path = _traj_tmp

        # Per-replica output directory
        if n_replicas > 1:
            _out_dir = os.path.join(tmp_dir, f"results_replica_{_rep_i+1}")
            _rep_label = f" (Replica {_rep_i+1})"
        else:
            _out_dir = os.path.join(tmp_dir, "results")
            _rep_label = ""
        os.makedirs(_out_dir)

        # Choose container for this replica
        if n_replicas > 1:
            _ctx = _rep_tabs[_rep_i]
        else:
            _ctx = st.container()

        with _ctx:
            if n_replicas > 1:
                st.subheader(f"Replica {_rep_i + 1}")
            report_data, numeric = _run_single_replica(
                _traj_path, top_path, plf_path, ff_path, _out_dir, _rep_label)
            all_numeric.append(numeric)
            all_report_data.append(report_data)
            all_out_dirs.append(_out_dir)



    # Mark analysis as done (for persistent downloads)
    st.session_state["analysis_done"] = True

    # ================================================================
    # COMBINED CROSS-REPLICA ANALYSIS (mean +/- std)
    # ================================================================
    if n_replicas > 1:
        with _rep_tabs[-1]:
            st.subheader("\U0001f4ca Cross-Replica Combined Analysis")
            st.caption(f"Aggregated over {n_replicas} replicas (mean \u00b1 std)")

            combined_out_dir = os.path.join(tmp_dir, "results_combined")
            os.makedirs(combined_out_dir, exist_ok=True)

            time_array = all_numeric[0].get("_time_array", make_time_array(
                all_report_data[0]["system"]["n_frames"], simulation_time_ns))
            sim_ns = time_array[-1]
            combined_report = {"system": all_report_data[0]["system"].copy(),
                               "analyses": {}, "plots": {}}
            combined_report["system"]["n_replicas"] = n_replicas

            # ── Display labels for combined analysis (mirrors _run_single_replica) ──
            _pep_n_res = combined_report["system"].get("peptide_n_res", 0)
            if is_holo and _pep_n_res > 40:
                _pep_label = "Protein"
                _prot_label = "Protein Target"
                _system_interaction = "Protein–Protein Target"
            elif is_holo:
                _pep_label = "Peptide"
                _prot_label = "Protein Target"
                _system_interaction = "Peptide–Protein Target"
            else:
                _apo_lbl = "Protein" if system_type == "APO Protein" else "Peptide"
                _pep_label = _apo_lbl
                _prot_label = _apo_lbl
                _system_interaction = f"APO {_apo_lbl}"


            # ──── RMSD (mean +/- std per series) ────
            if all(("rmsd" in n) for n in all_numeric):
                with st.expander("\U0001f4c9 RMSD \u2014 Combined", expanded=True):
                    _rmsd_labels = list(all_numeric[0]["rmsd"].keys())
                    _rmsd_colors = all_numeric[0].get("rmsd_colors", {})
                    _default_cols = {"Protein": "#2196F3", "Peptide": "#E91E63", "Complex": "#4CAF50"}
                    fig, axes = plt.subplots(len(_rmsd_labels), 1,
                                             figsize=(10, max(4, len(_rmsd_labels)*3.3)),
                                             sharex=True, squeeze=False)
                    axes = axes.flatten()
                    for ax, lbl in zip(axes, _rmsd_labels):
                        all_vals = np.array([n["rmsd"][lbl] for n in all_numeric])
                        _mean = np.mean(all_vals, axis=0)
                        _std = np.std(all_vals, axis=0)
                        col = _rmsd_colors.get(lbl, _default_cols.get(lbl, "#333333"))
                        ax.plot(time_array, _mean, color=col, lw=1.2, label=f"{lbl} (mean)")
                        ax.fill_between(time_array, _mean - _std, _mean + _std,
                                        alpha=0.25, color=col, label=f"\u00b1 std")
                        # Individual replicas as thin lines
                        for ri in range(n_replicas):
                            ax.plot(time_array, all_vals[ri], color=col, alpha=0.15, lw=0.4)
                        ax.set_ylabel("RMSD (\u00c5)"); ax.set_title(lbl, fontweight="bold")
                        ax.legend(fontsize=8, loc="upper right")
                        ax.set_xlim(0, sim_ns)
                    axes[-1].set_xlabel("Time (ns)")
                    fig.suptitle(f"RMSD \u2014 Combined ({n_replicas} replicas)",
                                 fontsize=14, fontweight="bold")
                    fig.tight_layout()
                    save_and_show(fig, "combined_rmsd", combined_out_dir)
                    # Save combined RMSD CSV
                    _csv_data = {"Time_ns": time_array}
                    for lbl in _rmsd_labels:
                        all_vals = np.array([n["rmsd"][lbl] for n in all_numeric])
                        _csv_data[f"RMSD_{lbl}_mean"] = np.mean(all_vals, axis=0)
                        _csv_data[f"RMSD_{lbl}_std"] = np.std(all_vals, axis=0)
                    pd.DataFrame(_csv_data).to_csv(
                        os.path.join(combined_out_dir, "combined_rmsd.csv"), index=False)

                    combined_report["analyses"]["RMSD"] = {
                        lbl: f"mean {np.mean([n['rmsd'][lbl] for n in all_numeric]):.2f} "
                             f"+/- {np.std([np.mean(n['rmsd'][lbl]) for n in all_numeric]):.2f} A"
                        for lbl in _rmsd_labels}
                    combined_report["plots"]["RMSD"] = ["combined_rmsd.png"]

            # ──── RMSF (mean +/- std) ────
            if all(("rmsf" in n) for n in all_numeric):
                with st.expander("\U0001f4c8 RMSF \u2014 Combined", expanded=True):
                    _min_len = min(len(n["rmsf"]) for n in all_numeric)
                    all_rmsf = np.array([n["rmsf"][:_min_len] for n in all_numeric])
                    _mean = np.mean(all_rmsf, axis=0)
                    _std = np.std(all_rmsf, axis=0)
                    residue_indices = np.arange(_min_len)

                    if is_holo:
                        _col = "#4CAF50"
                        _title = "RMSF \u2014 Combined (Peptide + Protein)"
                    else:
                        _apo_label = "Protein" if system_type == "APO Protein" else "Peptide"
                        _col = "#2196F3" if system_type == "APO Protein" else "#E91E63"
                        _title = f"RMSF \u2014 Combined APO {_apo_label}"

                    fig, ax = plt.subplots(figsize=(14, 5))
                    ax.plot(residue_indices, _mean, color=_col, lw=1.2, label="Mean")
                    ax.fill_between(residue_indices, _mean - _std, _mean + _std,
                                    alpha=0.25, color=_col, label="\u00b1 std")
                    for ri in range(n_replicas):
                        ax.plot(residue_indices, all_rmsf[ri], color=_col, alpha=0.15, lw=0.3)
                    ax.set_xlabel("Residue Index", fontsize=13, fontweight="bold")
                    ax.set_ylabel("RMSF ($\\AA$)", fontsize=13, fontweight="bold")
                    ax.set_title(_title, fontsize=14, fontweight="bold")
                    ax.legend(fontsize=9)
                    ax.set_xlim(0, _min_len - 1)
                    fig.tight_layout()
                    save_and_show(fig, "combined_rmsf", combined_out_dir)
                    # Save combined RMSF CSV
                    pd.DataFrame({
                        "Residue_Index": residue_indices,
                        "RMSF_mean": _mean, "RMSF_std": _std,
                    }).to_csv(os.path.join(combined_out_dir, "combined_rmsf.csv"), index=False)

                    combined_report["analyses"]["RMSF"] = {
                        "mean_rmsf": f"{np.mean(_mean):.2f} A",
                        "inter_replica_std": f"{np.mean(_std):.2f} A"}
                    combined_report["plots"]["RMSF"] = ["combined_rmsf.png"]

            # ──── Radius of Gyration (mean +/- std) ────
            if all(("rg" in n) for n in all_numeric):
                with st.expander("\U0001f535 Radius of Gyration \u2014 Combined", expanded=True):
                    _rg_labels = list(all_numeric[0]["rg"].keys())
                    _rg_colors = all_numeric[0].get("rg_colors", {})
                    _default_cols = {"Protein": "#2196F3", "Peptide": "#E91E63", "Complex": "#4CAF50"}
                    fig, axes = plt.subplots(len(_rg_labels), 1,
                                             figsize=(10, max(4, len(_rg_labels)*3.3)),
                                             sharex=True, squeeze=False)
                    axes = axes.flatten()
                    for ax, lbl in zip(axes, _rg_labels):
                        all_vals = np.array([n["rg"][lbl] for n in all_numeric])
                        _mean = np.mean(all_vals, axis=0)
                        _std = np.std(all_vals, axis=0)
                        col = _rg_colors.get(lbl, _default_cols.get(lbl, "#333333"))
                        ax.plot(time_array, _mean, color=col, lw=1.2, label=f"{lbl} (mean)")
                        ax.fill_between(time_array, _mean - _std, _mean + _std,
                                        alpha=0.25, color=col, label=f"\u00b1 std")
                        for ri in range(n_replicas):
                            ax.plot(time_array, all_vals[ri], color=col, alpha=0.15, lw=0.4)
                        ax.set_ylabel("Rg (\u00c5)"); ax.set_title(lbl, fontweight="bold")
                        ax.legend(fontsize=8, loc="upper right")
                        ax.set_xlim(0, sim_ns)
                    axes[-1].set_xlabel("Time (ns)")
                    fig.suptitle(f"Radius of Gyration \u2014 Combined ({n_replicas} replicas)",
                                 fontsize=14, fontweight="bold")
                    fig.tight_layout()
                    save_and_show(fig, "combined_rg", combined_out_dir)
                    # Save combined Rg CSV
                    _csv_data = {"Time_ns": time_array}
                    for lbl in _rg_labels:
                        all_vals = np.array([n["rg"][lbl] for n in all_numeric])
                        _csv_data[f"Rg_{lbl}_mean"] = np.mean(all_vals, axis=0)
                        _csv_data[f"Rg_{lbl}_std"] = np.std(all_vals, axis=0)
                    pd.DataFrame(_csv_data).to_csv(
                        os.path.join(combined_out_dir, "combined_rg.csv"), index=False)

                    combined_report["analyses"]["Radius of Gyration"] = {
                        lbl: f"mean {np.mean([np.mean(n['rg'][lbl]) for n in all_numeric]):.2f} "
                             f"+/- {np.std([np.mean(n['rg'][lbl]) for n in all_numeric]):.2f} A"
                        for lbl in _rg_labels}
                    combined_report["plots"]["Radius of Gyration"] = ["combined_rg.png"]

            # ──── Distance (mean +/- std) ────
            if all(("distance" in n) for n in all_numeric):
                with st.expander("\U0001f4cf Distance \u2014 Combined", expanded=True):
                    all_dists = np.array([n["distance"] for n in all_numeric])
                    _mean = np.mean(all_dists, axis=0)
                    _std = np.std(all_dists, axis=0)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(time_array, _mean, color="#9C27B0", lw=1.2, label="Mean")
                    ax.fill_between(time_array, _mean - _std, _mean + _std,
                                    alpha=0.25, color="#9C27B0", label="\u00b1 std")
                    for ri in range(n_replicas):
                        ax.plot(time_array, all_dists[ri], color="#9C27B0", alpha=0.15, lw=0.3)
                    ax.set_xlabel("Time (ns)"); ax.set_ylabel("Distance (\u00c5)")
                    ax.set_title(f"{_system_interaction} COM Distance — Combined ({n_replicas} replicas)",
                                 fontweight="bold")
                    ax.legend(fontsize=9); ax.set_xlim(0, sim_ns)
                    fig.tight_layout()
                    save_and_show(fig, "combined_distance", combined_out_dir)
                    # Save combined Distance CSV
                    pd.DataFrame({
                        "Time_ns": time_array,
                        "Distance_mean": _mean, "Distance_std": _std,
                    }).to_csv(os.path.join(combined_out_dir, "combined_distance.csv"), index=False)

                    combined_report["analyses"]["Distance"] = {
                        "mean": f"{np.mean(_mean):.2f} +/- {np.std([np.mean(n['distance']) for n in all_numeric]):.2f} A"}
                    combined_report["plots"]["Distance"] = ["combined_distance.png"]

            # ──── Contact (average frequency) ────
            _valid_contacts = [n["contact_freq"] for n in all_numeric
                               if n.get("contact_freq") is not None]
            if _valid_contacts:
                with st.expander("\U0001f91d Contact \u2014 Combined", expanded=True):
                    # Merge on resid/resname, average contact_fraction
                    _all_freq = pd.concat(_valid_contacts, ignore_index=False)
                    _avg_freq = _all_freq.groupby(["resid", "resname"]).agg(
                        mean_frac=("contact_fraction", "mean"),
                        std_frac=("contact_fraction", "std"),
                    ).reset_index().sort_values("mean_frac", ascending=False)
                    _avg_freq["label"] = _avg_freq["resname"] + _avg_freq["resid"].astype(str)
                    _avg_freq["std_frac"] = _avg_freq["std_frac"].fillna(0)

                    fig, ax = plt.subplots(figsize=(12, 5))
                    x = range(len(_avg_freq))
                    ax.bar(x, _avg_freq["mean_frac"], yerr=_avg_freq["std_frac"],
                           color="#E91E63", alpha=0.7, edgecolor="#C2185B", lw=0.5,
                           capsize=2)
                    ax.set_xticks(list(x))
                    ax.set_xticklabels(_avg_freq["label"], rotation=45, ha="right", fontsize=9)
                    ax.set_xlabel(f"{_pep_label} Residue", fontsize=12, fontweight="bold")
                    ax.set_ylabel("Contact Occupancy (%)", fontsize=12, fontweight="bold")
                    ax.set_title(f"Contact Frequency \u2014 Combined ({n_replicas} replicas)",
                                 fontsize=13, fontweight="bold")
                    fig.tight_layout()
                    save_and_show(fig, "combined_contact", combined_out_dir)
                    # Save combined Contact CSV
                    _avg_freq.to_csv(
                        os.path.join(combined_out_dir, "combined_contact_frequency.csv"), index=False)

                    combined_report["analyses"]["Contact"] = {
                        "top_contacts": ", ".join(_avg_freq.head(5)["label"].values)}
                    combined_report["plots"]["Contact"] = ["combined_contact.png"]

            # ──── ProLIF (average occupancy) ────
            _valid_prolif = [n["prolif_occ"] for n in all_numeric
                             if n.get("prolif_occ") is not None]
            if _valid_prolif:
                with st.expander("\U0001f9f2 ProLIF \u2014 Combined", expanded=True):
                    _all_occ = pd.concat(_valid_prolif, axis=1)
                    _mean_occ = _all_occ.mean(axis=1).sort_values(ascending=False).head(prolif_top_n)
                    _std_occ = _all_occ.std(axis=1).reindex(_mean_occ.index).fillna(0)

                    labels_f = [f"{c[0]}-{c[1]}\n({c[2]})"
                                if isinstance(c, tuple) else str(c)
                                for c in _mean_occ.index]
                    fig, ax = plt.subplots(figsize=(14, max(4, len(_mean_occ)*0.5)))
                    ax.barh(range(len(_mean_occ)), _mean_occ.values,
                            xerr=_std_occ.values,
                            color=plt.cm.Set2(np.linspace(0, 1, len(_mean_occ))),
                            capsize=2)
                    ax.set_yticks(range(len(labels_f)))
                    ax.set_yticklabels(labels_f, fontsize=8)
                    ax.set_xlabel("Occupancy (%)")
                    ax.set_title(f"Top Interactions (ProLIF) \u2014 Combined ({n_replicas} replicas)",
                                 fontweight="bold")
                    ax.invert_yaxis()
                    fig.tight_layout()
                    save_and_show(fig, "combined_prolif", combined_out_dir)
                    combined_report["analyses"]["ProLIF"] = {
                        "top_interactions": ", ".join(
                            f"{c[0]}-{c[1]} ({c[2]})" if isinstance(c, tuple) else str(c)
                            for c in _mean_occ.head(5).index)}
                    # Save combined ProLIF CSV
                    pd.DataFrame({
                        "Interaction": [f"{c[0]}-{c[1]} ({c[2]})" if isinstance(c, tuple)
                                        else str(c) for c in _mean_occ.index],
                        "Mean_Occupancy": _mean_occ.values,
                        "Std_Occupancy": _std_occ.values,
                    }).to_csv(os.path.join(combined_out_dir, "combined_prolif_occupancy.csv"), index=False)
                    combined_report["plots"]["ProLIF"] = ["combined_prolif.png"]

            # ──── eRMSF (mean heatmap) ────
            if all(("ermsf" in n) for n in all_numeric):
                with st.expander("\U0001f321\ufe0f eRMSF \u2014 Combined", expanded=True):
                    # Get the main label (Complex for Holo, or the APO label)
                    if is_holo:
                        _lbl = "Complex"
                    else:
                        _lbl = "Protein" if system_type == "APO Protein" else "Peptide"
                    _mats = [n["ermsf"][_lbl] for n in all_numeric if _lbl in n["ermsf"]]
                    if _mats:
                        _min_rows = min(m.shape[0] for m in _mats)
                        _min_cols = min(m.shape[1] for m in _mats)
                        _trimmed = np.array([m[:_min_rows, :_min_cols] for m in _mats])
                        _mean_mat = np.mean(_trimmed, axis=0)

                        fig, ax = plt.subplots(figsize=(14, 6))
                        im = ax.imshow(_mean_mat, cmap="viridis", aspect="auto",
                                       vmin=ermsf_vmin, vmax=ermsf_vmax,
                                       origin="lower", interpolation="bicubic",
                                       extent=[0, sim_ns, -0.5, _mean_mat.shape[0]-0.5])
                        ax.set_xlabel("Time (ns)", fontsize=12, fontweight="bold")
                        ax.set_ylabel("Residue", fontsize=12, fontweight="bold")
                        ax.set_title(f"eRMSF \u2014 Combined Mean ({n_replicas} replicas)",
                                     fontsize=14, fontweight="bold")
                        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
                        cbar.set_label("RMSF ($\\AA$)", fontsize=12, fontweight="bold")
                        fig.tight_layout()
                        save_and_show(fig, "combined_ermsf", combined_out_dir)
                        combined_report["analyses"]["eRMSF"] = {
                            "mean": f"{_mean_mat.mean():.2f} A",
                            "max": f"{_mean_mat.max():.2f} A"}
                        # Save combined eRMSF CSV (mean heatmap as matrix)
                    pd.DataFrame(_mean_mat).to_csv(
                        os.path.join(combined_out_dir, "combined_ermsf_matrix.csv"), index=False)
                    combined_report["plots"]["eRMSF"] = ["combined_ermsf.png"]

            # ──── Interaction Energy (mean +/- std) ────
            if all(("ie_total" in n) for n in all_numeric):
                with st.expander("\u26a1 Interaction Energy \u2014 Combined", expanded=True):
                    all_total = np.array([n["ie_total"] for n in all_numeric])
                    all_elec = np.array([n["ie_elec"] for n in all_numeric])
                    all_vdw = np.array([n["ie_vdw"] for n in all_numeric])
                    ie_mode = all_numeric[0].get("ie_mode", "Unknown")

                    fig, ax = plt.subplots(figsize=(12, 5))
                    for arr, col, lbl in [(all_total, "blue", "Total"),
                                           (all_elec, "green", "Electrostatic"),
                                           (all_vdw, "red", "van der Waals")]:
                        _m = np.mean(arr, axis=0)
                        _s = np.std(arr, axis=0)
                        ax.plot(time_array, _m, color=col, lw=1.0, label=f"{lbl} (mean)")
                        ax.fill_between(time_array, _m - _s, _m + _s,
                                        alpha=0.15, color=col)
                    ax.set_xlabel("Time (ns)", fontsize=12, fontweight="bold")
                    ax.set_ylabel("Energy (kcal/mol)", fontsize=12, fontweight="bold")
                    ax.set_title(f"Interaction Energy \u2014 Combined ({n_replicas} replicas, {ie_mode})",
                                 fontsize=13, fontweight="bold")
                    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
                    ax.set_xlim(0, sim_ns)
                    fig.tight_layout()
                    save_and_show(fig, "combined_interaction_energy", combined_out_dir)
                    # Save combined Interaction Energy CSV
                    pd.DataFrame({
                        "Time_ns": time_array,
                        "Total_mean": np.mean(all_total, axis=0),
                        "Total_std": np.std(all_total, axis=0),
                        "Electrostatic_mean": np.mean(all_elec, axis=0),
                        "Electrostatic_std": np.std(all_elec, axis=0),
                        "VdW_mean": np.mean(all_vdw, axis=0),
                        "VdW_std": np.std(all_vdw, axis=0),
                    }).to_csv(os.path.join(combined_out_dir, "combined_interaction_energy.csv"), index=False)

                    combined_report["analyses"]["Interaction Energy"] = {
                        "total": f"{np.mean(all_total):.1f} +/- {np.std([np.mean(n['ie_total']) for n in all_numeric]):.1f} kcal/mol",
                        "electrostatic": f"{np.mean(all_elec):.1f} +/- {np.std([np.mean(n['ie_elec']) for n in all_numeric]):.1f} kcal/mol",
                        "van_der_waals": f"{np.mean(all_vdw):.1f} +/- {np.std([np.mean(n['ie_vdw']) for n in all_numeric]):.1f} kcal/mol"}
                    combined_report["plots"]["Interaction Energy"] = ["combined_interaction_energy.png"]

            # ──── FEL (combined from all replicas) ────
            if all(("fel_rmsd" in n and "fel_rg" in n) for n in all_numeric):
                with st.expander("\U0001f30b Free Energy Landscape \u2014 Combined", expanded=True):
                    rmsd_all_rep = np.concatenate([n["fel_rmsd"] for n in all_numeric])
                    rg_all_rep = np.concatenate([n["fel_rg"] for n in all_numeric])

                    kB = 0.001987
                    RT = kB * fel_temperature
                    hist, xedges, yedges = np.histogram2d(rmsd_all_rep, rg_all_rep, bins=fel_bins)
                    hist_prob = hist / np.sum(hist)
                    hist_prob[hist_prob == 0] = np.min(hist_prob[hist_prob > 0]) * 0.01
                    fel = -RT * np.log(hist_prob)
                    fel = fel - np.min(fel)
                    fel_smooth = _gaussian_filter(fel, sigma=1.0)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
                    levels = np.linspace(0, np.percentile(fel_smooth, 95), 20)
                    contour = ax.contourf(X, Y, fel_smooth.T, levels=levels, cmap="viridis")
                    ax.contour(X, Y, fel_smooth.T, levels=levels,
                               colors="white", linewidths=0.5, alpha=0.3)
                    cbar = plt.colorbar(contour, ax=ax)
                    cbar.set_label("Free Energy (kcal/mol)", fontsize=12, fontweight="bold")
                    min_idx = np.unravel_index(np.argmin(fel_smooth), fel_smooth.shape)
                    ax.plot(xedges[min_idx[0]], yedges[min_idx[1]],
                            "r*", markersize=20, label="Global Minimum",
                            markeredgecolor="white", markeredgewidth=1)
                    ax.set_xlabel("RMSD (\u00c5)", fontsize=12, fontweight="bold")
                    ax.set_ylabel("Radius of Gyration (\u00c5)", fontsize=12, fontweight="bold")
                    ax.set_title(f"Free Energy Landscape \u2014 Combined ({n_replicas} replicas)",
                                 fontsize=13, fontweight="bold")
                    ax.legend(fontsize=9)
                    fig.tight_layout()
                    save_and_show(fig, "combined_fel", combined_out_dir)
                    # Save combined FEL CSV
                    pd.DataFrame({
                        "RMSD": rmsd_all_rep, "Rg": rg_all_rep,
                    }).to_csv(os.path.join(combined_out_dir, "combined_fel_data.csv"), index=False)
                    combined_report["plots"]["Free Energy Landscape"] = ["combined_fel.png"]

            # ──── DSSP (average SS composition) ────
            _valid_dssp = [n for n in all_numeric if "dssp_num" in n]
            if _valid_dssp:
                with st.expander("\U0001f9e9 DSSP \u2014 Combined", expanded=True):
                    # Average SS fractions across replicas
                    for d_key, d_label, d_col in [("dssp_pep_idx", "Peptide", "#E91E63"),
                                                    ("dssp_prot_idx", "Protein", "#2196F3")]:
                        _idxs = [n[d_key] for n in _valid_dssp if n.get(d_key)]
                        if not _idxs or not _idxs[0]:
                            continue
                        h_fracs, e_fracs, c_fracs = [], [], []
                        for n in _valid_dssp:
                            idx = n[d_key]
                            if not idx:
                                continue
                            _min_frames = n["dssp_num"].shape[0]
                            data_ss = n["dssp_num"][:, idx]
                            nr = data_ss.shape[1]
                            h_fracs.append(np.sum(data_ss == 0, axis=1) / nr * 100)
                            e_fracs.append(np.sum(data_ss == 1, axis=1) / nr * 100)
                            c_fracs.append(np.sum(data_ss == 2, axis=1) / nr * 100)
                        _min_len = min(len(h) for h in h_fracs)
                        h_arr = np.array([h[:_min_len] for h in h_fracs])
                        e_arr = np.array([e[:_min_len] for e in e_fracs])
                        c_arr = np.array([c[:_min_len] for c in c_fracs])
                        _ta = np.linspace(0, sim_ns, _min_len)

                        fig, ax = plt.subplots(figsize=(12, 4))
                        for arr, lbl, col in [(h_arr, "Helix", "#E91E63"),
                                               (e_arr, "Strand", "#2196F3"),
                                               (c_arr, "Coil", "#BDBDBD")]:
                            _m = np.mean(arr, axis=0)
                            _s = np.std(arr, axis=0)
                            ax.plot(_ta, _m, color=col, lw=1.2, label=lbl)
                            ax.fill_between(_ta, _m - _s, _m + _s, alpha=0.2, color=col)
                        ax.set_xlabel("Time (ns)", fontsize=11, fontweight="bold")
                        ax.set_ylabel("SS Content (%)", fontsize=11, fontweight="bold")
                        ax.set_title(f"{d_label} SS Composition \u2014 Combined ({n_replicas} replicas)",
                                     fontsize=12, fontweight="bold")
                        ax.set_xlim(0, sim_ns); ax.set_ylim(0, 100)
                        ax.legend(fontsize=9)
                        fig.tight_layout()
                        save_and_show(fig, f"combined_dssp_{d_label.lower()}", combined_out_dir)
                        # Save combined DSSP CSV for this domain
                        pd.DataFrame({
                            "Time_ns": _ta,
                            "Helix_mean": np.mean(h_arr, axis=0),
                            "Helix_std": np.std(h_arr, axis=0),
                            "Strand_mean": np.mean(e_arr, axis=0),
                            "Strand_std": np.std(e_arr, axis=0),
                            "Coil_mean": np.mean(c_arr, axis=0),
                            "Coil_std": np.std(c_arr, axis=0),
                        }).to_csv(os.path.join(combined_out_dir, f"combined_dssp_{d_label.lower()}.csv"), index=False)
                    combined_report["analyses"]["DSSP"] = "Cross-replica average SS composition"
                    combined_report["plots"]["DSSP"] = [
                        "combined_dssp_peptide.png", "combined_dssp_protein.png"]



            # ──── Combined AI Report ────
            if generate_report and (selected_model["provider"] in ("databricks", "ollama") or llm_api_key):
                with st.expander("\U0001f4dd Combined AI Report", expanded=True):
                    try:
                        if not HAS_FPDF:
                            raise ImportError("fpdf2")

                        # Build combined prompt
                        metrics_text = []
                        metrics_text.append(f"SYSTEM: {combined_report['system']['n_atoms']} atoms, "
                            f"{combined_report['system']['n_frames']} frames, "
                            f"{combined_report['system']['simulation_ns']:.0f} ns simulation, "
                            f"{n_replicas} replicas")
                        metrics_text.append("")
                        for analysis_name, metrics in combined_report["analyses"].items():
                            metrics_text.append(f"--- {analysis_name} (cross-replica) ---")
                            if isinstance(metrics, dict):
                                for k, v in metrics.items():
                                    metrics_text.append(f"  {k}: {v}")
                            else:
                                metrics_text.append(f"  {metrics}")
                            metrics_text.append("")

                        if is_holo_ligand:
                            _system_desc = (
                                f"a protein-ligand complex simulation "
                                f"(ligand residue: {ligand_resname})")
                        elif is_holo:
                            _system_desc = "a protein-peptide complex simulation"
                        else:
                            _apo_label = "Protein" if system_type == "APO Protein" else "Peptide"
                            _system_desc = f"an APO {_apo_label.lower()} simulation"

                        prompt = (
                            f"You are a computational biochemist writing a professional molecular "
                            f"dynamics analysis report for {_system_desc} with {n_replicas} independent replicas. "
                            "The following data represents CROSS-REPLICA AVERAGES (mean +/- std). "
                            "Discuss reproducibility across replicas and statistical significance.\n\n"
                            + "\n".join(metrics_text) +
                            "\n\nGenerate the report with these EXACT section headers (use ## for each):\n"
                            "## EXECUTIVE SUMMARY\n"
                            "(Key findings from the combined replica analysis)\n\n"
                        )
                        for analysis_name in combined_report["analyses"]:
                            prompt += f"## {analysis_name.upper()}\n(Cross-replica interpretation)\n\n"
                        prompt += (
                            "## CONCLUSIONS AND RECOMMENDATIONS\n"
                            "(Key findings, reproducibility assessment, suggestions)\n\n"
                            "Reference specific values. Discuss inter-replica variability. "
                            "Use markdown: **bold** for key terms, bullet lists with -, "
                            "markdown tables where appropriate. Avoid Unicode special characters."
                        )

                        # Key images for combined report
                        key_images = [f for f in sorted(os.listdir(combined_out_dir)) if f.endswith(".png")]
                        image_paths = [os.path.join(combined_out_dir, img) for img in key_images[:5]]

                        st.write(f"Calling {selected_model['label']} for combined report...")
                        report_text = call_llm(
                            prompt=prompt, model_config=selected_model,
                            api_key=llm_api_key, image_paths=image_paths, max_tokens=8192)

                        # Parse sections
                        sections = {}
                        current_section = "PREAMBLE"
                        current_text = []
                        for rline in report_text.split("\n"):
                            if rline.startswith("## "):
                                if current_text:
                                    sections[current_section] = "\n".join(current_text).strip()
                                current_section = rline[3:].strip().upper()
                                current_text = []
                            else:
                                current_text.append(rline)
                        if current_text:
                            sections[current_section] = "\n".join(current_text).strip()

                        # Build PDF
                        pdf = FPDF()
                        _register_fonts(pdf)
                        pdf.set_auto_page_break(auto=True, margin=20)
                        pdf.add_page()
                        pdf.set_font(_FONT_NAME, "B", 28)
                        pdf.ln(20)
                        if _HAS_LOGO:
                            try:
                                pdf.image(_LOGO_PATH, x=75, w=60)
                                pdf.ln(5)
                            except Exception:
                                pass
                        pdf.cell(0, 15, "PepIntProt (PIP)", ln=True, align="C")
                        pdf.set_font(_FONT_NAME, "", 16)
                        if is_holo_ligand:
                            _comb_subtitle = f"Protein-Ligand Combined Report ({n_replicas} Replicas)"
                        elif is_holo and _pep_n_res > 40:
                            _comb_subtitle = f"Protein-Protein Combined Report ({n_replicas} Replicas)"
                        elif is_holo:
                            _comb_subtitle = f"Peptide-Protein Combined Report ({n_replicas} Replicas)"
                        else:
                            _apo_lbl_cover = "Protein" if system_type == "APO Protein" else "Peptide"
                            _comb_subtitle = f"APO {_apo_lbl_cover} Combined Report ({n_replicas} Replicas)"
                        pdf.cell(0, 10, _comb_subtitle, ln=True, align="C")
                        pdf.ln(10)
                        pdf.set_font(_FONT_NAME, "", 11)
                        import datetime
                        pdf.cell(0, 7, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                            ln=True, align="C")
                        pdf.cell(0, 7, f"AI analysis by {selected_model['label']}", ln=True, align="C")

                        _add_text = _add_text_to_pdf

                        def _add_image(pdf_obj, img_path, max_w=180):
                            if os.path.exists(img_path):
                                try:
                                    pdf_obj.ln(3)
                                    pdf_obj.image(img_path, x=15, w=max_w)
                                    pdf_obj.ln(5)
                                except Exception:
                                    pass

                        if "EXECUTIVE SUMMARY" in sections:
                            pdf.add_page()
                            pdf.set_font(_FONT_NAME, "B", 18)
                            pdf.cell(0, 10, "Executive Summary", ln=True)
                            pdf.ln(3)
                            _add_text(pdf, sections["EXECUTIVE SUMMARY"])

                        for analysis_name in combined_report["analyses"]:
                            pdf.add_page()
                            pdf.set_font(_FONT_NAME, "B", 16)
                            pdf.cell(0, 10, _sanitize_for_pdf(analysis_name), ln=True)
                            pdf.ln(2)
                            section_key = analysis_name.upper()
                            matched = sections.get(section_key)
                            if not matched:
                                for sk, sv in sections.items():
                                    if section_key in sk or sk in section_key:
                                        matched = sv
                                        break
                                if not matched:
                                    first_word = section_key.split()[0] if section_key else ""
                                    for sk, sv in sections.items():
                                        if first_word and sk.startswith(first_word):
                                            matched = sv
                                            break
                            if matched:
                                _add_text(pdf, matched)
                            for pf in combined_report["plots"].get(analysis_name, []):
                                _add_image(pdf, os.path.join(combined_out_dir, pf))

                        if "CONCLUSIONS AND RECOMMENDATIONS" in sections:
                            pdf.add_page()
                            pdf.set_font(_FONT_NAME, "B", 18)
                            pdf.cell(0, 10, "Conclusions and Recommendations", ln=True)
                            pdf.ln(3)
                            _add_text(pdf, sections["CONCLUSIONS AND RECOMMENDATIONS"])

                        pdf_path = os.path.join(combined_out_dir, "PepIntProt_Combined_Report.pdf")
                        pdf.output(pdf_path)
                        st.success(f"Combined PDF report generated!")

                        if "EXECUTIVE SUMMARY" in sections:
                            st.markdown("**Executive Summary:**")
                            st.markdown(sections["EXECUTIVE SUMMARY"][:500] + "...")

                        with open(pdf_path, "rb") as f_pdf:
                            _pdf_bytes = f_pdf.read()
                            st.session_state["pdf_data_combined"] = _pdf_bytes
                            st.session_state["pdf_name_combined"] = "PepIntProt_Combined_Report.pdf"
                            if "all_pdf_keys" not in st.session_state:
                                st.session_state["all_pdf_keys"] = []
                            if "combined" not in st.session_state["all_pdf_keys"]:
                                st.session_state["all_pdf_keys"].append("combined")
                    except Exception as e:
                        st.error(f"Combined report failed: {e}")

            # ──── Combined ZIP ────
            st.success(f"\u2705 Combined analysis complete ({n_replicas} replicas)!")
            with st.expander("\U0001f4c2 Combined Files & Download", expanded=True):
                # Build a master ZIP with all replica results + combined
                master_zip_buf = io.BytesIO()
                with zipfile.ZipFile(master_zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for ri, od in enumerate(all_out_dirs):
                        prefix = f"replica_{ri+1}/" if n_replicas > 1 else ""
                        for fname in sorted(os.listdir(od)):
                            zf.write(os.path.join(od, fname), prefix + fname)
                    # Add combined results
                    if os.path.exists(combined_out_dir):
                        for fname in sorted(os.listdir(combined_out_dir)):
                            zf.write(os.path.join(combined_out_dir, fname),
                                     f"combined/{fname}")
                master_zip_buf.seek(0)

                _n_files = sum(len(os.listdir(od)) for od in all_out_dirs)
                _n_comb = len(os.listdir(combined_out_dir)) if os.path.exists(combined_out_dir) else 0
                st.write(f"**{_n_files}** per-replica files + **{_n_comb}** combined files")

                st.session_state["zip_data_master"] = master_zip_buf.getvalue()
                st.session_state["zip_name_master"] = "PepIntProt_all_replicas.zip"
                if "all_zip_keys" not in st.session_state:
                    st.session_state["all_zip_keys"] = []
                if "master" not in st.session_state["all_zip_keys"]:
                    st.session_state["all_zip_keys"].append("master")



# ====================================================================
# PERSISTENT DOWNLOADS (using @st.fragment to avoid full page reruns)
# ====================================================================
# @st.fragment makes the download buttons rerun independently.
# Clicking a download button ONLY reruns this fragment, NOT the whole page.
# All analysis plots, metrics, and expanders stay visible.

@st.fragment
def _download_fragment():
    """Fragment for download buttons — clicking won't trigger full page rerun."""
    if not st.session_state.get("analysis_done", False):
        return

    st.divider()
    st.subheader("\U0001f4e5 Downloads")

    _zip_keys = st.session_state.get("all_zip_keys", [])
    _pdf_keys = st.session_state.get("all_pdf_keys", [])

    _n_buttons = len(_zip_keys) + len(_pdf_keys)
    if _n_buttons > 0:
        _dl_cols = st.columns(min(_n_buttons, 4))
        _col_idx = 0
        for zk in _zip_keys:
            with _dl_cols[_col_idx % len(_dl_cols)]:
                _zdata = st.session_state.get(f"zip_data_{zk}")
                _zname = st.session_state.get(f"zip_name_{zk}", "results.zip")
                if _zdata:
                    _label = "\u2b07\ufe0f " + _zname.replace("PepIntProt_", "").replace(".zip", "")
                    st.download_button(
                        _label, data=_zdata, file_name=_zname,
                        mime="application/zip", use_container_width=True,
                        key=f"persist_zip_{zk}",
                    )
            _col_idx += 1
        for pk in _pdf_keys:
            with _dl_cols[_col_idx % len(_dl_cols)]:
                _pdata = st.session_state.get(f"pdf_data_{pk}")
                _pname = st.session_state.get(f"pdf_name_{pk}", "report.pdf")
                if _pdata:
                    _label = "\U0001f4c4 " + _pname.replace("PepIntProt_", "").replace(".pdf", "")
                    st.download_button(
                        _label, data=_pdata, file_name=_pname,
                        mime="application/pdf", use_container_width=True,
                        key=f"persist_pdf_{pk}",
                    )
            _col_idx += 1

# Render the fragment (always called, but only shows content when analysis_done)
_download_fragment()

if not st.session_state.get("analysis_done", False):
    st.markdown("""
    ### \U0001f44b Welcome to PepIntProt (PIP) v4.0!

    **Supports**: Holo (peptide\u2013protein complex) and APO (single-chain) simulations.
    Now with **multi-replica support** (mean \u00b1 std cross-replica analysis).

    **Upload** your MD simulation files in the sidebar, choose your analyses,
    then click **\U0001f680 Run Analysis**.  When complete, download everything as a
    single ZIP.

    **Required uploads:**
    | File | Description |
    |---|---|
    | Trajectory | DCD, XTC, TRR, NC, or multi-frame PDB (one per replica) |
    | Topology PDB | Single-frame PDB with chain IDs (shared) |

    **Optional:**
    | File | Description |
    |---|---|
    | ProLIF topology | PDB with hydrogens + correct element column |
    | FF Topology | PRMTOP / TOP for accurate Interaction Energy |

    **Available analyses:**
    | Analysis | Description |
    |---|---|
    | \U0001f310 3D Visualization | Trajectory snapshots, COM trace (3D+XY), conformational overlay |
    | \U0001f4c9 RMSD | Root Mean Square Deviation over time |
    | \U0001f4c8 RMSF | Root Mean Square Fluctuation per residue (with shadow fill) |
    | \U0001f535 Radius of Gyration | Compactness over time |
    | \U0001f3af PCA | Principal Component Analysis |
    | \U0001f9e9 DSSP | Secondary structure evolution (with Helix/Strand/Coil legend) |
    | \U0001f4cf Distance | Peptide\u2013Protein COM distance |
    | \U0001f91d Contact | Residue contacts + timeline heatmap |
    | \U0001f9f2 ProLIF | Interaction fingerprints (works with or without ProLIF PDB) |
    | \U0001f321\ufe0f eRMSF | Ensemble RMSF (complex heatmap with boundary) |
    | \U0001f30b FEL | Free Energy Landscape + representative frame PDB extraction |
    | \u26a1 Interaction Energy | Coulomb + LJ peptide-protein interaction energy |

    **Replica support (v3.0):**
    - Upload 1\u201310 independent trajectories
    - Each replica analyzed individually with its own report
    - Cross-replica combined analysis: mean \u00b1 std plots
    - Combined AI report discussing reproducibility

    **Features:**
    - APO / Holo system detection
    - Auto-detection of peptide / protein chains
    - All plots saved at 200 dpi + CSV data
    - Download all results as a single ZIP
    - AI-powered PDF report (6 providers: Databricks, Anthropic, Groq, OpenAI, Gemini, Ollama)
    """)

