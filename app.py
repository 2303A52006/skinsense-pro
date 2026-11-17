import hashlib
import html
import json
import os
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageOps
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="SkinSense Pro",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== CONSTANTS ====================
SESSION_MAX_AGE_HOURS = 3  # public mode: delete a visitor's data after this much inactivity
MODEL_PATH = "models/skinsense_model.pth"
EVAL_PATH = "data/eval_results.json"
VAL_ACCURACY = "74.2%"  # update this after you retrain the model

THEMES = {
    "Light": {
        "bg": "#eef3f9",
        "surface": "#ffffff",
        "border": "#c9d6e6",
        "text": "#0b1730",
        "muted": "#53627d",
        "accent": "#0369a1",
        "accent_hover": "#075985",
        "on_accent": "#ffffff",
        "accent_solid": "#0284c7",
        "glow": "rgba(2, 132, 199, 0.06)",
        "glow_strong": "rgba(2, 132, 199, 0.12)",
        "hero_bg": "linear-gradient(180deg, #ffffff 0%, #e4f1fc 100%)",
        "shadow": "0 1px 2px rgba(16, 40, 80, 0.06), 0 10px 28px -14px rgba(16, 40, 80, 0.18)",
        "high": "#b91c1c",
        "medium": "#b45309",
        "low": "#047857",
        "uncertain": "#475569",
        "plotly": "plotly_white",
    },
    "Dark": {
        "bg": "#05070f",
        "surface": "#0b1020",
        "border": "#1c2744",
        "text": "#eaf0ff",
        "muted": "#8fa0c4",
        "accent": "#38a8ff",
        "accent_hover": "#7cc4ff",
        "on_accent": "#ffffff",
        "accent_solid": "#0a84ff",
        "glow": "rgba(10, 132, 255, 0.28)",
        "glow_strong": "rgba(10, 132, 255, 0.45)",
        "hero_bg": "radial-gradient(700px 260px at 50% 125%, rgba(10, 132, 255, 0.45), transparent 70%), #0b1020",
        "shadow": "0 0 0 0 rgba(0, 0, 0, 0)",
        "high": "#f87171",
        "medium": "#fbbf24",
        "low": "#34d399",
        "uncertain": "#94a3b8",
        "plotly": "plotly_dark",
    },
}
FONT_SIZES = {"Small": "14px", "Medium": "16px", "Large": "18px"}

DEFAULT_SETTINGS = {
    "theme": "Light",
    "font_size": "Medium",
    "display_name": "",
    "conf_threshold": 60,
    "show_gradcam": True,
}

PAGES = {
    "welcome": "Welcome",
    "dashboard": "Dashboard",
    "analysis": "New Analysis",
    "journal": "Skin Journal",
    "compare": "Compare Analyses",
    "timeline": "Lesion Timeline",
    "library": "Educational Library",
    "reports": "Reports",
    "evaluation": "Model Evaluation",
    "settings": "Profile & Settings",
}

# Body sites used to track the same spot over time
MIDLINE_SITES = ["Face", "Scalp", "Neck", "Chest", "Back", "Abdomen"]
PAIRED_SITES = ["Shoulder", "Upper arm", "Forearm", "Hand", "Thigh", "Lower leg", "Foot"]
BODY_SITES = (
    ["Not specified"] + MIDLINE_SITES
    + [f"{side} {part}" for part in PAIRED_SITES for side in ("Left", "Right")]
    + ["Other"]
)

disease_info_full = {
    "akiec": {
        "name": "Actinic Keratosis",
        "severity": "MEDIUM",
        "advice": "The model suggests a pre-cancerous sun-damage lesion. Book a dermatology appointment within a month.",
        "description": "Actinic keratosis is a rough, scaly patch caused by years of sun exposure. It is considered pre-cancerous and can develop into skin cancer if left untreated.",
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "severity": "HIGH",
        "advice": "The model suggests a possible skin cancer. See a dermatologist within 2-4 weeks.",
        "description": "Basal cell carcinoma is the most common form of skin cancer. It usually appears on sun-exposed areas and rarely spreads, but it should be treated promptly.",
    },
    "bkl": {
        "name": "Benign Keratosis",
        "severity": "LOW",
        "advice": "The model suggests a benign growth. Have a doctor check it if it changes or if you are worried.",
        "description": "Benign keratosis is a non-cancerous skin growth that is common with age and usually needs no treatment.",
    },
    "df": {
        "name": "Dermatofibroma",
        "severity": "LOW",
        "advice": "The model suggests a benign nodule. Have a doctor check it if it changes or if you are worried.",
        "description": "Dermatofibroma is a common, benign skin nodule. It does not need removal unless it bothers you.",
    },
    "mel": {
        "name": "Melanoma",
        "severity": "HIGH",
        "advice": "The model suggests a possible melanoma. See a dermatologist as soon as possible, ideally within 1-2 weeks.",
        "description": "Melanoma is the most serious type of skin cancer. It develops in melanocytes and can spread to other parts of the body if not treated early.",
    },
    "nv": {
        "name": "Melanocytic Nevus",
        "severity": "LOW",
        "advice": "The model suggests a common mole. Check it regularly and see a doctor if it changes in size, shape or colour.",
        "description": "A melanocytic nevus is a common mole. Most are benign, but moles should be watched for changes in size, shape or colour.",
    },
    "vasc": {
        "name": "Vascular Lesion",
        "severity": "LOW",
        "advice": "The model suggests a vascular growth. See a doctor if it grows, bleeds or changes.",
        "description": "Vascular lesions are growths made of blood vessels. Most are benign but should be monitored for changes.",
    },
}


# ==================== PUBLIC MODE (privacy when deployed online) ====================
def is_public_mode():
    """True when the app is deployed online. Set SKINSENSE_PUBLIC=1 as an environment variable or Streamlit secret."""
    if os.environ.get("SKINSENSE_PUBLIC") == "1":
        return True
    try:
        return str(st.secrets.get("SKINSENSE_PUBLIC", "0")) == "1"
    except Exception:
        return False


PUBLIC_MODE = is_public_mode()
PRIVACY_NOTICE = (
    "Your photos and results are private to this browser session and are deleted automatically after a few hours of "
    "inactivity. Refreshing or closing the page starts a new, empty journal."
)


def data_dir():
    """Local use: the shared 'data' folder. Public use: a private folder for each visitor."""
    return st.session_state.data_dir


def db_path():
    return os.path.join(data_dir(), "skin_sense.db")


def upload_dir():
    return os.path.join(data_dir(), "uploads")


def cleanup_old_sessions():
    """Public mode only: remove visitor folders that have been inactive for a few hours."""
    root = os.path.join("data", "sessions")
    if not os.path.isdir(root):
        return
    cutoff = time.time() - SESSION_MAX_AGE_HOURS * 3600
    for name in os.listdir(root):
        path = os.path.join(root, name)
        times = [os.path.getmtime(os.path.join(folder, f)) for folder, _, files in os.walk(path) for f in files]
        newest = max(times) if times else os.path.getmtime(path)
        if newest < cutoff:
            shutil.rmtree(path, ignore_errors=True)


# ==================== DATABASE ====================
@contextmanager
def db():
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    os.makedirs(upload_dir(), exist_ok=True)
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                disease TEXT,
                confidence REAL,
                severity TEXT,
                advice TEXT,
                timestamp TEXT,
                image_path TEXT
            )
            """
        )
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(analyses)")]
        if "notes" not in columns:
            conn.execute("ALTER TABLE analyses ADD COLUMN notes TEXT DEFAULT ''")
        if "body_site" not in columns:
            conn.execute("ALTER TABLE analyses ADD COLUMN body_site TEXT DEFAULT 'Not specified'")
        conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
    if PUBLIC_MODE:
        os.utime(db_path())  # marks this visitor as active


def save_analysis(filename, disease, confidence, severity, advice, image_path, notes="", body_site="Not specified"):
    with db() as conn:
        conn.execute(
            """
            INSERT INTO analyses (filename, disease, confidence, severity, advice, timestamp, image_path, notes, body_site)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (filename, disease, confidence, severity, advice, datetime.now().isoformat(), image_path, notes, body_site),
        )


def get_all_analyses():
    with db() as conn:
        return pd.read_sql_query("SELECT * FROM analyses ORDER BY timestamp DESC", conn)


def get_analysis_by_id(analysis_id):
    with db() as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (int(analysis_id),)).fetchone()
    return dict(row) if row else None


def update_note(analysis_id, notes):
    with db() as conn:
        conn.execute("UPDATE analyses SET notes = ? WHERE id = ?", (notes, int(analysis_id)))


def update_body_site(analysis_id, body_site):
    with db() as conn:
        conn.execute("UPDATE analyses SET body_site = ? WHERE id = ?", (body_site, int(analysis_id)))


def delete_analysis(analysis_id):
    row = get_analysis_by_id(analysis_id)
    if row and row["image_path"] and os.path.exists(row["image_path"]):
        os.remove(row["image_path"])
    with db() as conn:
        conn.execute("DELETE FROM analyses WHERE id = ?", (int(analysis_id),))


def clear_all_analyses():
    df = get_all_analyses()
    for path in df["image_path"].dropna().tolist():
        if os.path.exists(path):
            os.remove(path)
    with db() as conn:
        conn.execute("DELETE FROM analyses")


# ==================== SETTINGS (saved in the database, so they survive a refresh) ====================
def load_settings():
    with db() as conn:
        stored = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM settings")}
    settings = dict(DEFAULT_SETTINGS)
    if stored.get("theme") in THEMES:
        settings["theme"] = stored["theme"]
    if stored.get("font_size") in FONT_SIZES:
        settings["font_size"] = stored["font_size"]
    settings["display_name"] = stored.get("display_name", "")
    if stored.get("conf_threshold", "").isdigit():
        settings["conf_threshold"] = int(stored["conf_threshold"])
    if "show_gradcam" in stored:
        settings["show_gradcam"] = stored["show_gradcam"] == "True"
    return settings


def update_setting(setting_key, widget_key):
    """Called by a widget's on_change. Copies the widget value into a safe key and saves it."""
    value = st.session_state[widget_key]
    st.session_state[setting_key] = value
    with db() as conn:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (setting_key, str(value)))


def go_to(page_key):
    st.session_state.page = page_key


# ==================== INIT STATE ====================
if "data_dir" not in st.session_state:
    if PUBLIC_MODE:
        cleanup_old_sessions()
        st.session_state.data_dir = os.path.join("data", "sessions", uuid.uuid4().hex)
    else:
        st.session_state.data_dir = "data"
init_db()
if "settings_loaded" not in st.session_state:
    st.session_state.update(load_settings())
    st.session_state.settings_loaded = True
    st.session_state.page = "welcome"
    st.session_state.saved = set()

T = THEMES[st.session_state.theme]
FS = FONT_SIZES[st.session_state.font_size]

# ==================== CSS ====================
root_css = f"""
:root {{
    --bg: {T['bg']}; --surface: {T['surface']}; --border: {T['border']};
    --text: {T['text']}; --muted: {T['muted']};
    --accent: {T['accent']}; --accent-hover: {T['accent_hover']}; --on-accent: {T['on_accent']};
    --accent-solid: {T['accent_solid']}; --glow: {T['glow']}; --glow-strong: {T['glow_strong']}; --hero-bg: {T['hero_bg']}; --shadow: {T['shadow']};
    --high: {T['high']}; --medium: {T['medium']}; --low: {T['low']}; --uncertain: {T['uncertain']};
    --fs: {FS};
}}
"""

base_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html { font-size: var(--fs); }
html, body, .stApp { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp { background: radial-gradient(1100px 520px at 50% -8%, var(--glow), transparent 70%), var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stHeader"] * { color: var(--muted) !important; }
.block-container { padding-top: 2.5rem; max-width: 1200px; }

/* Text colours (this is what was invisible in light theme) */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color: var(--text) !important; font-weight: 700; letter-spacing: -0.01em; }
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] strong,
.stApp label, .stApp label p, .stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stCaptionContainer"] { color: var(--text) !important; }
.stApp [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
.stApp hr { border-color: var(--border); }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .stButton > button { justify-content: flex-start; background: transparent; border-color: transparent; font-weight: 500; }
[data-testid="stSidebar"] .stButton > button:hover { background: var(--bg); border-color: var(--border); }
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] { background: var(--accent-solid); color: var(--on-accent) !important; }
.brand { font-size: 1.35rem; font-weight: 700; color: var(--text); margin: 0; }
.brand-sub { font-size: 0.85rem; color: var(--muted); margin: 0 0 0.5rem 0; }

/* Buttons */
[data-testid="stButton"], [data-testid="stDownloadButton"] { width: 100% !important; }
.stButton > button, .stDownloadButton > button {
    width: 100% !important; border-radius: 8px; padding: 0.55rem 1rem; font-weight: 600;
    background: var(--surface); color: var(--text); border: 1px solid var(--border); box-shadow: none;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--accent); color: var(--accent); }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--accent-solid); color: var(--on-accent) !important; border-color: var(--accent-solid);
    box-shadow: 0 8px 24px -8px var(--glow-strong);
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: var(--accent-solid); border-color: var(--accent-solid); color: var(--on-accent) !important; filter: brightness(1.12);
}
.stButton > button:disabled, .stDownloadButton > button:disabled { opacity: 0.5; }
.stApp button [data-testid="stMarkdownContainer"] p { color: inherit !important; }

/* Metrics */
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.2rem; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: var(--muted) !important; font-weight: 500; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: var(--text) !important; font-weight: 700; }

/* Inputs */
.stApp input, .stApp textarea { background: var(--surface) !important; color: var(--text) !important; border-color: var(--border) !important; }
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"] {
    background: var(--surface) !important; color: var(--text) !important; border-color: var(--border) !important;
}
[data-baseweb="select"] * { color: var(--text) !important; }
[data-baseweb="popover"], [data-baseweb="popover"] ul, [data-baseweb="menu"], [role="listbox"] { background: var(--surface) !important; }
[data-baseweb="popover"] li, [role="option"] { background: var(--surface) !important; color: var(--text) !important; }
[data-baseweb="popover"] li:hover, [role="option"]:hover { background: var(--bg) !important; }
[data-testid="stFileUploaderDropzone"] { background: var(--surface); border: 1.5px dashed var(--border); border-radius: 10px; }
[data-testid="stFileUploaderDropzone"] * { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button { color: var(--text) !important; background: var(--bg); border: 1px solid var(--border); }
[data-testid="stExpander"] details { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--text) !important; }

/* Custom components */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; color: var(--text); }
.card h3 { margin-top: 0; font-size: 1.1rem; }
.card ul { margin: 0; padding-left: 1.1rem; }
.card li { color: var(--text); margin-bottom: 0.25rem; }
.card p { color: var(--text); margin-bottom: 0.4rem; }
.muted { color: var(--muted) !important; }
.page-sub { color: var(--muted); margin-top: -0.6rem; margin-bottom: 1.5rem; }
.callout { --c: var(--accent); background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--c); border-radius: 8px; padding: 0.85rem 1.1rem; margin: 0.6rem 0; color: var(--text); }
.callout strong { color: var(--c); }
.callout.high { --c: var(--high); }
.callout.medium { --c: var(--medium); }
.callout.low { --c: var(--low); }
.callout.uncertain { --c: var(--uncertain); }
.badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; border: 1px solid var(--c); color: var(--c); }
.badge.high { --c: var(--high); } .badge.medium { --c: var(--medium); } .badge.low { --c: var(--low); } .badge.uncertain { --c: var(--uncertain); }
.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }
table.ss-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
table.ss-table th { text-align: left; padding: 0.6rem 0.9rem; color: var(--muted); font-weight: 600; border-bottom: 1px solid var(--border); }
table.ss-table td { padding: 0.6rem 0.9rem; color: var(--text); border-bottom: 1px solid var(--border); }
table.ss-table tr:last-child td { border-bottom: none; }

/* Muted text inside markdown (needs higher specificity than the global text rule) */
.stApp [data-testid="stMarkdownContainer"] p.muted,
.stApp [data-testid="stMarkdownContainer"] p.page-sub,
.stApp [data-testid="stMarkdownContainer"] p.brand-sub,
.stApp [data-testid="stMarkdownContainer"] p.hero-text,
.stApp [data-testid="stMarkdownContainer"] .step p { color: var(--muted) !important; }
.stApp [data-testid="stMarkdownContainer"] p.hero-kicker { color: var(--accent) !important; }

/* Welcome page */
.hero { position: relative; overflow: hidden; text-align: center; background: var(--hero-bg); border: 1px solid var(--border); border-radius: 18px; padding: 3rem 2.4rem 3.2rem; margin-bottom: 1.2rem; box-shadow: var(--shadow), 0 24px 60px -30px var(--glow-strong); }
.hero-kicker { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin: 0 0 0.6rem 0; }
.hero-title { font-size: 3.1rem; font-weight: 800; line-height: 1.15; letter-spacing: -0.02em; color: var(--text); margin: 0 0 0.7rem 0; }
.hero-text { font-size: 1.1rem; padding: 0 10%; margin: 0; }
.step { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem 1.3rem; }
.step-num { width: 2rem; height: 2rem; border-radius: 999px; background: var(--accent-solid); color: var(--on-accent); display: flex; align-items: center; justify-content: center; font-weight: 700; margin-bottom: 0.75rem; }
.step-title { font-size: 1.05rem; font-weight: 700; color: var(--text); margin: 0 0 0.3rem 0; }
.step p { margin: 0; font-size: 0.92rem; }
.card, .step, [data-testid="stMetric"] { box-shadow: var(--shadow); }
.card, .step { transition: border-color 0.2s; }
.card:hover, .step:hover { border-color: var(--accent); }
"""
st.markdown(f"<style>{root_css}{base_css}</style>", unsafe_allow_html=True)


# ==================== UI HELPERS ====================
def callout(text, kind="info", title=None):
    heading = f"<strong>{title}</strong><br>" if title else ""
    st.markdown(f'<div class="callout {kind}">{heading}{text}</div>', unsafe_allow_html=True)


def page_header(title, subtitle=None):
    st.title(title)
    if subtitle:
        st.markdown(f'<p class="page-sub">{subtitle}</p>', unsafe_allow_html=True)


def html_table(df):
    table = df.to_html(index=False, classes="ss-table", border=0).replace("\n", "")
    st.markdown(f'<div class="table-wrap">{table}</div>', unsafe_allow_html=True)


def style_fig(fig):
    fig.update_layout(
        template=T["plotly"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["text"], family="Inter, sans-serif"),
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=T["border"], title_text="")
    fig.update_yaxes(gridcolor=T["border"], title_text="")
    return fig


def severity_color(severity):
    return T[severity.lower()]


def format_analysis_label(analysis_id, df):
    row = df[df["id"] == analysis_id].iloc[0]
    return f"{row['timestamp'][:16].replace('T', ' ')} - {row['disease']}"


# ==================== MODEL ====================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except Exception:
        # Falls back for older checkpoints. Only do this with a model file you trained yourself.
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

try:
    model, class_names = load_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()


def generate_gradcam(model, image_tensor, original_image, target_category):
    target_layers = [model.layer4[-1]]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(target_category)])[0]
    img_np = np.array(original_image.convert("RGB"))
    grayscale_cam = cv2.resize(grayscale_cam, (img_np.shape[1], img_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.6 * img_np.astype(np.float32) + 0.4 * heatmap.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def open_image(file_bytes):
    """Open an upload, fix phone-photo rotation, force RGB."""
    return ImageOps.exif_transpose(Image.open(BytesIO(file_bytes))).convert("RGB")


def crop_by_percent(image, x_range, y_range):
    """x_range and y_range are (start, end) percentages of the width and height."""
    w, h = image.size
    box = (
        int(w * x_range[0] / 100),
        int(h * y_range[0] / 100),
        int(w * x_range[1] / 100),
        int(h * y_range[1] / 100),
    )
    return image.crop(box), box


def draw_crop_box(image, box):
    preview = image.copy()
    ImageDraw.Draw(preview).rectangle(box, outline=(10, 132, 255), width=max(3, image.size[0] // 150))
    return preview


def to_png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def check_image_quality(image):
    """Simple checks for size, blur and lighting. Returns (list of problems, measurements)."""
    issues = []
    width, height = image.size
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    scale = 512 / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    if min(width, height) < 224:
        issues.append(f"The image is small ({width}x{height} px). The model works at 224 px, so detail is lost when it is enlarged.")
    if sharpness < 30:
        issues.append("The image looks blurry. Retake it in focus and hold the camera steady.")
    if brightness < 60:
        issues.append("The image is quite dark. Retake it in bright, even light.")
    elif brightness > 200:
        issues.append("The image is very bright or washed out. Avoid flash glare and direct sun.")
    return issues, {"sharpness": sharpness, "brightness": brightness}


@st.cache_data(show_spinner="Analysing image...", max_entries=20, ttl=3600)
def run_inference(file_bytes):
    """Runs once per image. Clicking other widgets no longer re-runs the model."""
    image = open_image(file_bytes)
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor)[0], dim=0).numpy()
    predicted = int(probs.argmax())
    try:
        heatmap = generate_gradcam(model, tensor, image, predicted)
    except Exception:
        heatmap = None
    return probs, predicted, heatmap


# ==================== PDF REPORT ====================
def build_report_pdf(data, images=None):
    """data: dict with filename, disease, confidence, severity, advice, timestamp. Returns PDF bytes."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title="SkinSense Report")
    styles = getSampleStyleSheet()
    esc = lambda value: html.escape(str(value))
    story = [
        Paragraph("SkinSense Report", styles["Title"]),
        Paragraph(f"Date: {esc(str(data['timestamp'])[:16].replace('T', ' '))}", styles["Normal"]),
        Paragraph(f"File: {esc(data['filename'])}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph(f"<b>Model result:</b> {esc(data['disease'])}", styles["Normal"]),
        Paragraph(f"<b>Confidence:</b> {float(data['confidence']):.1f}%", styles["Normal"]),
        Paragraph(f"<b>Severity level:</b> {esc(data['severity'])}", styles["Normal"]),
        Paragraph(f"<b>Advice:</b> {esc(data['advice'])}", styles["Normal"]),
        Spacer(1, 12),
    ]
    site = data.get("body_site")
    if isinstance(site, str) and site != "Not specified":
        story.insert(3, Paragraph(f"Body site: {esc(site)}", styles["Normal"]))
    keep_alive = []
    for caption, pil_image in images or []:
        thumb = pil_image.copy()
        thumb.thumbnail((700, 700))
        buf = BytesIO()
        thumb.save(buf, "JPEG")
        buf.seek(0)
        keep_alive.append(buf)
        width = 3 * inch
        story.append(Paragraph(esc(caption), styles["Heading4"]))
        story.append(RLImage(buf, width=width, height=width * thumb.height / thumb.width))
        story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "<i>Disclaimer: SkinSense is an educational tool, not a medical device. It cannot diagnose "
            "any condition. Please consult a qualified dermatologist for any skin concern.</i>",
            styles["Normal"],
        )
    )
    doc.build(story)
    return buffer.getvalue()


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown('<p class="brand">SkinSense Pro</p><p class="brand-sub">Educational skin lesion classifier</p>', unsafe_allow_html=True)
    st.markdown("---")
    for key, label in PAGES.items():
        st.button(
            label,
            key=f"nav_{key}",
            type="primary" if st.session_state.page == key else "secondary",
            on_click=go_to,
            args=(key,),
        )
    st.markdown("---")
    st.markdown('<p class="brand-sub">v1.1 · Not a medical device</p>', unsafe_allow_html=True)
    if PUBLIC_MODE:
        st.markdown('<p class="brand-sub">Public demo: your data is private to this session.</p>', unsafe_allow_html=True)

page = st.session_state.page

# ==================== PAGE: WELCOME ====================
if page == "welcome":
    name = st.session_state.display_name.strip()
    greeting = f"Welcome back, {html.escape(name)}" if name else "Welcome to SkinSense"
    st.markdown(
        '<div class="hero"><p class="hero-kicker">Pigmented skin lesion classifier</p>'
        f'<div class="hero-title">{greeting}</div>'
        '<p class="hero-text">Upload a photo of a skin spot, see what the model thinks, and understand why. '
        "It recognises 7 types of pigmented lesions, such as moles and skin cancers, trained on more than 10,000 "
        "dermatologist-labelled images. It does not cover rashes or infections.</p></div>",
        unsafe_allow_html=True,
    )

    _l, b1, b2, _r = st.columns([1.3, 1, 1, 1.3])
    b1.button("Start new analysis", type="primary", on_click=go_to, args=("analysis",))
    b2.button("Open dashboard", on_click=go_to, args=("dashboard",))

    st.write("")
    analyses = get_all_analyses()
    mel_recall = None
    if os.path.exists(EVAL_PATH):
        try:
            with open(EVAL_PATH) as f:
                ev = json.load(f)
            mel_recall = next(r["recall"] for r in ev["per_class"] if r["code"] == "mel")
        except Exception:
            mel_recall = None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lesion classes", len(class_names))
    c2.metric("Validation accuracy", VAL_ACCURACY)
    c3.metric("Melanomas caught", f"{mel_recall:.1f}%" if mel_recall is not None else "n/a")
    c4.metric("Analyses saved", len(analyses))
    st.caption("Accuracy and melanoma detection are measured on images the model never saw during training.")

    st.subheader("How it works")
    s1, s2, s3 = st.columns(3)
    steps = [
        (s1, "1", "Upload", "Add a clear, close-up photo of the skin area you want to check."),
        (s2, "2", "Analyse", "The model ranks 7 lesion types and shows how confident it is. If it is unsure, it says so."),
        (s3, "3", "Understand", "A Grad-CAM heatmap shows where the model looked. Save results to track changes over time."),
    ]
    for col, num, title, text in steps:
        with col:
            st.markdown(
                f'<div class="step"><div class="step-num">{num}</div><div class="step-title">{title}</div><p>{text}</p></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div class="card"><h3>What it does</h3>'
            "<p>A ResNet-18 model trained on the HAM10000 dataset looks at a skin image and ranks 7 lesion types by probability. "
            "Grad-CAM highlights the areas of the image that influenced the result.</p>"
            '<p class="muted">Limits: the model was trained on dermoscopy images, so phone photos may give unreliable results.</p></div>',
            unsafe_allow_html=True,
        )
        callout(
            "SkinSense is an educational tool, not a medical device. If a spot is changing, bleeding, crusted or worrying you, see a dermatologist.",
            title="Please note",
        )
        if PUBLIC_MODE:
            callout(PRIVACY_NOTICE, title="Privacy")
    with col2:
        items = "".join(
            f"<li>{info['name']} "
            f'<span class="badge {info["severity"].lower()}">{info["severity"]}</span></li>'
            for info in disease_info_full.values()
        )
        st.markdown(f'<div class="card"><h3>Conditions covered</h3><ul>{items}</ul></div>', unsafe_allow_html=True)

    if not analyses.empty:
        st.subheader("Recent activity")
        recent = analyses[["disease", "confidence", "severity", "timestamp"]].head(3).copy()
        recent.columns = ["Result", "Confidence", "Severity", "Date"]
        recent["Confidence"] = recent["Confidence"].map(lambda v: f"{v:.1f}%")
        recent["Date"] = recent["Date"].str[:16].str.replace("T", " ")
        html_table(recent)

# ==================== PAGE: DASHBOARD ====================
elif page == "dashboard":
    page_header("Dashboard", "A summary of everything saved in your skin journal.")
    analyses = get_all_analyses()

    if analyses.empty:
        callout("No analyses saved yet. Run a new analysis and save it to see charts here.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total analyses", len(analyses))
        c2.metric("Average confidence", f"{analyses['confidence'].mean():.1f}%")
        c3.metric("High severity", int((analyses["severity"] == "HIGH").sum()))
        c4.metric("Most common", analyses["disease"].mode()[0])

        st.write("")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Disease distribution")
            counts = analyses["disease"].value_counts().sort_values()
            fig = px.bar(x=counts.values, y=counts.index, orientation="h", color_discrete_sequence=[T["accent"]])
            st.plotly_chart(style_fig(fig), theme=None)
        with col2:
            st.subheader("Severity levels")
            sev = analyses["severity"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "UNCERTAIN"]).fillna(0)
            fig = px.bar(x=sev.index, y=sev.values, color=sev.index, color_discrete_map={k: severity_color(k) for k in sev.index})
            st.plotly_chart(style_fig(fig), theme=None)

        if len(analyses) >= 2:
            st.subheader("Confidence over time")
            ordered = analyses.sort_values("timestamp")
            fig = px.line(ordered, x="timestamp", y="confidence", markers=True, color_discrete_sequence=[T["accent"]])
            st.plotly_chart(style_fig(fig), theme=None)

        st.subheader("Recent analyses")
        recent = analyses[["filename", "disease", "confidence", "severity", "timestamp"]].head(10).copy()
        recent["confidence"] = recent["confidence"].map(lambda v: f"{v:.1f}%")
        recent["timestamp"] = recent["timestamp"].str[:16].str.replace("T", " ")
        html_table(recent)

# ==================== PAGE: NEW ANALYSIS ====================
elif page == "analysis":
    page_header("New analysis", "Upload a clear, well-lit close-up of the skin area.")
    st.markdown(
        '<div class="card"><h3>For the best result</h3><ul>'
        "<li>Use the crop tool below so the lesion fills most of the frame</li>"
        "<li>Use bright, even light and keep the image in focus</li>"
        "<li>Avoid filters, shadows, hair and jewellery over the spot</li>"
        "<li>Include only one lesion per photo</li>"
        "</ul>"
        '<p class="muted" style="margin-top:0.6rem;">The model was trained on dermoscopy images, so ordinary phone photos are less reliable.</p></div>',
        unsafe_allow_html=True,
    )
    if PUBLIC_MODE:
        callout(PRIVACY_NOTICE, title="Privacy")
    uploaded_file = st.file_uploader("Choose a skin image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.caption("Waiting for an image...")
    else:
        file_bytes = uploaded_file.getvalue()
        source_hash = hashlib.md5(file_bytes).hexdigest()
        original = open_image(file_bytes)

        # ----- Crop tool -----
        with st.expander("Crop to the lesion (recommended)", expanded=True):
            st.caption("Move the sliders so the blue box surrounds only the spot. A closer crop usually gives a more reliable result.")
            x_range = st.slider("Left to right (%)", 0, 100, (0, 100), key=f"cx_{source_hash}")
            y_range = st.slider("Top to bottom (%)", 0, 100, (0, 100), key=f"cy_{source_hash}")
            too_small = (x_range[1] - x_range[0] < 10) or (y_range[1] - y_range[0] < 10)
            is_cropped = (tuple(x_range) != (0, 100) or tuple(y_range) != (0, 100)) and not too_small
            if too_small:
                callout("The crop is too small. Make the box larger (at least 10% in each direction). Using the full image for now.", kind="medium")
            cropped, box = crop_by_percent(original, x_range, y_range)
            p1, p2 = st.columns(2)
            p1.image(draw_crop_box(original, box) if not too_small else original, caption="Selected area")
            if is_cropped:
                p2.image(cropped, caption="Cropped image used for analysis")
            else:
                p2.caption("No crop applied. The full image will be analysed.")

        image = cropped if is_cropped else original
        analysed_bytes = to_png_bytes(image) if is_cropped else file_bytes
        file_hash = hashlib.md5(analysed_bytes).hexdigest()

        # ----- Image quality check -----
        issues, quality = check_image_quality(image)
        if issues:
            callout("<br>".join(f"&bull; {text}" for text in issues), kind="medium", title="Image quality warning")
        else:
            st.caption(f"Image quality check passed (sharpness {quality['sharpness']:.0f}, brightness {quality['brightness']:.0f}).")

        probs, predicted, heatmap = run_inference(analysed_bytes)

        code = class_names[predicted]
        info = disease_info_full[code]
        confidence = float(probs[predicted]) * 100
        threshold = st.session_state.conf_threshold
        uncertain = confidence < threshold
        top3 = np.argsort(probs)[::-1][:3]
        serious_options = [
            disease_info_full[class_names[i]]["name"]
            for i in top3
            if disease_info_full[class_names[i]]["severity"] == "HIGH"
        ]

        if uncertain:
            disease_label = f"Uncertain ({info['name']})"
            severity = "UNCERTAIN"
            advice = (
                "The model is not confident enough to give a useful result. Please see a dermatologist, "
                "especially if the spot is crusted, bleeding, growing or not healing. "
                "It may also be a condition this model does not cover, such as a rash or an infection."
            )
        else:
            disease_label = info["name"]
            severity = info["severity"]
            advice = info["advice"]

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Image analysed (cropped)" if is_cropped else "Image analysed")
        with col2:
            st.subheader("Result")
            m1, m2 = st.columns(2)
            m1.metric("Result", "Uncertain" if uncertain else info["name"])
            m2.metric("Confidence", f"{confidence:.1f}%")
            if uncertain:
                st.caption(f"Best guess: {info['name']}. Below your {threshold}% confidence threshold.")
                callout(advice, kind="uncertain", title="Result uncertain")
                if serious_options:
                    callout(
                        f"A serious condition is among the model's top guesses ({', '.join(serious_options)}). Do not rely on a 'benign' result.",
                        kind="medium",
                        title="Please get this checked",
                    )
            else:
                callout(advice, kind=severity.lower(), title=f"{severity} severity")

            top_items = "".join(
                f"<li>{disease_info_full[class_names[i]]['name']}: {probs[i] * 100:.1f}%</li>" for i in top3
            )
            st.markdown(
                f'<div class="card"><h3>Top 3 possibilities</h3><ul>{top_items}</ul></div>',
                unsafe_allow_html=True,
            )

        st.subheader("All class probabilities")
        prob_df = pd.DataFrame(
            {"Condition": [disease_info_full[c]["name"] for c in class_names], "Probability": probs * 100}
        ).sort_values("Probability")
        fig = px.bar(prob_df, x="Probability", y="Condition", orientation="h", color_discrete_sequence=[T["accent"]])
        st.plotly_chart(style_fig(fig), theme=None)

        if st.session_state.show_gradcam:
            st.subheader("Why this prediction? (Grad-CAM)")
            if heatmap is None:
                callout("Grad-CAM could not be generated for this image.", kind="medium")
            else:
                g1, g2 = st.columns(2)
                g1.image(image, caption="Original")
                g2.image(heatmap, caption="Areas the model focused on")

        st.markdown("---")
        st.subheader("Save this result")
        s1, s2 = st.columns(2)
        with s1:
            body_site = st.selectbox("Body site (lets you track this spot over time)", BODY_SITES, key=f"site_{source_hash}")
        with s2:
            note = st.text_input("Note (optional)", placeholder="e.g. noticed 2 weeks ago", key=f"note_{source_hash}")
        already_saved = file_hash in st.session_state.saved

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Save to skin journal", type="primary", disabled=already_saved):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(upload_dir(), f"upload_{stamp}_{file_hash[:6]}.jpg")
                image.save(image_path)
                save_analysis(uploaded_file.name, disease_label, confidence, severity, advice, image_path, note, body_site)
                st.session_state.saved.add(file_hash)
                already_saved = True
        with b2:
            pdf_images = [("Image analysed", image)]
            if heatmap is not None:
                pdf_images.append(("Grad-CAM focus areas", Image.fromarray(heatmap)))
            pdf_bytes = build_report_pdf(
                {
                    "filename": uploaded_file.name,
                    "disease": disease_label,
                    "confidence": confidence,
                    "severity": severity,
                    "advice": advice,
                    "timestamp": datetime.now().isoformat(),
                    "body_site": body_site,
                },
                pdf_images,
            )
            st.download_button("Download PDF report", pdf_bytes, file_name=f"skinsense_report_{file_hash[:6]}.pdf", mime="application/pdf")

        if already_saved:
            callout("Saved to your skin journal.", kind="low")

# ==================== PAGE: SKIN JOURNAL ====================
elif page == "journal":
    page_header("Skin journal", "Your saved analyses, newest first.")
    analyses = get_all_analyses()
    if analyses.empty:
        callout("No analyses saved yet.")
    else:
        for _, row in analyses.iterrows():
            site = row["body_site"] if row["body_site"] in BODY_SITES else "Not specified"
            site_text = f" · {site}" if site != "Not specified" else ""
            title = f"{row['timestamp'][:16].replace('T', ' ')} - {row['disease']} ({row['confidence']:.1f}%){site_text}"
            with st.expander(title):
                c1, c2 = st.columns([1, 2])
                with c1:
                    if row["image_path"] and os.path.exists(row["image_path"]):
                        st.image(row["image_path"], width=220)
                with c2:
                    st.markdown(
                        f"**Result:** {html.escape(row['disease'])} &nbsp; "
                        f'<span class="badge {row["severity"].lower()}">{row["severity"]}</span>',
                        unsafe_allow_html=True,
                    )
                    st.write(f"**Confidence:** {row['confidence']:.1f}%")
                    st.write(f"**Advice:** {row['advice']}")
                    new_site = st.selectbox("Body site", BODY_SITES, index=BODY_SITES.index(site), key=f"site_edit_{row['id']}")
                    new_note = st.text_input("Note", value=row["notes"] or "", key=f"note_edit_{row['id']}")
                    n1, n2 = st.columns(2)
                    if n1.button("Save changes", key=f"upd_{row['id']}"):
                        update_note(row["id"], new_note)
                        update_body_site(row["id"], new_site)
                        st.rerun()
                    if n2.button("Delete entry", key=f"del_{row['id']}"):
                        delete_analysis(row["id"])
                        st.rerun()

# ==================== PAGE: COMPARE ====================
elif page == "compare":
    page_header("Compare analyses", "Track how a lesion changes between two saved analyses.")
    analyses = get_all_analyses()
    if len(analyses) < 2:
        callout("You need at least 2 saved analyses to compare.")
    else:
        ids = analyses["id"].tolist()
        c1, c2 = st.columns(2)
        with c1:
            o1 = st.selectbox("First analysis", ids, format_func=lambda x: format_analysis_label(x, analyses), key="cmp1")
        with c2:
            o2 = st.selectbox("Second analysis", ids, index=1, format_func=lambda x: format_analysis_label(x, analyses), key="cmp2")

        if o1 == o2:
            callout("Pick two different analyses to compare.", kind="medium")
        else:
            a1, a2 = get_analysis_by_id(o1), get_analysis_by_id(o2)
            col1, col2 = st.columns(2)
            for col, label, a in ((col1, "Analysis 1", a1), (col2, "Analysis 2", a2)):
                with col:
                    st.subheader(label)
                    if a["image_path"] and os.path.exists(a["image_path"]):
                        st.image(a["image_path"], width=260)
                    st.write(f"**Result:** {a['disease']}")
                    st.write(f"**Confidence:** {a['confidence']:.1f}%")
                    st.write(f"**Severity:** {a['severity']}")
                    st.write(f"**Date:** {a['timestamp'][:16].replace('T', ' ')}")
            diff = a2["confidence"] - a1["confidence"]
            if a1["disease"] == a2["disease"]:
                callout(f"Same result in both analyses. Confidence changed by {diff:+.1f} points.", kind="low")
            else:
                callout(f"The result changed from {a1['disease']} to {a2['disease']}. Show both images to a dermatologist.", kind="medium")

# ==================== PAGE: TIMELINE ====================
elif page == "timeline":
    page_header("Lesion timeline", "Follow the same spot over time. Choose a body site whenever you save an analysis.")
    analyses = get_all_analyses()
    tracked = analyses[analyses["body_site"].fillna("Not specified") != "Not specified"].copy()

    if tracked.empty:
        callout(
            "Pick a body site when you save a new analysis. Entries for the same site then appear here as a timeline.",
            title="Nothing to show yet",
        )
    else:
        counts = tracked["body_site"].value_counts()
        site = st.selectbox(
            "Body site",
            counts.index.tolist(),
            format_func=lambda name: f"{name} ({counts[name]} {'analysis' if counts[name] == 1 else 'analyses'})",
        )
        entries = tracked[tracked["body_site"] == site].copy()
        entries["date"] = pd.to_datetime(entries["timestamp"])
        entries = entries.sort_values("date").reset_index(drop=True)
        first, last = entries.iloc[0], entries.iloc[-1]

        if len(entries) == 1:
            callout(
                "Only one analysis for this site so far. Save another photo of the same spot in a few weeks to see how it changes.",
                title="Just one entry",
            )
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Analyses", len(entries))
            m2.metric("First", first["date"].strftime("%d %b %Y"))
            m3.metric("Latest", last["date"].strftime("%d %b %Y"))
            m4.metric("Days covered", (last["date"] - first["date"]).days)

            def base_name(label):
                return label[11:-1] if label.startswith("Uncertain (") and label.endswith(")") else label

            rank = {"LOW": 0, "UNCERTAIN": 1, "MEDIUM": 2, "HIGH": 3}
            changed = base_name(first["disease"]) != base_name(last["disease"])
            worse = rank.get(last["severity"], 0) > rank.get(first["severity"], 0)
            if changed or worse:
                callout(
                    f"The model's result went from <strong>{html.escape(first['disease'])}</strong> to "
                    f"<strong>{html.escape(last['disease'])}</strong>. This could be a real change or just a different photo. "
                    "Show both photos to a dermatologist.",
                    kind="medium",
                    title="Result changed",
                )
            else:
                callout(
                    "The model's result has stayed the same. Lighting and angle also affect results, so keep watching the spot "
                    "and see a doctor if it changes in size, shape or colour.",
                    kind="low",
                    title="No change in result",
                )

            st.subheader("Confidence over time")
            fig = px.line(
                entries, x="date", y="confidence", markers=True,
                hover_data=["disease", "severity"], color_discrete_sequence=[T["accent"]],
            )
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(style_fig(fig), theme=None)

        st.subheader("Photos")
        per_row = 4
        for start_idx in range(0, len(entries), per_row):
            cols = st.columns(per_row)
            for col, (_, row) in zip(cols, entries.iloc[start_idx:start_idx + per_row].iterrows()):
                with col:
                    if row["image_path"] and os.path.exists(row["image_path"]):
                        st.image(row["image_path"], caption=f"{row['date'].strftime('%d %b %Y')} - {row['disease']}")

        st.subheader("All entries")
        table = entries[["date", "disease", "confidence", "severity", "notes"]].copy()
        table["date"] = table["date"].dt.strftime("%Y-%m-%d %H:%M")
        table["confidence"] = table["confidence"].map(lambda v: f"{v:.1f}%")
        table["notes"] = table["notes"].fillna("")
        table.columns = ["Date", "Result", "Confidence", "Severity", "Note"]
        html_table(table)
        st.caption("The model compares each photo separately. It cannot measure real growth, so judge changes by looking at the photos.")

# ==================== PAGE: LIBRARY ====================
elif page == "library":
    page_header("Educational library", "Plain-language notes on each lesion type the model knows.")
    st.markdown(
        '<div class="card"><h3>The ABCDE rule for checking moles</h3><ul>'
        "<li><strong>A</strong>symmetry: one half does not match the other</li>"
        "<li><strong>B</strong>order: edges are irregular or blurred</li>"
        "<li><strong>C</strong>olour: several shades in one mole</li>"
        "<li><strong>D</strong>iameter: bigger than about 6 mm</li>"
        "<li><strong>E</strong>volving: changing in size, shape or colour</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )
    can_items = "".join(f"<li>{info['name']}</li>" for info in disease_info_full.values())
    cannot_items = "".join(
        f"<li>{text}</li>"
        for text in [
            "Rashes and inflammatory conditions (eczema, psoriasis, hives, dermatitis)",
            "Fungal infections (ringworm, athlete's foot)",
            "Bacterial and viral infections (impetigo, warts, cold sores, shingles)",
            "Acne and rosacea",
            "Pigment conditions (vitiligo, melasma)",
            "Bites, burns, scabies and allergic reactions",
            "Other skin cancers, such as squamous cell carcinoma",
        ]
    )
    k1, k2 = st.columns(2)
    k1.markdown(f'<div class="card"><h3>What the model can detect</h3><ul>{can_items}</ul></div>', unsafe_allow_html=True)
    k2.markdown(
        f'<div class="card"><h3>What it cannot detect</h3><ul>{cannot_items}</ul>'
        '<p class="muted" style="margin-top:0.6rem;">For these, the model can only guess among its 7 classes, so the answer will be wrong or Uncertain. '
        "See a doctor for any skin problem that worries you.</p></div>",
        unsafe_allow_html=True,
    )
    for info in disease_info_full.values():
        with st.expander(f"{info['name']} ({info['severity']})"):
            st.write(info["description"])
            st.write(f"**Advice:** {info['advice']}")

# ==================== PAGE: REPORTS ====================
elif page == "reports":
    page_header("Reports", "Download a PDF for any saved analysis.")
    analyses = get_all_analyses()
    if analyses.empty:
        callout("No analyses available yet.")
    else:
        selected_id = st.selectbox(
            "Select analysis",
            analyses["id"].tolist(),
            format_func=lambda x: format_analysis_label(x, analyses),
        )
        data = get_analysis_by_id(selected_id)
        images = []
        if data["image_path"] and os.path.exists(data["image_path"]):
            images.append(("Saved image", Image.open(data["image_path"]).convert("RGB")))
        st.download_button(
            "Download PDF report",
            build_report_pdf(data, images),
            file_name=f"skinsense_report_{selected_id}.pdf",
            mime="application/pdf",
        )

# ==================== PAGE: MODEL EVALUATION ====================
elif page == "evaluation":
    page_header("Model evaluation", "How well the model really performs on images it never saw during training.")

    if not os.path.exists(EVAL_PATH):
        callout(
            "No evaluation results found. Run evaluate.py in your Colab notebook, download eval_results.json "
            f"and place it at <strong>{EVAL_PATH}</strong> in your project folder.",
            kind="medium",
            title="Results file missing",
        )
    else:
        with open(EVAL_PATH) as f:
            ev = json.load(f)
        names = ev["class_names"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{ev['accuracy']:.1f}%")
        c2.metric("Balanced accuracy", f"{ev['balanced_accuracy']:.1f}%")
        c3.metric("Calibration error", f"{ev['ece']:.1f} pts")
        c4.metric("Lesions in train & val", ev["lesion_overlap"])

        if ev["lesion_overlap"] == 0:
            callout("No lesion appears in both the training and validation sets, so the scores are not inflated by leakage.", kind="low", title="Clean split")
        else:
            callout("Some lesions appear in both sets, so the scores are probably too optimistic.", kind="high", title="Data leakage")

        per_class = pd.DataFrame(ev["per_class"])
        weakest = per_class.sort_values("recall").iloc[0]
        callout(
            f"Lowest recall: <strong>{html.escape(weakest['name'])}</strong> at {weakest['recall']:.1f}%. "
            "Recall is the share of real cases of that condition that the model catches. "
            "For serious conditions like melanoma, a low recall means missed cases.",
            kind="medium",
            title="Weakest class",
        )

        st.subheader("Recall per condition")
        recall_df = per_class.sort_values("recall")
        fig = px.bar(recall_df, x="recall", y="name", orientation="h", color_discrete_sequence=[T["accent"]])
        fig.update_xaxes(range=[0, 100])
        st.plotly_chart(style_fig(fig), theme=None)

        table = per_class[["name", "precision", "recall", "f1", "support"]].copy()
        table.columns = ["Condition", "Precision %", "Recall %", "F1 %", "Images"]
        html_table(table)

        st.subheader("Confusion matrix")
        st.caption("Rows are the true condition, columns are what the model predicted. A bright diagonal is good.")
        show_pct = st.toggle("Show as % of each true condition", value=True, key="eval_pct")
        cm = np.array(ev["confusion_matrix"], dtype=float)
        if show_pct:
            cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig = px.imshow(cm, x=names, y=names, color_continuous_scale="Teal", text_auto=".0%" if show_pct else True, aspect="auto")
        fig.update_layout(coloraxis_showscale=False, height=520)
        style_fig(fig)
        fig.update_xaxes(title_text="Predicted", side="bottom")
        fig.update_yaxes(title_text="True")
        st.plotly_chart(fig, theme=None)

        st.subheader("Is the confidence score honest?")
        st.caption("If the model says 80% confident, it should be right about 80% of the time. Points on the dashed line are perfectly calibrated.")
        cal = pd.DataFrame(ev["calibration"])
        fig = px.line(cal, x="confidence", y="accuracy", markers=True, hover_data=["bin", "count"], color_discrete_sequence=[T["accent"]])
        fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(dash="dash", color=T["muted"]))
        style_fig(fig)
        fig.update_xaxes(title_text="Average confidence %", range=[0, 100])
        fig.update_yaxes(title_text="Actual accuracy %", range=[0, 100])
        st.plotly_chart(fig, theme=None)

        st.subheader("Choosing the confidence threshold")
        st.caption("If the app only trusts results above a threshold, how many images are covered and how accurate are they?")
        thr = pd.DataFrame(ev["thresholds"])
        thr_table = pd.DataFrame(
            {
                "Confidence at least": thr["threshold"].map(lambda v: f"{v}%"),
                "Share of images": thr["coverage"].map(lambda v: f"{v:.1f}%"),
                "Accuracy among them": thr["accuracy"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.1f}%"),
            }
        )
        html_table(thr_table)
        st.caption(f"Evaluated on {ev['n_val']} validation images (model trained on {ev['n_train']}).")

# ==================== PAGE: SETTINGS ====================
elif page == "settings":
    page_header("Profile & settings", "Changes apply instantly and are remembered next time you open the app.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Appearance")
        theme_options = list(THEMES)
        font_options = list(FONT_SIZES)
        st.selectbox(
            "Theme", theme_options, index=theme_options.index(st.session_state.theme),
            key="w_theme", on_change=update_setting, args=("theme", "w_theme"),
        )
        st.selectbox(
            "Font size", font_options, index=font_options.index(st.session_state.font_size),
            key="w_font", on_change=update_setting, args=("font_size", "w_font"),
        )
    with col2:
        st.subheader("Profile")
        st.text_input(
            "Display name", value=st.session_state.display_name,
            key="w_name", on_change=update_setting, args=("display_name", "w_name"),
        )

    st.subheader("Analysis preferences")
    p1, p2 = st.columns(2)
    with p1:
        st.slider(
            "Low-confidence warning below (%)", 30, 90, st.session_state.conf_threshold,
            key="w_conf", on_change=update_setting, args=("conf_threshold", "w_conf"),
        )
    with p2:
        st.write("")
        st.toggle(
            "Show Grad-CAM heatmap", value=st.session_state.show_gradcam,
            key="w_cam", on_change=update_setting, args=("show_gradcam", "w_cam"),
        )

    st.markdown("---")
    st.subheader("Data & privacy")
    d1, d2 = st.columns(2)
    with d1:
        export_df = get_all_analyses()
        st.download_button(
            "Export my data (CSV)", export_df.to_csv(index=False), "skinsense_data.csv", "text/csv",
            disabled=export_df.empty,
        )
    with d2:
        confirm = st.checkbox("I understand this permanently deletes all saved analyses and images")
        if st.button("Clear all history", disabled=not confirm):
            clear_all_analyses()
            st.session_state.saved = set()
            st.rerun()

    st.markdown("---")
    st.subheader("About")
    st.markdown(
        f"""
**Version:** 1.1.0

**Model:** ResNet-18 (transfer learning)

**Training data:** HAM10000

**Validation accuracy:** {VAL_ACCURACY}

**Disclaimer:** SkinSense is an educational tool, not a medical diagnostic device. Always consult a qualified dermatologist for medical concerns.
"""
    )