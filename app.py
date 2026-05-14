import streamlit as st
import pandas as pd
import numpy as np
import pickle
import hashlib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="سند | SANAD",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Language & Session State ────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "ar"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_type" not in st.session_state:
    st.session_state.user_type = None
if "username" not in st.session_state:
    st.session_state.username = ""

LANG = st.session_state.lang

def t(ar, en):
    return ar if LANG == "ar" else en

DIR = "rtl" if LANG == "ar" else "ltr"

# ─── Admin Credentials (simple file-based store) ──────────────────────────────
USERS_FILE = "Database/sanad_users.json"
NEW_CASES_FILE = "Database/new_cases.csv"


def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    # Default admin
    default = {"admin": hashlib.sha256("admin123".encode()).hexdigest()}
    with open(USERS_FILE, "w") as f:
        json.dump(default, f)
    return default

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def check_password(username, password):
    users = load_users()
    hashed = hashlib.sha256(password.encode()).hexdigest()
    return users.get(username) == hashed

def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = hashlib.sha256(password.encode()).hexdigest()
    save_users(users)
    return True

# ─── Data & Model Loading ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Database/database.csv", dtype={"ssn": str})

        df["ssn"] = (
            df["ssn"]
            .str.replace(".0", "", regex=False)
            .str.strip()
        )
        return df
    except:
        return pd.DataFrame()

#___ new cases _____________________________________
def ensure_new_cases_file():
    """تأكد إن ملف new_cases.csv موجود وفيه headers الصح."""
    import os, pandas as pd
    cols = [
        "full_name", "gender", "ssn", "phone_number", "age", "family_size",
        "number_of_children", "governorate", "center", "village", "rural_or_urban",
        "housing_type", "monthly_income", "income_stability", "income_source",
        "expenses_estimate", "has_debt", "debt_amount", "education_level_head",
        "literacy", "children_in_school", "chronic_disease", "disabled_member",
        "medical_cost_estimate", "employment_status", "years_of_experience",
        "skills", "has_stable_job", "willing_to_work", "has_previous_business",
        "has_savings", "owns_assets", "access_to_water", "access_to_electricity",
        "application_date", "cluster",
    ]
    if not os.path.exists(NEW_CASES_FILE):
        pd.DataFrame(columns=cols).to_csv(NEW_CASES_FILE, index=False)

# ════════════════════════════════════════════════════════════════════════════
# FIX 1: load_models — بنجرب joblib الأول، لو فشل بنجرب pickle
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    try:
        import joblib
        HAS_JOBLIB = True
    except ImportError:
        HAS_JOBLIB = False

    models = {}
    for name, fname in [("processor", "sanad_processor.pkl"),
                         ("pca", "sanad_pca.pkl"),
                         ("gmm", "sanad_gmm_model.pkl"),
                         ("features", "features_list.pkl"),
                         ("cluster_names", "cluster_mapping.pkl")]:
        loaded = None
        # --- جرب joblib الأول ---
        if HAS_JOBLIB:
            try:
                loaded = joblib.load(fname)
            except Exception:
                loaded = None
        # --- لو joblib فشل جرب pickle ---
        if loaded is None:
            try:
                with open(fname, "rb") as f:
                    loaded = pickle.load(f)
            except Exception:
                loaded = None
        models[name] = loaded
    return models

CLUSTER_INFO = {
    0: {
        "ar": {"name": "حالة حرجة - أولوية قصوى", "desc": "أسرة تعاني من فقر شديد مع أعباء صحية وتعليمية مرتفعة. تحتاج تدخلاً فورياً وشاملاً.", "color": "#e74c3c", "priority": 1},
        "en": {"name": "Critical - Highest Priority", "desc": "Families in extreme poverty with high health and education burdens. Requires immediate comprehensive intervention.", "color": "#e74c3c", "priority": 1}
    },
    1: {
        "ar": {"name": "هشاشة عالية - دعم عاجل", "desc": "أسرة ذات دخل منخفض وعدم استقرار. تحتاج دعماً اقتصادياً وتنموياً.", "color": "#e67e22", "priority": 2},
        "en": {"name": "High Vulnerability - Urgent Support", "desc": "Low income and unstable households. Needs economic and developmental support.", "color": "#e67e22", "priority": 2}
    },
    2: {
        "ar": {"name": "هشاشة متوسطة - متابعة دورية", "desc": "أسرة في وضع متوسط مع بعض المخاطر. تستفيد من برامج التمكين.", "color": "#f39c12", "priority": 3},
        "en": {"name": "Moderate Vulnerability - Regular Follow-up", "desc": "Households in moderate condition with some risks. Benefits from empowerment programs.", "color": "#f39c12", "priority": 3}
    },
    3: {
        "ar": {"name": "استقرار نسبي - تمكين", "desc": "أسرة في وضع أفضل نسبياً. تحتاج برامج تمكين وتطوير.", "color": "#27ae60", "priority": 4},
        "en": {"name": "Relative Stability - Empowerment", "desc": "Relatively better-off households. Needs empowerment and development programs.", "color": "#27ae60", "priority": 4}
    },
}

def get_cluster_info(cluster_id, lang="ar"):
    info = CLUSTER_INFO.get(int(cluster_id), CLUSTER_INFO[0])
    return info[lang]

# ─── CSS ──────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(f"""
    <style>

    /* ══════════════════════════════════════════════════════════════
       📦 FONTS & ROOT VARIABLES
    ══════════════════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&family=Tajawal:wght@300;400;500;700&family=Poppins:wght@300;400;600;700&display=swap');

    :root {{
        --primary: #1a3a5c;
        --secondary: #2980b9;
        --accent: #e74c3c;
        --accent2: #f39c12;
        --success: #27ae60;
        --bg: #0d1117;
        --card: #161b22;
        --text: #e6edf3;
        --muted: #8b949e;
        --border: #30363d;
        --gradient: linear-gradient(135deg, #111827 0%, #1e293b 50%, #334155 100%);
        --gold: rgb(216, 174, 52);
    }}

    /* ══════════════════════════════════════════════════════════════
       🌍 GLOBAL BASE STYLES — font, direction, background
    ══════════════════════════════════════════════════════════════ */
    html, body, [class*="css"] {{
        font-family: {'Cairo, Tajawal' if LANG == 'ar' else 'Poppins'}, sans-serif;
        direction: {DIR};
        background: var(--bg);
        color: var(--text);
    }}

    .stApp {{ background: var(--bg); }}

    /* ══════════════════════════════════════════════════════════════
       🗂️ SIDEBAR
    ══════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0a0f1e 0%, #050810 100%);
        border-right: none;
    }}
    [data-testid="stSidebar"] * {{ color: #ecf0f1 !important; }}
    [data-testid="stSidebar"] .stRadio label {{
        color: #bdc3c7 !important;
        font-size: 15px;
        padding: 8px 0;
    }}
    [data-testid="stSidebar"] .stRadio div[data-testid="stMarkdownContainer"] p {{
        color: white !important;
    }}
    /* منع تكبير الأزرار في الـ sidebar */
    [data-testid="stSidebar"] div[data-testid="stButton"] button {{
        padding: 8px 10px !important;
        font-size: 0.85rem !important;
    }}

    /* ══════════════════════════════════════════════════════════════
       🏷️ MAIN HEADER BANNER
    ══════════════════════════════════════════════════════════════ */
    .main-header {{
        background: var(--gradient);
        padding: 32px 40px;
        border-radius: 20px;
        margin-bottom: 28px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 8px 32px rgba(26,58,92,0.25);
        position: relative;
        overflow: hidden;
    }}
    .main-header::before {{
        content: '';
        position: absolute;
        top: -50%;
        {'right' if LANG=='ar' else 'left'}: -10%;
        width: 300px; height: 300px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }}
    .main-header h1 {{ margin: 0; font-size: 2.2rem; font-weight: 900; letter-spacing: -0.5px; }}
    .main-header p  {{ margin: 6px 0 0; opacity: 0.85; font-size: 1rem; }}

    /* ══════════════════════════════════════════════════════════════
       📊 KPI CARDS GRID
    ══════════════════════════════════════════════════════════════ */
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin-bottom: 28px;
    }}
    @media (max-width: 768px) {{
        .kpi-grid {{ grid-template-columns: repeat(2,1fr); }}
    }}

    .kpi-card {{
        background: var(--card);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 2px 16px rgba(0,0,0,0.07);
        border-top: 5px solid var(--secondary);
        transition: transform 0.2s, box-shadow 0.2s;
    }}
    .kpi-card:hover  {{ transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0,0,0,0.12); }}
    .kpi-card .value {{ font-size: 2.2rem; font-weight: 900; color: var(--primary); margin: 0; }}
    .kpi-card .label {{ font-size: 0.85rem; color: var(--muted); margin-top: 4px; }}
    .kpi-card .icon  {{ font-size: 2rem; margin-bottom: 8px; }}

    /* ══════════════════════════════════════════════════════════════
       🃏 SECTION CARDS & TITLES
    ══════════════════════════════════════════════════════════════ */
    .section-card {{
        background: var(--card);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid var(--border);
    }}
    .section-title {{
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 2px solid var(--border);
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    /* ══════════════════════════════════════════════════════════════
       🏷️ CLUSTER BADGE
    ══════════════════════════════════════════════════════════════ */
    .cluster-badge {{
        display: inline-block;
        padding: 6px 16px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.9rem;
        color: white;
    }}

    /* ══════════════════════════════════════════════════════════════
       🔐 LOGIN PAGE — container styles (legacy fallback)
    ══════════════════════════════════════════════════════════════ */
    .login-container {{
        max-width: 460px;
        margin: 60px auto;
        background: var(--card);
        border-radius: 24px;
        padding: 48px 40px;
        box-shadow: 0 16px 48px rgba(26,58,92,0.15);
        text-align: center;
    }}
    .login-logo  {{ font-size: 4rem; margin-bottom: 8px; }}
    .login-title {{ font-size: 2rem; font-weight: 900; color: var(--primary); margin: 0 0 4px; }}
    .login-sub   {{ color: var(--muted); font-size: 0.95rem; margin-bottom: 32px; }}

    /* ══════════════════════════════════════════════════════════════
       🔘 BUTTONS — global style + hover + active
    ══════════════════════════════════════════════════════════════ */
    .stButton > button {{
        background: linear-gradient(135deg, #1f2937, #111827) !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        font-weight: 700 !important;
        transition: all 0.2s ease !important;
        box-shadow: none !important;
        font-size: 0.85rem !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        border-color: rgb(216, 174, 52) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.35) !important;
        color: rgb(216, 174, 52) !important;
    }}
    .stButton > button:active {{
        transform: translateY(0px);
        box-shadow: none !important;
    }}

    /* data-testid override (أكثر specificity) */
    div[data-testid="stButton"] button {{
        padding: 8px 12px !important;
        font-size: 0.9rem !important;
        border-radius: 10px !important;
        background: linear-gradient(135deg, #111827, #1f2937) !important;
        border: 1px solid #30363d !important;
        box-shadow: none !important;
    }}
    div[data-testid="stButton"] button:hover {{
        transform: translateY(-2px);
        border-color: rgb(216, 174, 52) !important;
    }}

    /* ── Active tab button (primary type) ── */

    div[data-testid="stButton"] button[kind="primary"],
    .stButton > button[kind="primary"] {{
        # background: linear-gradient(135deg, #d8ae34, #b88d22); !important;
        color: white !important;
        border-color: #d8ae34 !important;
        box-shadow: 0 2px 5px rgba(216, 174, 52,0.35) !important;
    }}
    div[data-testid="stButton"] button[kind="primary"]:hover {{
        border-color: rgb(216, 174, 52) !important;
        color: rgb(216, 174, 52) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.35) !important;
    }}

    /* ══════════════════════════════════════════════════════════════
       🔗 LINK BUTTONS
    ══════════════════════════════════════════════════════════════ */
    .stLinkButton a {{
        display: inline-block;
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: #e6edf3 !important;
        padding: 8px 12px;
        border-radius: 12px;
        border: 1px solid #30363d;
        text-decoration: none !important;
        font-weight: 700;
        transition: 0.2s;
        text-align: center;
    }}
    .stLinkButton a:hover {{
        border-color: rgb(216, 174, 52);
        color: rgb(216, 174, 52) !important;
        transform: translateY(-2px);
    }}

    /* ══════════════════════════════════════════════════════════════
       📝 INPUTS — text, number, textarea, select (dark theme + gold border)
    ══════════════════════════════════════════════════════════════ */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea,
    input,
    textarea {{
        background: #0d1117 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        outline: none !important;
        box-shadow: none !important;
    }}

    /* focus state */
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus,
    input:focus,
    textarea:focus {{
        border: 1px solid #d8ae34 !important;
        box-shadow: 0 0 0 2px rgba(216, 174, 52, 0.15) !important;
        outline: none !important;
    }}

    /* إزالة الـ validation border الافتراضية */
    input:invalid, input:valid,
    [data-testid="stTextInput"] input:invalid,
    [data-testid="stTextInput"] input:valid,
    [data-testid="stNumberInput"] input:invalid,
    [data-testid="stNumberInput"] input:valid {{
        box-shadow: none !important;
        border-color: #30363d !important;
    }}

    /* إخفاء helper text تحت الـ input */
    [data-testid="stTextInput"] small {{ display: none !important; }}

    /* إزالة outline عند focus-visible */
    input:focus-visible, textarea:focus-visible {{ outline: none !important; }}

    /* spinner buttons في number input */
    [data-testid="stNumberInput"] button {{
        background: transparent !important;
        border: none !important;
    }}

    /* ── Override Streamlit red validation border ── */
    [data-testid="stTextInput"] input:invalid,
    [data-testid="stTextInput"] input[aria-invalid="true"],
    [data-testid="stNumberInput"] input:invalid,
    input:invalid {{
        border-color: #d8ae34 !important;
        box-shadow: 0 0 0 2px rgba(216,174,52,0.15) !important;
    }}

    /* الـ wrapper اللي Streamlit بيحط عليه الـ red border */
    [data-testid="stTextInput"]:has(input:invalid),
    [data-testid="stTextInput"][data-invalid="true"] {{
        outline: none !important;
        border-color: #d8ae34 !important;
    }}

    /* بعض نسخ Streamlit بتحط الـ border على div خارجي */
    div[data-baseweb="input"]:focus-within {{
        border-color: #d8ae34 !important;
        box-shadow: 0 0 0 2px rgba(216,174,52,0.15) !important;
    }}
    div[data-baseweb="base-input"] {{
        background: #0d1117 !important;
        border-color: #30363d !important;
    }}
    div[data-baseweb="base-input"]:focus-within {{
        border-color: #d8ae34 !important;
    }}

    /* ══ SELECT / SELECTBOX ══ */
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] div[role="combobox"] {{
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        color: #e6edf3 !important;
        box-shadow: none !important;
        outline: none !important;
    }}
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="select"] div[role="combobox"]:focus-within {{
        border-color: #d8ae34 !important;
        box-shadow: 0 0 0 2px rgba(216,174,52,0.15) !important;
    }}
    /* dropdown list */
    div[data-baseweb="popover"] ul,
    div[role="listbox"] {{
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
    }}
    div[role="option"]:hover,
    div[role="option"][aria-selected="true"] {{
        background: rgba(216,174,52,0.12) !important;
        color: #d8ae34 !important;
    }}

    /* ══ NUMBER INPUT wrapper + stepper buttons ══ */
    div[data-testid="stNumberInput"] > div {{
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }}
    div[data-testid="stNumberInput"] > div:focus-within {{
        border-color: #d8ae34 !important;
        box-shadow: 0 0 0 2px rgba(216,174,52,0.15) !important;
    }}
    /* stepper +/- buttons */
    div[data-testid="stNumberInput"] button {{
        background: transparent !important;
        border: none !important;
        color: #8b949e !important;
    }}
    div[data-testid="stNumberInput"] button:hover {{
        color: #d8ae34 !important;
        transform: none !important;
    }}

    /* ══════════════════════════════════════════════════════════════
       🌊 DONOR HERO SECTION
    ══════════════════════════════════════════════════════════════ */
    .donor-hero {{
        background: linear-gradient(135deg, #0d2137, #1a5276, #117a65);
        border-radius: 20px;
        padding: 36px;
        color: white;
        text-align: center;
        margin-bottom: 28px;
    }}

    /* ══════════════════════════════════════════════════════════════
       💳 PAYMENT CARDS
    ══════════════════════════════════════════════════════════════ */
    .pay-card {{
        background: linear-gradient(145deg, #161b22, #0f141b);
        border: 1px solid #30363d;
        border-radius: 18px;
        padding: 26px 20px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
        margin-bottom: 10px;
    }}
    .pay-card:hover {{
        transform: translateY(-6px);
        border-color: rgb(216, 174, 52);
        box-shadow: 0 12px 30px rgba(0,0,0,0.4);
    }}
    /* دائرة زخرفية داخل الكارد */
    .pay-card::before {{
        content: "";
        position: absolute;
        top: -40%; left: -30%;
        width: 200px; height: 200px;
        background: rgba(216, 174, 52, 0.08);
        border-radius: 50%;
    }}
    .pay-icon {{ font-size: 3rem; margin-bottom: 10px; }}
    .pay-name {{ font-size: 1.1rem; font-weight: 800; color: #e6edf3; margin-bottom: 6px; }}
    .pay-desc {{ font-size: 0.85rem; color: #8b949e; }}

    /* ══════════════════════════════════════════════════════════════
       🟦 ALERT / INFO BOXES
    ══════════════════════════════════════════════════════════════ */
    .info-box {{
        background: #182131;
        border-{'right' if LANG=='ar' else 'left'}: 4px solid #384456;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
        color: #ffffff;
    }}
    .success-box {{
        background: #eafaf1;
        border-{'right' if LANG=='ar' else 'left'}: 4px solid var(--success);
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
    }}

    /* ══════════════════════════════════════════════════════════════
       🔘 RADIO BUTTONS
    ══════════════════════════════════════════════════════════════ */
    .stRadio input[type="radio"] {{
        accent-color: rgb(216, 174, 52);
    }}
    .stRadio label {{
        gap: 10px;
    }}

    /* ══════════════════════════════════════════════════════════════
       📐 COLUMN CENTER ALIGNMENT
    ══════════════════════════════════════════════════════════════ */
    div[data-testid="column"] {{
        display: flex;
        justify-content: center;
    }}

    /* ══════════════════════════════════════════════════════════════
       🔐 LOGIN SECTION — tab switcher + form area
    ══════════════════════════════════════════════════════════════ */

    /* إخفاء الـ tabs الافتراضية من Streamlit */
    div[data-testid="stTabs"] > div:first-child {{ display: none !important; }}

    /* wrapper أزرار التبديل */
    .tab-switcher {{
        display: flex;
        gap: 8px;
        justify-content: center;
        flex-wrap: wrap;
        margin-bottom: 28px;
    }}

    /* الـ form box المحدود العرض */
    .login-form-area {{
        max-width: 440px;
        margin: 0 auto;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 36px 32px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }}
    .login-form-area .stTextInput input {{
        background: #0d1117 !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
    }}
    .login-form-area .stTextInput input:focus {{
        border-color: var(--secondary) !important;
        box-shadow: 0 0 0 3px rgba(41,128,185,0.15) !important;
    }}

    /* زر الإرسال محدود العرض */
    .login-btn-wrap {{ max-width: 440px; margin: 12px auto 0; }}
    .login-btn-wrap .stButton > button {{
        width: 100%;
        padding: 13px !important;
        font-size: 1rem !important;
        border-radius: 12px !important;
        background: linear-gradient(135deg, #1a3a5c, #2980b9) !important;
        color: white !important;
    }}

    /* كارد المتبرع في صفحة اللوجين */
    .donor-login-card {{
        max-width: 440px;
        margin: 0 auto;
        background: linear-gradient(135deg, #293246, #131720);
        border-radius: 20px;
        padding: 36px 28px;
        color: white;
        text-align: center;
    }}

    /* hint box (الحساب الافتراضي) */
    .hint-box {{
        max-width: 440px;
        margin: 12px auto 0;
        background: rgba(41,128,185,0.12);
        border-right: 3px solid var(--secondary);
        border-radius: 10px;
        padding: 12px 16px;
        font-size: .82rem;
        color: #7fb3d3;
    }}

    /* ══════════════════════════════════════════════════════════════
       📱 RESPONSIVE — mobile breakpoints
    ══════════════════════════════════════════════════════════════ */
    @media (max-width: 768px) {{
        .main-header {{ padding: 20px !important; flex-direction: column; gap: 12px; }}
        .main-header h1 {{ font-size: 1.4rem !important; }}
        .kpi-grid {{ grid-template-columns: repeat(2,1fr) !important; }}
    }}
    @media (max-width: 640px) {{
        .login-form-area {{ padding: 24px 16px; }}
        .login-form-area,
        .login-btn-wrap,
        .donor-login-card,
        .hint-box {{ max-width: 100% !important; }}
    }}
    @media (max-width: 480px) {{
        .kpi-grid {{ grid-template-columns: 1fr !important; }}
    }}

    /* ══════════════════════════════════════════════════════════════
       🙈 HIDE STREAMLIT BRANDING
    ══════════════════════════════════════════════════════════════ */
    #MainMenu, footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 24px; padding-bottom: 40px; }}

    /* إخفاء الـ header بس مع استثناء زرار الـ sidebar */
    header {{
        visibility: hidden;
    }}
    header [data-testid="collapsedControl"],
    header [data-testid="stSidebarCollapseButton"],
    header button {{
        visibility: visible !important;
    }}

    /* ══════════════════════════════════════════════════════════════
       📐 SIDEBAR WIDTH — تثبيت العرض فقط
    ══════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {{
        min-width: 270px !important;
        max-width: 270px !important;
        width: 270px !important;
        transition: transform 0.3s ease !important;
    }}

    /* ══════════════════════════════════════════════════════════════
       🔘 SIDEBAR TOGGLE BUTTON
    ══════════════════════════════════════════════════════════════ */
    [data-testid="collapsedControl"] {{
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
        position: fixed !important;
        top: 12px !important;
        left: 12px !important;
        right: auto !important;
        z-index: 999999 !important;
        transform: none !important;
    }}
    [data-testid="collapsedControl"] button {{
        visibility: visible !important;
        background: #1a3a5c !important;
        border: 1px solid #d8ae34 !important;
        border-radius: 8px !important;
        padding: 6px 8px !important;
        cursor: pointer !important;
        transform: none !important;
    }}
    [data-testid="collapsedControl"] button:hover {{
        background: #2980b9 !important;
    }}
    [data-testid="collapsedControl"] svg {{
        fill: #e6edf3 !important;
        color: #e6edf3 !important;
        visibility: visible !important;
    }}

    /* زرار القفل جوا الـ sidebar */
    [data-testid="stSidebarCollapseButton"] {{
        visibility: visible !important;
        opacity: 1 !important;
    }}
    [data-testid="stSidebarCollapseButton"] button {{
        visibility: visible !important;
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(216,174,52,0.4) !important;
        border-radius: 8px !important;
        transform: none !important;
    }}
    [data-testid="stSidebarCollapseButton"] button:hover {{
        background: rgba(216,174,52,0.15) !important;
        transform: none !important;
    }}
    [data-testid="stSidebarCollapseButton"] svg {{
        fill: #e6edf3 !important;
        visibility: visible !important;
    }}

    /* ══════════════════════════════════════════════════════════════
       🌐 RTL FIX — تصحيح اتجاه الـ sidebar مع العربي
    ══════════════════════════════════════════════════════════════ */

    /* منع الـ RTL من التأثير على الـ sidebar نفسه */
    [data-testid="stSidebar"] {{
        direction: ltr !important;
    }}
    /* لكن المحتوى جواه يفضل RTL */
    [data-testid="stSidebar"] > div {{
        direction: {'rtl' if LANG == 'ar' else 'ltr'} !important;
    }}

    /* تأكد إن الـ sidebar بيتحرك لليسار صح */
    [data-testid="stSidebar"][aria-expanded="false"],
    [data-testid="stSidebar"].st-emotion-cache-collapsed {{
        transform: translateX(-270px) !important;
        margin-left: -270px !important;
    }}

    /* منع أي transform تاني من الـ RTL */
    .stApp > section:first-child {{
        direction: ltr !important;
    }}
    .stApp > section:first-child * {{
        direction: {'rtl' if LANG == 'ar' else 'ltr'} !important;
    }}
    /* استثناء الـ sidebar wrapper نفسه */
    .stApp > section[data-testid="stSidebar"] {{
        direction: ltr !important;
    }}

    /* ══════════════════════════════════════════════════════════════
       📱 MOBILE ONLY — الموبايل فقط
    ══════════════════════════════════════════════════════════════ */
    @media (max-width: 768px) {{
        .main-header {{ padding: 20px !important; flex-direction: column; gap: 12px; }}
        .main-header h1 {{ font-size: 1.4rem !important; }}
        .kpi-grid {{ grid-template-columns: repeat(2,1fr) !important; }}

        [data-testid="stSidebar"] {{
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            height: 100vh !important;
            z-index: 99998 !important;
        }}
        [data-testid="collapsedControl"] {{
            top: 8px !important;
            left: 8px !important;
        }}
        .main .block-container {{
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }}
    }}

    @media (max-width: 640px) {{
        .login-form-area {{ padding: 24px 16px; }}
        .login-form-area,
        .login-btn-wrap,
        .donor-login-card,
        .hint-box {{ max-width: 100% !important; }}
    }}
    @media (max-width: 480px) {{
        .kpi-grid {{ grid-template-columns: 1fr !important; }}
    }}
    
    </style>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# FIX 1 (continued): predict_cluster — أضفنا debug info وتحسين error handling
# ════════════════════════════════════════════════════════════════════════════
def predict_cluster(input_data: dict, models: dict) -> int:
    try:
        features = models.get("features")
        processor = models.get("processor")
        pca = models.get("pca")
        gmm = models.get("gmm")

        # لو أي موديل مش موجود نرجع -1
        if not all([features is not None, processor is not None,
                    pca is not None, gmm is not None]):
            return -1

        df_input = pd.DataFrame([input_data])

        # Keep only known features
        available = [f for f in features if f in df_input.columns]
        if not available:
            return -1

        df_input = df_input[available]

        # بعض الـ processors بتتوقع كل الـ features — نكمّل الناقص بصفر
        missing_features = [f for f in features if f not in df_input.columns]
        for mf in missing_features:
            df_input[mf] = 0
        df_input = df_input[features]  # رتّب الأعمدة بنفس ترتيب التدريب

        X = processor.transform(df_input)
        X_pca = pca.transform(X)
        cluster = gmm.predict(X_pca)[0]
        return int(cluster)
    except Exception:
        return -1

# ─── CHARTS (using plotly via st) ─────────────────────────────────────────────
def make_cluster_pie(df):
    import plotly.express as px
    if "cluster" not in df.columns or df.empty:
        return None
    counts = df["cluster"].value_counts().reset_index()
    counts.columns = ["cluster", "count"]
    counts["label"] = counts["cluster"].apply(lambda x: get_cluster_info(x, LANG)["name"])
    colors = [get_cluster_info(x, LANG)["color"] for x in counts["cluster"]]
    fig = px.pie(counts, values="count", names="label",
                 color_discrete_sequence=colors,
                 hole=0.45,
                 title=t("توزيع الحالات حسب الأولوية", "Case Distribution by Priority"))
    fig.update_layout(font_family="Cairo" if LANG=="ar" else "Poppins",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.1),
                      margin=dict(t=50, b=20))
    return fig

def make_gov_bar(df):
    import plotly.express as px
    if "governorate" not in df.columns or df.empty:
        return None
    gov = df["governorate"].value_counts().head(10).reset_index()
    gov.columns = ["gov", "count"]
    fig = px.bar(gov, x="count", y="gov", orientation="h",
                 color="count", color_continuous_scale="Blues",
                 title=t("أعلى 10 محافظات (عدد الحالات)", "Top 10 Governorates by Cases"))
    fig.update_layout(font_family="Cairo" if LANG=="ar" else "Poppins",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=False, yaxis_title="", xaxis_title=t("عدد الحالات","Cases"),
                      margin=dict(t=50, b=20))
    fig.update_coloraxes(showscale=False)
    return fig

def make_income_hist(df):
    import plotly.express as px
    if "monthly_income" not in df.columns or df.empty:
        return None
    fig = px.histogram(df, x="monthly_income", nbins=30,
                       color_discrete_sequence=["#2980b9"],
                       title=t("توزيع الدخل الشهري", "Monthly Income Distribution"))
    fig.update_layout(font_family="Cairo" if LANG=="ar" else "Poppins",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title=t("الدخل الشهري (جنيه)","Monthly Income (EGP)"),
                      yaxis_title=t("عدد الأسر","Families"),
                      margin=dict(t=50, b=20))
    return fig

def make_family_size_box(df):
    import plotly.express as px
    if "family_size" not in df.columns or "cluster" not in df.columns or df.empty:
        return None
    df2 = df.copy()
    df2["cluster_label"] = df2["cluster"].apply(lambda x: get_cluster_info(x, LANG)["name"])
    fig = px.box(df2, x="cluster_label", y="family_size",
                 color="cluster_label",
                 color_discrete_map={get_cluster_info(c, LANG)["name"]: get_cluster_info(c, LANG)["color"] for c in range(4)},
                 title=t("حجم الأسرة حسب التصنيف", "Family Size by Cluster"))
    fig.update_layout(font_family="Cairo" if LANG=="ar" else "Poppins",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=False, xaxis_title="", yaxis_title=t("حجم الأسرة","Family Size"),
                      margin=dict(t=50, b=20))
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# ─── LOGIN PAGE ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
def login_page():
    inject_css()

    # ── زر تقديم الحالة (أعلى الصفحة، عكس زر اللغة) ──
    col_submit_btn, _, col_lang_btn = st.columns([1, 7, 1])
 
    with col_submit_btn:
        if st.button(
            t("تقديم حالة", "Submit Case"),
            key="go_to_public_form",
            use_container_width=True,
        ):
            st.session_state["show_public_form"] = True
            st.rerun()
 
    with col_lang_btn:
        if st.button("ᯓ  " + ("English" if LANG == "ar" else "عربي"),
                     key="lang_toggle_login"):
            st.session_state.lang = "en" if LANG == "ar" else "ar"
            st.rerun()


    st.markdown(f"""
    <div style="text-align:center; margin: 20px 0 10px;">
        <div style="font-size:8rem;">🎗️</div>
        <h1 style="font-size:2.8rem; font-weight:900; color:#bf9e3f; margin:0; padding-{'right' if LANG == 'ar' else 'left'}: 30px">سند · SANAD</h1>
        <p style="color:#7f8c8d; font-size:1rem; margin-top:6px;">
            منظومة تحليل وإدارة بيانات الأسر المحتاجة<br>
            <span style="font-size:0.88rem;">Needy Families Data Analysis & Management System</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Tab state ──────────────────────────────────────────────────────────
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "login"
    tab = st.session_state.active_tab

    # ── أزرار التبديل — تسجيل دخول + متبرع فقط (إنشاء حساب أدمن انتقل للداخل) ──
    st.markdown("<div class='tab-switcher'>", unsafe_allow_html=True)
    left, c1, c2, right = st.columns([1.5, 1.4, 1.4, 1.5])

    with c1:
      if st.button(f" {t('تسجيل الدخول','Admin Login')}",
                 use_container_width=True,
                 type="primary" if tab == "login" else "secondary"):
        st.session_state.active_tab = "login"
        st.rerun()

    with c2:
      if st.button(f"{t('متبرع','Donor')}",
                 use_container_width=True,
                 type="primary" if tab == "donor" else "secondary"):
        st.session_state.active_tab = "donor"
        st.rerun()
    # ── المحتوى — محدود العرض في المنتصف ───────────────────────────────────
    _, mid, _ = st.columns([1, 2, 1])
    with mid:

        # ── تسجيل الدخول ───────────────────────────────────────────────
        if tab == "login":
            # ملاحظة: الـ inputs لازم تكون خارج الـ HTML div
            username = st.text_input(t("اسم المستخدم","Username"),
                                     key="login_user", placeholder="Ahmed_Safty")
            password = st.text_input(t("كلمة المرور","Password"),
                                     type="password", key="login_pass", placeholder="••••••••")

        # ── دخول كمتبرع ────────────────────────────────────────────────
        # else:
        #     st.markdown(f"""
        #     <div class='donor-login-card'>
        #         <div style='font-size:3rem;margin-bottom:10px;'>💛</div>
        #         <h3 style='margin:0 0 8px;font-size:1.3rem;'>
        #             {t('أهلاً بك كمتبرع كريم','Welcome, Generous Donor')}
        #         </h3>
        #         <p style='opacity:.8;font-size:.88rem;margin:0;'>
        #             {t('اطّلع على الإحصائيات وتبرع بدون تسجيل',
        #                'View statistics and donate without registration')}
        #         </p>
        #     </div>
        #     """, unsafe_allow_html=True)

    # ── أزرار الإرسال + رسائل الخطأ (خارج mid col عشان تتحكم في العرض) ───
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        st.markdown("<div class='login-btn-wrap'>", unsafe_allow_html=True)

        if tab == "login":
            if st.button(t("تسجيل الدخول","Sign In"),
                         use_container_width=True, type="primary"):
                if check_password(username, password):
                    st.session_state.logged_in = True
                    st.session_state.user_type  = "admin"
                    st.session_state.username   = username
                    st.rerun()
                else:
                    st.error(t("اسم المستخدم أو كلمة المرور غير صحيحة",
                               "Invalid username or password"))

        else:
            if st.button(t("دخول كمتبرع","Enter as Donor"),
                         use_container_width=True, type="primary"):
                st.session_state.logged_in  = True
                st.session_state.user_type  = "donor"
                st.session_state.username   = t("متبرع","Donor")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# ==============================================================================
# ==============================================================================
def public_case_submission():
    scroll_to_top()
    """واجهة تقديم الحالات للعامة — بدون تسجيل دخول."""
    inject_css()
    ensure_new_cases_file()
    models = load_models()
    # ── شريط العنوان مع زر الرجوع ──
    col_back, col_title = st.columns([1, 8])
    with col_back:
        if st.button(t("← رجوع", "← Back"), key="back_public"):
            st.session_state["show_public_form"] = False
            st.rerun()
 
    st.markdown(f"""
    <div class="main-header">
        <div>
            <h1>{t('تقديم طلب مساعدة', 'Submit Aid Request')}</h1>
            <p>{t('أدخل بياناتك وسنتواصل معك في أقرب وقت ممكن',
                   'Enter your data and we will contact you as soon as possible')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # ── قواميس الترجمة ──
    GENDER_MAP = {t("ذكر","Male"): "Male", t("أنثى","Female"): "Female"}
    RURAL_MAP  = {t("ريف","Rural"): "rural", t("حضر","Urban"): "urban"}
    INCOME_STABILITY_MAP = {
        t("مستقر","Stable"): "stable",
        t("غير مستقر","Unstable"): "unstable",
        t("غير منتظم","Irregular"): "irregular",
    }
    INCOME_SOURCE_MAP = {
        t("وظيفة","Employment"): "employment",
        t("عمل حر","Self Employment"): "self_employment",
        t("معاش","Pension"): "pension",
        t("مساعدات خيرية","Charity"): "charity",
        t("لا يوجد","None"): "none",
        t("أخرى","Other"): "other",
    }
    EDUCATION_MAP = {
        t("أمي","Illiterate"): "illiterate",
        t("ابتدائي","Primary"): "primary",
        t("إعدادي","Preparatory"): "preparatory",
        t("ثانوي","Secondary"): "secondary",
        t("جامعي","University"): "university",
        t("دراسات عليا","Postgraduate"): "postgraduate",
    }
    LITERACY_MAP = {
        t("أمي","Illiterate"): "illiterate",
        t("يقرأ ويكتب","Can Read"): "can_read",
        t("متعلم","Literate"): "literate",
    }
    EMPLOYMENT_MAP = {
        t("موظف","Employed"): "employed",
        t("عاطل","Unemployed"): "unemployed",
        t("عمل حر","Self Employed"): "self_employed",
        t("متقاعد","Retired"): "retired",
        t("عاجز عن العمل","Disabled"): "disabled",
    }
    HOUSING_MAP = {
        t("مملوك","Owned"): "owned",
        t("مستأجر","Rented"): "rented",
        t("عشوائي","Informal"): "informal",
        t("مع الأسرة","With Family"): "with_family",
    }
    BOOL_YES_NO = {t("نعم","Yes"): True, t("لا","No"): False}
 
    # قائمة المحافظات ثابتة (لأن المستخدم مش مسجل دخول)
    GOVERNORATES = [
        "القاهرة","الجيزة","الإسكندرية","الدقهلية","البحيرة","الشرقية",
        "المنوفية","الغربية","القليوبية","الفيوم","بني سويف","المنيا",
        "أسيوط","سوهاج","قنا","الأقصر","أسوان","البحر الأحمر","دمياط",
        "كفر الشيخ","مطروح","شمال سيناء","جنوب سيناء","السويس","الإسماعيلية",
        "بورسعيد","الوادي الجديد",
    ]
 
    with st.form("public_case_form", clear_on_submit=True):
 
        # ── البيانات الشخصية ──────────────────────────────────────────
        st.markdown(f"<div class='section-title'> {t('البيانات الشخصية','Personal Information')}</div>",
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        full_name = c1.text_input(t("الاسم الكامل *","Full Name *"))
        gender_d  = c2.selectbox(t("الجنس","Gender"), list(GENDER_MAP.keys()))
        ssn       = c3.text_input(t("الرقم القومي","National ID"))
 
        c4, c5, c6 = st.columns(3)
        phone      = c4.text_input(t("رقم الهاتف *","Phone Number *"))
        age        = c5.number_input(t("السن","Age"), 18, 100, 35)
        family_size = c6.number_input(t("حجم الأسرة","Family Size"), 1, 20, 4)
 
        c7, c8 = st.columns(2)
        number_of_children = c7.number_input(t("عدد الأطفال","Number of Children"), 0, 15, 2)
        children_in_school = c8.number_input(t("الأطفال في المدرسة","Children in School"), 0, 15, 0)
 
        # ── الموقع الجغرافي ──────────────────────────────────────────
        st.markdown(f"<div class='section-title'> {t('الموقع الجغرافي','Location')}</div>",
                    unsafe_allow_html=True)
        c9, c10, c11, c12 = st.columns(4)
        governorate = c9.selectbox(t("المحافظة","Governorate"), GOVERNORATES)
        center      = c10.text_input(t("المركز","Center"),
                                     placeholder=t("مثال: المنصورة","e.g. Mansoura"))
        village     = c11.text_input(t("القرية / الحي","Village / District"),
                                     placeholder=t("مثال: ميت غمر","e.g. Meet Ghamr"))
        rural_d     = c12.selectbox(t("ريف/حضر","Rural/Urban"), list(RURAL_MAP.keys()))
 
        # ── البيانات الاقتصادية ───────────────────────────────────────
        st.markdown(f"<div class='section-title'> {t('البيانات الاقتصادية','Economic Data')}</div>",
                    unsafe_allow_html=True)
        c13, c14, c15 = st.columns(3)
        monthly_income      = c13.number_input(t("الدخل الشهري (ج)","Monthly Income (EGP)"), 0, 100000, 0)
        income_stability_d  = c14.selectbox(t("استقرار الدخل","Income Stability"), list(INCOME_STABILITY_MAP.keys()))
        income_source_d     = c15.selectbox(t("مصدر الدخل","Income Source"), list(INCOME_SOURCE_MAP.keys()))
 
        c16, c17 = st.columns(2)
        expenses_estimate = c16.number_input(t("تقدير المصروفات","Expenses Estimate"), 0, 100000, 0)
        debt_amount       = c17.number_input(t("مبلغ الدين (ج)","Debt Amount (EGP)"), 0, 500000, 0)
        has_debt = debt_amount > 0
 
        # ── التعليم والتوظيف ──────────────────────────────────────────
        st.markdown(f"<div class='section-title'> {t('التعليم والتوظيف','Education & Employment')}</div>",
                    unsafe_allow_html=True)
        c18, c19, c20 = st.columns(3)
        education_d   = c18.selectbox(t("مستوى التعليم","Education Level"), list(EDUCATION_MAP.keys()))
        literacy_d    = c19.selectbox(t("محو الأمية","Literacy"), list(LITERACY_MAP.keys()))
        employment_d  = c20.selectbox(t("حالة التوظيف","Employment Status"), list(EMPLOYMENT_MAP.keys()))
 
        c21, c22, c23 = st.columns(3)
        years_of_experience   = c21.number_input(t("سنوات الخبرة","Years of Experience"), 0, 50, 0)
        has_stable_job_d      = c22.selectbox(t("وظيفة ثابتة؟","Stable Job?"), list(BOOL_YES_NO.keys()))
        willing_to_work_d     = c23.selectbox(t("مستعد للعمل؟","Willing to Work?"), list(BOOL_YES_NO.keys()))
 
        skills = st.text_input(t("المهارات","Skills"),
                               placeholder=t("مثال: نجارة، خياطة...","e.g., carpentry, sewing..."))
 
        # ── الصحة والمعيشة ───────────────────────────────────────────
        st.markdown(f"<div class='section-title'> {t('الصحة والمعيشة','Health & Living')}</div>",
                    unsafe_allow_html=True)
        c24, c25, c26 = st.columns(3)
        housing_d          = c24.selectbox(t("نوع السكن","Housing Type"), list(HOUSING_MAP.keys()))
        chronic_disease_d  = c25.selectbox(t("مرض مزمن؟","Chronic Disease?"), list(BOOL_YES_NO.keys()))
        disabled_member_d  = c26.selectbox(t("عضو من ذوي الهمم؟","Disabled Member?"), list(BOOL_YES_NO.keys()))
        medical_cost_estimate = st.number_input(t("تكلفة العلاج الشهرية","Monthly Medical Cost"), 0, 50000, 0)
 
        c27, c28, c29 = st.columns(3)
        access_to_water_d       = c27.selectbox(t("وصول للمياه؟","Access to Water?"), list(BOOL_YES_NO.keys()))
        access_to_electricity_d = c28.selectbox(t("وصول للكهرباء؟","Access to Electricity?"), list(BOOL_YES_NO.keys()))
        has_savings_d           = c29.selectbox(t("لديه مدخرات؟","Has Savings?"), list(BOOL_YES_NO.keys()))
 
        c30, c31 = st.columns(2)
        has_previous_business_d = c30.selectbox(t("تجربة عمل سابقة؟","Previous Business?"), list(BOOL_YES_NO.keys()))
        owns_assets_d           = c31.selectbox(t("يمتلك أصولاً؟","Owns Assets?"), list(BOOL_YES_NO.keys()))
 
        # ── زر الإرسال ──────────────────────────────────────────────
        submitted = st.form_submit_button(
            t("إرسال الطلب", "Submit Request"),
            type="primary",
            use_container_width=True,
        )
 
    if submitted:
        if not full_name:
            st.error(t("يرجى إدخال الاسم الكامل", "Please enter the full name"))
            return
        if not phone:
            st.error(t("يرجى إدخال رقم الهاتف", "Please enter the phone number"))
            return
 
        import math
        log_income = math.log(monthly_income + 1)
        log_debt   = math.log(debt_amount + 1)
        is_disabled   = 1 if BOOL_YES_NO[disabled_member_d] else 0
        health_burden = (1 if BOOL_YES_NO[chronic_disease_d] else 0) + is_disabled
        edu_map    = {"illiterate": 0, "primary": 1, "preparatory": 2,
                      "secondary": 3, "university": 4, "postgraduate": 5}
        edu_weight = edu_map.get(EDUCATION_MAP[education_d], 0)

        predict_input = {
            "full_name": full_name,
            "gender": GENDER_MAP[gender_d],
            "ssn": ssn,
            "phone_number": phone,
            "age": age,
            "family_size": family_size,
            "number_of_children": number_of_children,
            "governorate": governorate,
            "center": center,
            "village": village,
            "rural_or_urban": RURAL_MAP[rural_d],
            "housing_type": HOUSING_MAP[housing_d],
            "monthly_income": monthly_income,
            "income_stability": INCOME_STABILITY_MAP[income_stability_d],
            "income_source": INCOME_SOURCE_MAP[income_source_d],
            "expenses_estimate": expenses_estimate,
            "has_debt": has_debt,
            "debt_amount": debt_amount,
            "education_level_head": EDUCATION_MAP[education_d],
            "literacy": LITERACY_MAP[literacy_d],
            "children_in_school": children_in_school,
            "chronic_disease": BOOL_YES_NO[chronic_disease_d],
            "disabled_member": BOOL_YES_NO[disabled_member_d],
            "medical_cost_estimate": medical_cost_estimate,
            "employment_status": EMPLOYMENT_MAP[employment_d],
            "years_of_experience": years_of_experience,
            "skills": skills,
            "has_stable_job": BOOL_YES_NO[has_stable_job_d],
            "willing_to_work": BOOL_YES_NO[willing_to_work_d],
            "has_previous_business": BOOL_YES_NO[has_previous_business_d],
            "has_savings": BOOL_YES_NO[has_savings_d],
            "owns_assets": BOOL_YES_NO[owns_assets_d],
            "access_to_water": BOOL_YES_NO[access_to_water_d],
            "access_to_electricity": BOOL_YES_NO[access_to_electricity_d],
            "application_date": datetime.now().strftime("%Y-%m-%d"),
            "log_income": log_income,
            "log_debt": log_debt,
            "is_disabled": is_disabled,
            "health_burden": health_burden,
            "edu_weight": edu_weight,
        }

        # ── Predict cluster ──
        cluster_id = predict_cluster(predict_input, models)
        if cluster_id == -1:
            if monthly_income < 500:   cluster_id = 0
            elif monthly_income < 1500: cluster_id = 1
            elif monthly_income < 3000: cluster_id = 2
            else:                       cluster_id = 3

        # ── new_row بالأعمدة المطلوبة فقط ──
        new_row = {
            "full_name": full_name,
            "gender": GENDER_MAP[gender_d],
            "ssn": ssn,
            "phone_number": phone,
            "age": age,
            "family_size": family_size,
            "number_of_children": number_of_children,
            "governorate": governorate,
            "center": center,
            "village": village,
            "rural_or_urban": RURAL_MAP[rural_d],
            "housing_type": HOUSING_MAP[housing_d],
            "monthly_income": monthly_income,
            "income_stability": INCOME_STABILITY_MAP[income_stability_d],
            "income_source": INCOME_SOURCE_MAP[income_source_d],
            "expenses_estimate": expenses_estimate,
            "has_debt": has_debt,
            "debt_amount": debt_amount,
            "education_level_head": EDUCATION_MAP[education_d],
            "literacy": LITERACY_MAP[literacy_d],
            "children_in_school": children_in_school,
            "chronic_disease": BOOL_YES_NO[chronic_disease_d],
            "disabled_member": BOOL_YES_NO[disabled_member_d],
            "medical_cost_estimate": medical_cost_estimate,
            "employment_status": EMPLOYMENT_MAP[employment_d],
            "years_of_experience": years_of_experience,
            "skills": skills,
            "has_stable_job": BOOL_YES_NO[has_stable_job_d],
            "willing_to_work": BOOL_YES_NO[willing_to_work_d],
            "has_previous_business": BOOL_YES_NO[has_previous_business_d],
            "has_savings": BOOL_YES_NO[has_savings_d],
            "owns_assets": BOOL_YES_NO[owns_assets_d],
            "access_to_water": BOOL_YES_NO[access_to_water_d],
            "access_to_electricity": BOOL_YES_NO[access_to_electricity_d],
            "application_date": datetime.now().strftime("%Y-%m-%d"),
            "cluster": cluster_id,
        }

        try:
            existing = pd.read_csv(NEW_CASES_FILE, dtype=str)
            updated  = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
            updated.to_csv(NEW_CASES_FILE, index=False)
            cinfo = get_cluster_info(cluster_id, LANG)
            st.success(t(
                f"تم إرسال طلبك بنجاح يا {full_name}! سيتم التواصل معك على رقم {phone} في أقرب وقت.",
                f"Your request was submitted successfully, {full_name}! We will contact you at {phone} soon.",
            ))
            st.info(t(f"تصنيفك: {cinfo['name']}", f"Your classification: {cinfo['name']}"))
            st.balloons()
        except Exception as e:
            st.error(t(f"حدث خطأ أثناء الحفظ: {e}", f"Save error: {e}"))

# ==============================================================================
# ==============================================================================
# ==============================================================================

# ═══════════════════════════════════════════════════════════════════════════════
# ─── ADMIN REGISTER PAGE (inside admin dashboard) ─────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
def admin_register_page():
    scroll_to_top()
    st.markdown(f"""
    <div class="main-header">
        <div>
            <h1> {t('إنشاء حساب أدمن جديد', 'Create New Admin Account')}</h1>
            <p>{t('إضافة مستخدم جديد بصلاحيات الإدارة', 'Add a new user with admin privileges')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        # st.markdown(f"""
        # <div style='background:var(--card); border:1px solid var(--border); border-radius:20px; padding:36px 32px; box-shadow:0 8px 32px rgba(0,0,0,0.3);'>
        # <div style='text-align:center; margin-bottom:24px;'>
        #     <div style='font-size:3rem;'>🔐</div>
        #     <h3 style='color:var(--text); margin:8px 0 4px;'>{t('حساب أدمن جديد','New Admin Account')}</h3>
        #     <p style='color:var(--muted); font-size:0.88rem; margin:0;'>{t('أدخل بيانات الحساب الجديد','Enter new account details')}</p>
        # </div>
        # </div>
        # """, unsafe_allow_html=True)

        new_user = st.text_input(t("اسم المستخدم","Username"), key="sidebar_reg_user",
                                  placeholder=t("مثال: Ahmed_Admin","e.g. Ahmed_Admin"))
        ca, cb = st.columns(2)
        new_pass  = ca.text_input(t("كلمة المرور","Password"),
                                   type="password", key="sidebar_reg_pass")
        new_pass2 = cb.text_input(t("تأكيد كلمة المرور","Confirm Password"),
                                   type="password", key="sidebar_reg_pass2")

        st.markdown("<div style='margin-top:8px;'>", unsafe_allow_html=True)
        if st.button(t(" إنشاء الحساب"," Create Account"),
                     use_container_width=True, type="primary"):
            if not new_user or not new_pass:
                st.warning(t("يرجى ملء جميع الحقول","Please fill all fields"))
            elif new_pass != new_pass2:
                st.error(t("كلمتا المرور غير متطابقتين","Passwords do not match"))
            elif len(new_pass) < 6:
                st.warning(t("كلمة المرور 6 أحرف على الأقل","Minimum 6 characters"))
            elif register_user(new_user, new_pass):
                st.success(t(f" تم إنشاء حساب '{new_user}' بنجاح!",
                             f" Account '{new_user}' created successfully!"))
            else:
                st.error(t("اسم المستخدم موجود بالفعل","Username already exists"))
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ─── ADMIN PAGES ──────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def admin_home(df, models):
    scroll_to_top()
    st.markdown(f"""
    <div class="main-header">
        <div>
            <h1>{t('لوحة التحكم الرئيسية', 'Main Dashboard')}</h1>
            <p>{t('نظرة شاملة على بيانات الأسر المحتاجة ونتائج النماذج', 'Comprehensive overview of needy families data and model results')}</p>
        </div>
        <div style="text-align:{'left' if LANG=='ar' else 'right'}; opacity:0.8;">
            <div style="font-size:0.85rem;">{t('آخر تحديث','Last Update')}</div>
            <div style="font-weight:700;">{datetime.now().strftime('%Y-%m-%d')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.warning(t("لم يتم العثور على بيانات. تأكد من وجود ملف database.csv", "No data found. Make sure database.csv exists."))
        return

    # KPI Cards
    total = len(df)
    avg_income = df["monthly_income"].mean() if "monthly_income" in df.columns else 0
    avg_family = df["family_size"].mean() if "family_size" in df.columns else 0
    critical = (df["cluster"] == 0).sum() if "cluster" in df.columns else 0
    govs = df["governorate"].nunique() if "governorate" in df.columns else 0

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card" style="border-top-color:#334155;">
            <div class="icon">
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>
        <circle cx="9" cy="7" r="4"/>
        <path d="M22 21v-2a4 4 0 0 0-3-3.87"/>
        <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
    </svg>
</div>
            <div class="value">{total:,}</div>
            <div class="label">{t('إجمالي الحالات','Total Cases')}</div>
        </div>
        <div class="kpi-card" style="border-top-color:#334155;">
            <div class="icon">
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/>
        <line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>
</div>
            <div class="value">{critical:,}</div>
            <div class="label">{t('حالات حرجة','Critical Cases')}</div>
        </div>
        <div class="kpi-card" style="border-top-color:#334155;">
            <div class="icon">
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="12" y1="1" x2="12" y2="23"/>
        <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H7"/>
    </svg>
</div>
            <div class="value">{avg_income:,.0f}</div>
            <div class="label">{t('متوسط الدخل (جنيه)','Avg Income (EGP)')}</div>
        </div>
        <div class="kpi-card" style="border-top-color:#334155;">
            <div class="icon">
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20 10c0 6-8 12-8 12S4 16 4 10a8 8 0 0 1 16 0Z"/>
        <circle cx="12" cy="10" r="3"/>
    </svg>
</div>
            <div class="value">{govs}</div>
            <div class="label">{t('عدد المحافظات','Governorates')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Cluster Summary
    if "cluster" in df.columns:
        st.markdown(f"<div class='section-title'>{t('ملخص التصنيف (Clustering)','Clustering Summary')}</div>", unsafe_allow_html=True)
        cols = st.columns(len(CLUSTER_INFO))
        for i, (cid, info) in enumerate(CLUSTER_INFO.items()):
            count = (df["cluster"] == cid).sum()
            pct = count / total * 100
            cinfo = info[LANG]
            with cols[i]:
                st.markdown(f"""
                <div class="kpi-card" style="border-top-color:{cinfo['color']};">
                    <div class="value" style="font-size:1.6rem;color:{cinfo['color']};">{count:,}</div>
                    <div style="font-size:0.75rem;font-weight:700;color:{cinfo['color']};margin:4px 0;">{pct:.1f}%</div>
                    <div class="label" style="font-size:0.78rem;">{cinfo['name']}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
      with st.container(border=True):
        fig = make_cluster_pie(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with col2:
      with st.container(border=True):
        fig = make_gov_bar(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
      with st.container(border=True):
        fig = make_income_hist(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with col4:
      with st.container(border=True):
        fig = make_family_size_box(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


    # Additional Stats
    # 1. الحسابات
    sick = df["chronic_disease"].sum() if df["chronic_disease"].dtype in [int, float, bool] else (df["chronic_disease"].str.lower().isin(["yes","نعم","1","true"])).sum()
    dis = df["disabled_member"].sum() if df["disabled_member"].dtype in [int, float, bool] else (df["disabled_member"].str.lower().isin(["yes","نعم","1","true"])).sum()
    debt = df["has_debt"].sum() if df["has_debt"].dtype in [int, float, bool] else (df["has_debt"].str.lower().isin(["yes","نعم","1","true"])).sum()
    rural = (df["rural_or_urban"].str.lower().str.contains("rural|ريف", na=False)).sum()

# 2. عرض الكارد بالكامل كـ HTML واحد (بداخلها الإحصائيات)
    st.markdown(f"""
<div class='section-card'>
<div class='section-title'>{t('إحصائيات إضافية','Additional Statistics')}</div>
<div style='display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;'>
<div style='flex: 1; min-width: 100px; text-align: center;'>
<div style='font-size: 1.3rem; color: var(--muted);'>{t("أمراض مزمنة","Chronic Disease")}</div>
<div style='font-size: 1.7rem; font-weight: 800; color: var(--primary);'>{int(sick):,}</div>
</div>
<div style='flex: 1; min-width: 100px; text-align: center;'>
<div style='font-size: 1.3rem; color: var(--muted);'>{t("ذوو الإعاقة","Disabled Members")}</div>
<div style='font-size: 1.7rem; font-weight: 800; color: var(--primary);'>{int(dis):,}</div>
</div>
<div style='flex: 1; min-width: 100px; text-align: center;'>
<div style='font-size: 1.3rem; color: var(--muted);'>{t("لديهم ديون","Have Debts")}</div>
<div style='font-size: 1.7rem; font-weight: 800; color: var(--primary);'>{int(debt):,}</div>
</div>
<div style='flex: 1; min-width: 100px; text-align: center;'>
<div style='font-size: 1.3rem; color: var(--muted);'>{t("الريف","Rural")}</div>
<div style='font-size: 1.7rem; font-weight: 800; color: var(--primary);'>{int(rural):,}</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)


def admin_cases(df):
    scroll_to_top()
    # ── اختيار مصدر البيانات ──
    if "cases_source" not in st.session_state:
        st.session_state.cases_source = "main"

    st.markdown(f"""
    <div class="main-header">
        <div>
            <h1> {t('إدارة الحالات', 'Case Management')}</h1>
            <p>{t('عرض وبحث في جميع الحالات المسجلة', 'View and search all registered cases')}</p>
        </div>
        <div style="display:flex; gap:12px; align-items:center;">
            <div id="src-btns" style="display:flex; gap:10px;">
    """, unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns([1, 1])

    is_main = st.session_state.cases_source == "main"
    is_new = st.session_state.cases_source == "new"

    with btn_col1:
        if st.button(
            t("قاعدة البيانات", "Main Database"),
            key="src_main",
            use_container_width=True,
            type="primary" if is_main else "secondary"
        ):
            st.session_state.cases_source = "main"
            st.rerun()

    with btn_col2:
        if st.button(
            t("الحالات الجديدة", "New Cases"),
            key="src_new",
            use_container_width=True,
            type="primary" if is_new else "secondary"
        ):
            st.session_state.cases_source = "new"
            st.rerun()
    st.markdown("</div></div></div>", unsafe_allow_html=True)

    # CSS ديناميكي للزر المحدد
    active_key = "src_main" if st.session_state.cases_source == "main" else "src_new"
    st.markdown(f"""
    <style>
    div[data-testid="stButton"] button[kind="secondary"]:nth-of-type(1) {{}}
    /* تلوين الزر النشط */
    div[data-testid="column"]:has(div[data-testid="stButton"] button:focus) button {{
        border-color: #d8ae34 !important;
        color: #d8ae34 !important;
    }}
    </style>
    <script>
    const btns = window.parent.document.querySelectorAll('button');
    btns.forEach(b => {{
        if (b.innerText.includes('{"قاعدة البيانات" if st.session_state.cases_source == "main" else "الحالات الجديدة"}') ||
            b.innerText.includes('{"Main Database" if st.session_state.cases_source == "main" else "New Cases"}')) {{
            b.style.borderColor = '#d8ae34';
            b.style.color = '#d8ae34';
            b.style.boxShadow = '0 0 10px rgba(216,174,52,0.45)';
        }}
    }});
    </script>
    """, unsafe_allow_html=True)

    # ── تحميل البيانات حسب الاختيار ──
    if st.session_state.cases_source == "new":
        try:
            df = pd.read_csv("Database/new_cases.csv", dtype={"ssn": str})
            df["ssn"] = df["ssn"].str.replace(".0", "", regex=False).str.strip()
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        st.warning(t("لا توجد بيانات", "No data available"))
        return

    # Search bar
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        search_q = st.text_input(t(" بحث بالاسم أو الرقم القومي", " Search by Name or SSN"),
                                 placeholder=t("اكتب اسم الحالة أو الرقم القومي...","Enter name or national ID..."),
                                 key="case_search")
    with col2:
        gov_filter = st.selectbox(t("فلتر بالمحافظة", "Filter by Governorate"),
                                  [t("الكل","All")] + sorted(df["governorate"].dropna().unique().tolist()) if "governorate" in df.columns else [t("الكل","All")])
    with col3:
        cluster_filter = st.selectbox(t("التصنيف", "Cluster"),
                                      [t("الكل","All")] + [f"Cluster {i}" for i in range(4)])
    st.markdown("</div>", unsafe_allow_html=True)

    filtered = df.copy()
    if search_q:
        mask = pd.Series([False] * len(filtered))
        if "full_name" in filtered.columns:
            mask |= filtered["full_name"].astype(str).str.contains(search_q, case=False, na=False)
        if "ssn" in filtered.columns:
            mask |= filtered["ssn"].astype(str).str.contains(search_q, case=False, na=False)
        filtered = filtered[mask]

    if gov_filter != t("الكل","All") and "governorate" in filtered.columns:
        filtered = filtered[filtered["governorate"] == gov_filter]

    if cluster_filter != t("الكل","All") and "cluster" in filtered.columns:
        cid = int(cluster_filter.split(" ")[-1])
        filtered = filtered[filtered["cluster"] == cid]

    st.markdown(f"<div class='info-box'>{t(f'عدد النتائج: {len(filtered):,} حالة', f'Results: {len(filtered):,} cases')}</div>", unsafe_allow_html=True)

    # Display table
    display_cols = [
    "full_name",
    "ssn",
    "phone_number",
    "age",
    "gender",
    "governorate",
    "center",
    "village",
    "family_size",
    "number_of_children",
    "monthly_income",
    "expenses_estimate",
    "medical_cost_estimate",
    "debt_amount",
    "employment_status",
    "education_level_head",
    "children_in_school",
    "chronic_disease",
    "disabled_member",
    "has_stable_job",
    "willing_to_work",
    "cluster",
    "application_date",
]

    available_cols = [c for c in display_cols if c in filtered.columns]
    show_df = filtered[available_cols].copy()

# تنظيف الرقم القومي
    if "ssn" in show_df.columns:
        show_df["ssn"] = (
            show_df["ssn"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.strip()
        )

    if "cluster" in show_df.columns:
        show_df["cluster_label"] = show_df["cluster"].apply(
            lambda x: get_cluster_info(x, LANG)["name"] if pd.notna(x) else ""
        )

        show_df["cluster_id"] = show_df["cluster"]

        show_df = show_df.drop(columns=["cluster"])
    
    if "application_date" in show_df.columns:
        show_df["application_date"] = pd.to_datetime(
            show_df["application_date"],
            errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    # ── تحويل True/False لـ نعم/لا أو Yes/No للعرض فقط ──
    bool_cols = ["chronic_disease", "disabled_member", "has_stable_job", "willing_to_work"]
    for bc in bool_cols:
        if bc in show_df.columns:
            show_df[bc] = show_df[bc].apply(
                lambda v: t("نعم", "Yes") if str(v).lower() in ["true","1","yes","نعم"] else t("لا", "No")
            )

    col_rename = {
    "full_name": t("الاسم", "Name"),
    "ssn": t("الرقم القومي", "SSN"),
    "phone_number": t("رقم الهاتف", "Phone"),
    "age": t("السن", "Age"),
    "gender": t("الجنس", "Gender"),
    "governorate": t("المحافظة", "Governorate"),
    "center": t("المركز", "Center"),
    "village": t("القرية", "Village"),
    "family_size": t("حجم الأسرة", "Family Size"),
    "number_of_children": t("عدد الأطفال", "Children"),
    "monthly_income": t("الدخل", "Income"),
    "expenses_estimate": t("المصروفات", "Expenses"),
    "medical_cost_estimate": t("تكلفة العلاج", "Medical Cost"),
    "debt_amount": t("الديون", "Debt"),
    "employment_status": t("حالة العمل", "Employment"),
    "education_level_head": t("مستوى التعليم", "Education"),
    "children_in_school": t("الأطفال في التعليم", "Children In School"),
    "chronic_disease": t("مرض مزمن", "Chronic Disease"),
    "disabled_member": t("فرد من ذوي الهمم", "Disabled Member"),
    "has_stable_job": t("وظيفة مستقرة", "Stable Job"),
    "willing_to_work": t("قابل للعمل", "Willing To Work"),
    "cluster_label": t("التصنيف", "Classification"),
    "cluster_id": t("رقم التصنيف", "Cluster ID"),
    "application_date": t("تاريخ التسجيل", "Application Date"),
}

    show_df.rename(
        columns={k: v for k, v in col_rename.items() if k in show_df.columns},
        inplace=True
    )

    st.dataframe(
        show_df,
        use_container_width=True,
        height=480,
        column_config={
        t("الدخل", "Income"): st.column_config.NumberColumn(format="EGP %.0f"),
        t("المصروفات", "Expenses"): st.column_config.NumberColumn(format="EGP %.0f"),
        t("الديون", "Debt"): st.column_config.NumberColumn(format="EGP %.0f"),
    }
    )
    # Detail view on search match
    if search_q and len(filtered) == 1:
        row = filtered.iloc[0]
        cid = int(row["cluster"]) if "cluster" in row and pd.notna(row["cluster"]) else -1
        cinfo = get_cluster_info(cid, LANG) if cid >= 0 else {"name": t("غير محدد","Unknown"), "desc": "", "color": "#95a5a6"}

        st.markdown(f"""
        <div class='section-card'>
            <div class='section-title'>👤 {t('تفاصيل الحالة','Case Details')}</div>
            <div style='display:flex; gap:20px; flex-wrap:wrap; align-items:center; margin-bottom:16px;'>
                <div>
                    <h2 style='margin:0; color:#1a3a5c;'>{row.get('full_name','N/A')}</h2>
                    <div style='color:#7f8c8d;'>{t('الرقم القومي:','SSN:')} {row.get('ssn','N/A')}</div>
                </div>
                <span class='cluster-badge' style='background:{cinfo["color"]};'>{cinfo['name']}</span>
            </div>
            <p style='color:#555;'>{cinfo['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric(t("الدخل الشهري","Monthly Income"), f"EGP {row.get('monthly_income', 0):,.0f}")
        c2.metric(t("حجم الأسرة","Family Size"), row.get('family_size', 'N/A'))
        c3.metric(t("عدد الأطفال","Children"), row.get('number_of_children', 'N/A'))


def admin_add_case(df, models):
    scroll_to_top()
    st.markdown(f"""
    <div class="main-header">
        <div>
            <h1> {t('إضافة حالة جديدة', 'Add New Case')}</h1>
            <p>{t('إدخال بيانات أسرة جديدة والحصول على نتيجة التصنيف', 'Enter new family data and get classification result')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    SAVE_OPTIONS = [
        t(" تصنيف فقط (بدون حفظ)", " Classify Only (No Save)"),
        t(" تصنيف وحفظ في قاعدة البيانات", " Classify & Save to Database"),
    ]
    SAVE_INDEX_SAVE = 1

    # ── قواميس الترجمة للقيم المخزنة (عرض ← قيمة مخزنة) ──
    GENDER_MAP = {
        t("ذكر","Male"): "Male",
        t("أنثى","Female"): "Female",
    }
    RURAL_MAP = {
        t("ريف","Rural"): "rural",
        t("حضر","Urban"): "urban",
    }
    INCOME_STABILITY_MAP = {
        t("مستقر","Stable"): "stable",
        t("غير مستقر","Unstable"): "unstable",
        t("غير منتظم","Irregular"): "irregular",
    }
    INCOME_SOURCE_MAP = {
        t("وظيفة","Employment"): "employment",
        t("عمل حر","Self Employment"): "self_employment",
        t("معاش","Pension"): "pension",
        t("مساعدات خيرية","Charity"): "charity",
        t("لا يوجد","None"): "none",
        t("أخرى","Other"): "other",
    }
    EDUCATION_MAP = {
        t("أمي","Illiterate"): "illiterate",
        t("ابتدائي","Primary"): "primary",
        t("إعدادي","Preparatory"): "preparatory",
        t("ثانوي","Secondary"): "secondary",
        t("جامعي","University"): "university",
        t("دراسات عليا","Postgraduate"): "postgraduate",
    }
    LITERACY_MAP = {
        t("أمي","Illiterate"): "illiterate",
        t("يقرأ ويكتب","Can Read"): "can_read",
        t("متعلم","Literate"): "literate",
    }
    EMPLOYMENT_MAP = {
        t("موظف","Employed"): "employed",
        t("عاطل","Unemployed"): "unemployed",
        t("عمل حر","Self Employed"): "self_employed",
        t("متقاعد","Retired"): "retired",
        t("عاجز عن العمل","Disabled"): "disabled",
    }
    HOUSING_MAP = {
        t("مملوك","Owned"): "owned",
        t("مستأجر","Rented"): "rented",
        t("عشوائي","Informal"): "informal",
        t("مع الأسرة","With Family"): "with_family",
    }
    BOOL_YES_NO = {
        t("نعم","Yes"): True,
        t("لا","No"): False,
    }

    with st.form("add_case_form"):
        st.markdown(f"<div class='section-title'> {t('البيانات الشخصية','Personal Information')}</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        full_name = c1.text_input(t("الاسم الكامل *","Full Name *"))
        gender_display = c2.selectbox(t("الجنس","Gender"), list(GENDER_MAP.keys()))
        ssn = c3.text_input(t("الرقم القومي","National ID"))

        c4, c5, c6 = st.columns(3)
        phone = c4.text_input(t("رقم الهاتف","Phone Number"))
        age = c5.number_input(t("السن","Age"), 18, 100)
        family_size = c6.number_input(t("حجم الأسرة","Family Size"), 1, 20, 4)

        c7, c8 = st.columns(2)
        number_of_children = c7.number_input(t("عدد الأطفال","Number of Children"), 0, 15, 2)
        children_in_school = c8.number_input(t("الأطفال في المدرسة","Children in School"), 0, 15, 0)

        st.markdown(f"<div class='section-title'> {t('الموقع الجغرافي','Location')}</div>", unsafe_allow_html=True)
        c9, c10, c11, c12 = st.columns(4)
        gov_options = sorted(df["governorate"].dropna().unique()) if not df.empty else []
        governorate = c9.selectbox(t("المحافظة","Governorate"), gov_options)

        center_options = sorted(df["center"].dropna().unique()) if not df.empty else []
        center = c10.selectbox(t("المركز","Center"), center_options)

        village_options = sorted(df["village"].dropna().unique()) if not df.empty else []
        village = c11.selectbox(t("القرية","Village"), village_options)

        rural_display = c12.selectbox(t("ريف/حضر","Rural/Urban"), list(RURAL_MAP.keys()))

        st.markdown(f"<div class='section-title'> {t('البيانات الاقتصادية','Economic Data')}</div>", unsafe_allow_html=True)
        c13, c14, c15 = st.columns(3)
        monthly_income = c13.number_input(t("الدخل الشهري (ج)","Monthly Income (EGP)"), 0, 100000, 1500)
        income_stability_display = c14.selectbox(t("استقرار الدخل","Income Stability"), list(INCOME_STABILITY_MAP.keys()))
        income_source_display = c15.selectbox(t("مصدر الدخل","Income Source"), list(INCOME_SOURCE_MAP.keys()))

        c16, c17 = st.columns(2)
        expenses_estimate = c16.number_input(
            t("تقدير المصروفات","Expenses Estimate"),
            0, 100000, 1200
        )
        debt_amount = c17.number_input(
            t("مبلغ الدين (ج)","Debt Amount (EGP)"),
            0, 500000, 0
        )
        has_debt = debt_amount > 0

        st.markdown(f"<div class='section-title'> {t('التعليم والتوظيف','Education & Employment')}</div>", unsafe_allow_html=True)
        c18, c19, c20 = st.columns(3)
        education_display = c18.selectbox(t("مستوى التعليم","Education Level"), list(EDUCATION_MAP.keys()))
        literacy_display = c19.selectbox(t("محو الأمية","Literacy"), list(LITERACY_MAP.keys()))
        employment_display = c20.selectbox(t("حالة التوظيف","Employment Status"), list(EMPLOYMENT_MAP.keys()))

        c21, c22, c23 = st.columns(3)
        years_of_experience = c21.number_input(t("سنوات الخبرة","Years of Experience"), 0, 50, 0)
        has_stable_job_display = c22.selectbox(t("وظيفة ثابتة؟","Stable Job?"), list(BOOL_YES_NO.keys()))
        willing_to_work_display = c23.selectbox(t("مستعد للعمل؟","Willing to Work?"), list(BOOL_YES_NO.keys()))

        st.markdown(f"<div class='section-title'> {t('الصحة والمعيشة','Health & Living')}</div>", unsafe_allow_html=True)
        c24, c25, c26 = st.columns(3)
        housing_display = c24.selectbox(t("نوع السكن","Housing Type"), list(HOUSING_MAP.keys()))
        chronic_disease_display = c25.selectbox(t("مرض مزمن؟","Chronic Disease?"), list(BOOL_YES_NO.keys()))
        disabled_member_display = c26.selectbox(t("عضو معاق؟","Disabled Member?"), list(BOOL_YES_NO.keys()))
        medical_cost_estimate = st.number_input(t("تكلفة العلاج الشهرية","Monthly Medical Cost"), 0, 50000, 0)

        c27, c28, c29 = st.columns(3)
        access_to_water_display = c27.selectbox(t("وصول للمياه؟","Access to Water?"), list(BOOL_YES_NO.keys()))
        access_to_electricity_display = c28.selectbox(t("وصول للكهرباء؟","Access to Electricity?"), list(BOOL_YES_NO.keys()))
        has_savings_display = c29.selectbox(t("لديه مدخرات؟","Has Savings?"), list(BOOL_YES_NO.keys()))

        c30, c31 = st.columns(2)
        has_previous_business_display = c30.selectbox(t("تجربة عمل سابقة؟","Previous Business?"), list(BOOL_YES_NO.keys()))
        owns_assets_display = c31.selectbox(t("يمتلك أصولاً؟","Owns Assets?"), list(BOOL_YES_NO.keys()))
        skills = st.text_input(t("المهارات","Skills"), placeholder=t("مثال: نجارة، خياطة...","e.g., carpentry, sewing..."))

        st.markdown(f"<div class='section-title'> {t('خيارات الحفظ','Save Options')}</div>", unsafe_allow_html=True)
        save_index = st.radio(
            t("اختر العملية", "Choose Action"),
            options=SAVE_OPTIONS,
            index=0
        )

        submitted = st.form_submit_button(t(" تنفيذ", " Execute"), type="primary", use_container_width=True)

    if submitted:
        if not full_name:
            st.error(t("يرجى إدخال الاسم الكامل", "Please enter the full name"))
            return

        # ── تحويل القيم المعروضة لقيم التخزين ──
        gender           = GENDER_MAP[gender_display]
        rural_or_urban   = RURAL_MAP[rural_display]
        income_stability = INCOME_STABILITY_MAP[income_stability_display]
        income_source    = INCOME_SOURCE_MAP[income_source_display]
        education_level_head = EDUCATION_MAP[education_display]
        literacy         = LITERACY_MAP[literacy_display]
        employment_status = EMPLOYMENT_MAP[employment_display]
        housing_type     = HOUSING_MAP[housing_display]
        chronic_disease       = BOOL_YES_NO[chronic_disease_display]
        disabled_member       = BOOL_YES_NO[disabled_member_display]
        has_stable_job        = BOOL_YES_NO[has_stable_job_display]
        willing_to_work       = BOOL_YES_NO[willing_to_work_display]
        access_to_water       = BOOL_YES_NO[access_to_water_display]
        access_to_electricity = BOOL_YES_NO[access_to_electricity_display]
        has_savings           = BOOL_YES_NO[has_savings_display]
        has_previous_business = BOOL_YES_NO[has_previous_business_display]
        owns_assets           = BOOL_YES_NO[owns_assets_display]

        import math
        log_income = math.log(monthly_income + 1)
        log_debt = math.log(debt_amount + 1)
        is_disabled = 1 if disabled_member else 0
        health_burden = (1 if chronic_disease else 0) + is_disabled
        edu_map = {"illiterate": 0, "primary": 1, "preparatory": 2, "secondary": 3, "university": 4, "postgraduate": 5}
        edu_weight = edu_map.get(education_level_head, 0)

        new_case = {
            "full_name": full_name, "gender": gender, "ssn": ssn, "phone_number": phone,
            "age": age, "family_size": family_size, "number_of_children": number_of_children,
            "governorate": governorate, "center": center, "village": village,
            "rural_or_urban": rural_or_urban, "housing_type": housing_type,
            "monthly_income": monthly_income, "income_stability": income_stability,
            "income_source": income_source, "expenses_estimate": expenses_estimate,
            "has_debt": has_debt, "debt_amount": debt_amount,
            "education_level_head": education_level_head, "literacy": literacy,
            "children_in_school": children_in_school, "chronic_disease": chronic_disease,
            "disabled_member": disabled_member, "medical_cost_estimate": medical_cost_estimate,
            "employment_status": employment_status, "years_of_experience": years_of_experience,
            "skills": skills, "has_stable_job": has_stable_job, "willing_to_work": willing_to_work,
            "has_previous_business": has_previous_business, "has_savings": has_savings,
            "owns_assets": owns_assets, "access_to_water": access_to_water,
            "access_to_electricity": access_to_electricity,
            "application_date": datetime.now().strftime("%Y-%m-%d"),
            "log_income": log_income, "log_debt": log_debt,
            "is_disabled": is_disabled, "health_burden": health_burden, "edu_weight": edu_weight
        }

        cluster_id = predict_cluster(new_case, models)
        if cluster_id == -1:
            # Fallback simple heuristic
            if monthly_income < 500: cluster_id = 0
            elif monthly_income < 1500: cluster_id = 1
            elif monthly_income < 3000: cluster_id = 2
            else: cluster_id = 3

        new_case["cluster"] = cluster_id
        cinfo = get_cluster_info(cluster_id, LANG)

        st.markdown(f"""
        <div class='section-card' style='border-top: 4px solid {cinfo["color"]};'>
            <div class='section-title'> {t('نتيجة التصنيف','Classification Result')}</div>
            <div style='text-align:center; padding: 20px;'>
                <span class='cluster-badge' style='background:{cinfo["color"]}; font-size:1.2rem; padding: 12px 28px;'>{cinfo['name']}</span>
                <p style='color:#555; margin-top:16px; font-size:1rem;'>{cinfo['desc']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if save_index == SAVE_OPTIONS[SAVE_INDEX_SAVE]:
            new_df = pd.concat([df, pd.DataFrame([new_case])], ignore_index=True)
            try:
                new_df.to_csv("Database/database.csv", index=False)
                st.success(t(
                    f"تم حفظ الحالة بنجاح! التصنيف: {cinfo['name']}",
                    f"Case saved successfully! Classification: {cinfo['name']}"
                ))
                st.cache_data.clear()
            except Exception as e:
                st.error(t(f"خطأ في الحفظ: {e}", f"Save error: {e}"))
        else:
            st.info(t(
                "ℹتم التصنيف فقط — لم يتم الحفظ في قاعدة البيانات.",
                "ℹClassified only — not saved to database."
            ))


# ═══════════════════════════════════════════════════════════════════════════════
# ─── DONOR PAGES ──────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def donor_stats(df):
    scroll_to_top()
    st.markdown(f"""
    <div class="main-header">
        <div>
            <h1> {t('إحصائيات الأسر المحتاجة', 'Needy Families Statistics')}</h1>
            <p>{t('بياناتك تُحدث فارقاً حقيقياً في حياة الأسر المحتاجة', 'Your donation makes a real difference in needy families lives')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.warning(t("لا توجد بيانات متاحة", "No data available"))
        return

    total = len(df)
    critical = (df["cluster"] == 0).sum() if "cluster" in df.columns else 0
    govs = df["governorate"].nunique() if "governorate" in df.columns else 0
    children = df["number_of_children"].sum() if "number_of_children" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t(" إجمالي الأسر"," Total Families"), f"{total:,}")
    c2.metric(t(" حالات حرجة"," Critical Cases"), f"{critical:,}", delta=f"{critical/total*100:.1f}%")
    c3.metric(t(" المحافظات"," Governorates"), f"{govs}")
    c4.metric(t(" إجمالي الأطفال"," Total Children"), f"{children:,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            fig = make_cluster_pie(df)
            if fig: st.plotly_chart(fig, use_container_width=True)
    with col2:
        with st.container(border=True):
            fig = make_gov_bar(df)
            if fig: st.plotly_chart(fig, use_container_width=True)
            
    # Impact message
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0d2137,#117a65); border-radius:16px; padding:32px; color:white; text-align:center; margin-top:20px;'>
        <div style='font-size:2rem; margin-bottom:8px;'>𐦂𖨆𐀪𖠋𐀪𐀪</div>
        <h3 style='margin:0;'>{t('كل تبرع يصل مباشرة للأسر الأكثر احتياجاً','Every donation goes directly to the most needy families')}</h3>
        <p style='opacity:0.8; margin-top:8px;'>{t(f'يوجد {critical:,} أسرة تنتظر مساعدتك الآن',f'There are {critical:,} families waiting for your help now')}</p>
    </div>
    """, unsafe_allow_html=True)


def donor_map(df):
    scroll_to_top()
    st.markdown(f"""
    <div class="main-header">
        <div>
            <h1> {t('خريطة الاحتياج','Need Map')}</h1>
            <p>{t('ابحث عن المنطقة لمعرفة مستوى الاحتياج','Search by area to know the need level')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.warning(t("لا توجد بيانات", "No data"))
        return

    col1, col2, col3 = st.columns(3)
    govs = [t("الكل","All")] + sorted(df["governorate"].dropna().unique().tolist()) if "governorate" in df.columns else [t("الكل","All")]
    sel_gov = col1.selectbox(t("المحافظة","Governorate"), govs)
    
    centers = [t("الكل","All")]
    if sel_gov != t("الكل","All") and "center" in df.columns:
        centers += sorted(df[df["governorate"]==sel_gov]["center"].dropna().unique().tolist())
    sel_center = col2.selectbox(t("المركز","Center"), centers)

    villages = [t("الكل","All")]
    filtered = df.copy()
    if sel_gov != t("الكل","All") and "governorate" in df.columns:
        filtered = filtered[filtered["governorate"]==sel_gov]
    if sel_center != t("الكل","All") and "center" in df.columns:
        filtered = filtered[filtered["center"]==sel_center]
        if "village" in df.columns:
            villages += sorted(filtered["village"].dropna().unique().tolist())
    sel_village = col3.selectbox(t("القرية","Village"), villages)
    if sel_village != t("الكل","All") and "village" in df.columns:
        filtered = filtered[filtered["village"]==sel_village]

    if filtered.empty:
        st.info(t("لا توجد بيانات لهذه المنطقة","No data for this area"))
        return

    total_area = len(filtered)
    critical_area = (filtered["cluster"]==0).sum() if "cluster" in filtered.columns else 0
    avg_inc = filtered["monthly_income"].mean() if "monthly_income" in filtered.columns else 0
    avg_fam = filtered["family_size"].mean() if "family_size" in filtered.columns else 0

    st.markdown(f"""
    <div class="kpi-grid" style="grid-template-columns:repeat(4,1fr);">
        <div class="kpi-card"><div class="value">{total_area:,}</div><div class="label">{t('إجمالي الأسر','Total Families')}</div></div>
        <div class="kpi-card" style="border-top-color:#e74c3c;"><div class="value" style="color:#e74c3c;">{critical_area:,}</div><div class="label">{t('حالات حرجة','Critical Cases')}</div></div>
        <div class="kpi-card"><div class="value">{avg_inc:,.0f}</div><div class="label">{t('متوسط الدخل','Avg Income EGP')}</div></div>
        <div class="kpi-card"><div class="value">{avg_fam:.1f}</div><div class="label">{t('متوسط حجم الأسرة','Avg Family Size')}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Cluster breakdown
    if "cluster" in filtered.columns:
        st.markdown(f"<div class='section-title'> {t('توزيع التصنيف في هذه المنطقة','Cluster Distribution in This Area')}</div>", unsafe_allow_html=True)
        for cid in range(4):
            cnt = (filtered["cluster"]==cid).sum()
            cinfo = get_cluster_info(cid, LANG)
            pct = cnt / total_area * 100 if total_area > 0 else 0
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:12px; margin:8px 0;'>
                <div style='width:180px; font-size:0.85rem;'>{cinfo['name']}</div>
                <div style='flex:1; background:#eee; border-radius:8px; height:20px; overflow:hidden;'>
                    <div style='width:{pct:.1f}%; background:{cinfo["color"]}; height:100%; border-radius:8px; transition:width 0.5s;'></div>
                </div>
                <div style='width:80px; font-size:0.85rem; color:#555;'>{cnt:,} ({pct:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)

    # Vision box
    need_score = (critical_area / total_area * 100) if total_area > 0 else 0
    if need_score >= 50:
        vision_ar = f" هذه المنطقة تعاني من أزمة احتياج شديدة. {critical_area:,} أسرة تحتاج تدخلاً فورياً وشاملاً."
        vision_en = f" This area faces a severe need crisis. {critical_area:,} families need immediate comprehensive intervention."
        color = "#e74c3c"
    elif need_score >= 25:
        vision_ar = f" مستوى الاحتياج مرتفع في هذه المنطقة. برامج الدعم والتمكين الاقتصادي ضرورية."
        vision_en = f" High need level in this area. Support and economic empowerment programs are needed."
        color = "#e67e22"
    else:
        vision_ar = f" الوضع في هذه المنطقة أفضل نسبياً. التركيز على برامج التمكين والتطوير."
        vision_en = f" The situation in this area is relatively better. Focus on empowerment and development programs."
        color = "#27ae60"

    area_name = " - ".join([x for x in [sel_gov, sel_center, sel_village] if x != t("الكل","All")])

    st.markdown(f"""
    <div style='background:{color}15; border-{'right' if LANG=='ar' else 'left'}:4px solid {color}; border-radius:12px; padding:24px; margin-top:20px;'>
        <h3 style='color:{color}; margin:0 0 8px;'> {t(f'رؤية المنطقة : {area_name}',f'Vision for Area: {area_name}')}</h3>
        <p style='margin:0; color:#eeeeee;'>{vision_ar if LANG=='ar' else vision_en}</p>
    </div>
    """, unsafe_allow_html=True)


def donor_donate():
    scroll_to_top()
    st.markdown(f"""
    <div class="main-header">
        <div>
            <h1> {t('صفحة التبرع','Donation Page')}</h1>
            <p>{t('تبرعك يُغير حياة أسرة كاملة','Your donation changes an entire family\'s life')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Amount selector
    st.markdown(f"<div class='section-title'>💰 {t('اختر مبلغ التبرع','Choose Donation Amount')}</div>", unsafe_allow_html=True)
    preset_amounts = [100, 250, 500, 1000, 2500, 5000]
    cols = st.columns(6)
    selected_preset = None
    for i, amt in enumerate(preset_amounts):
        with cols[i]:
            if st.button(f"EGP {amt:,}", use_container_width=True, key=f"amt_{amt}"):
                selected_preset = amt
                st.session_state["donation_amount"] = amt

    custom_amount = st.number_input(t("أو أدخل مبلغاً مخصصاً (ج.م)","Or enter custom amount (EGP)"),
                                    min_value=10, max_value=1000000,
                                    value=st.session_state.get("donation_amount", 500), step=50)
    st.session_state["donation_amount"] = custom_amount

    # Impact calculator
    amount = st.session_state.get("donation_amount", 500)
    meals = int(amount / 50)
    families = max(1, int(amount / 1000))
    months = max(1, int(amount / 1500))

    st.markdown(f"""
    <div style='background:#151d2b; border-radius:12px; padding:20px; margin:16px 0; display:flex; gap:32px; flex-wrap:wrap;'>
        <div style='text-align:center;'>
            <div style='font-size:2rem; font-weight:900; color:#27ae60;'>{meals}</div>
            <div style='color:#eeeeee; font-size:1.2rem;'>{t('وجبة','Meals')}</div>
        </div>
        <div style='text-align:center;'>
            <div style='font-size:2rem; font-weight:900; color:#2980b9;'>{families}</div>
            <div style='color:#eeeeee; font-size:1.2rem;'>{t('أسرة مستفيدة','Families Supported')}</div>
        </div>
        <div style='text-align:center;'>
            <div style='font-size:2rem; font-weight:900; color:#8e44ad;'>{amount:,} {t('ج.م','EGP')}</div>
            <div style='color:#eeeeee; font-size:1.2rem;'>{t('قيمة تبرعك','Your Donation')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Payment methods
    st.markdown(f"<div class='section-title'> {t('طرق الدفع','Payment Methods')}</div>", unsafe_allow_html=True)

    import urllib.parse

    amount = st.session_state.get("donation_amount", 500)

    instapay_account = "mohamednbashar@instapay"
    vodafone_cash_number = "1023475060"
    whatsapp_number = "1122968626"

    c1, c2, c3 = st.columns(3)

    # ───────── InstaPay ─────────
    with c1:
        st.markdown(f"""
        <div class='pay-card'>
            <div class='pay-icon'>💳</div>
            <div class='pay-name'>InstaPay</div>
            <div class='pay-desc'>{t('الدفع الفوري عبر إنستاباي','Instant payment via InstaPay')}</div>
            <div style='margin-top:16px; font-size:0.85rem; color: rgb(216, 174, 52); font-weight:700;'>
                {instapay_account}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(t("الدفع عبر InstaPay","Pay via InstaPay"), use_container_width=True, key="instapay"):
            msg = (
                f"يرجى تحويل {amount:,} جنيه عبر InstaPay إلى: {instapay_account}\nثم إرسال إثبات الدفع"
                if LANG == "ar"
                else f"Please transfer {amount:,} EGP via InstaPay to: {instapay_account}\nThen send payment proof"
            )
            st.info(msg)


    # ───────── Vodafone Cash ─────────
    with c2:
        st.markdown(f"""
        <div class='pay-card'>
            <div class='pay-icon'>📱</div>
            <div class='pay-name'>Vodafone Cash</div>
            <div class='pay-desc'>{t('التحويل عبر فودافون كاش','Transfer via Vodafone Cash')}</div>
            <div style='margin-top:16px; font-size:0.85rem; color: rgb(216, 174, 52); font-weight:700;'>
                +20 {vodafone_cash_number}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(t("الدفع عبر فودافون كاش","Pay via Vodafone Cash"), use_container_width=True, key="vcash"):
            msg = (
                f"يرجى تحويل {amount:,} جنيه عبر Vodafone Cash إلى: +20{vodafone_cash_number}"
                if LANG == "ar"
                else f"Please transfer {amount:,} EGP via Vodafone Cash to: +20{vodafone_cash_number}"
            )
            st.info(msg)


# ───────── WhatsApp ─────────
    with c3:
        st.markdown(f"""
        <div class='pay-card'>
            <div class='pay-icon'>💬</div>
            <div class='pay-name'>WhatsApp</div>
            <div class='pay-desc'>{t('تواصل معنا مباشرة عبر واتساب','Contact us directly via WhatsApp')}</div>
            <div style='margin-top:16px; font-size:0.85rem; color: rgb(216, 174, 52); font-weight:700;'>
                +20 {whatsapp_number}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
        whatsapp_msg = (
            f"مرحبا، أريد التبرع بمبلغ {amount:,} جنيه لمساعدة الأسر المحتاجة\n\n سند · SANAD"
            if LANG == "ar"
            else f"Hello, I want to donate {amount:,} EGP to help needy families\n\n سند · SAND"
        )
    
        wa_url = f"https://wa.me/{whatsapp_number}?text={urllib.parse.quote(whatsapp_msg)}"
    
        st.link_button(
            t("تواصل عبر واتساب","Contact via WhatsApp"),
            wa_url,
            use_container_width=True
        )
    
    # Trust badges
    st.markdown(f"""
    <div style='display:flex; gap:20px; flex-wrap:wrap; margin-top:32px; justify-content:center;'>
        <div style='text-align:center; color:#7f8c8d;'> {t('آمن 100%','100% Secure')}</div>
        <div style='text-align:center; color:#7f8c8d;'> - </div>
        <div style='text-align:center; color:#7f8c8d;'> {t('كل جنيه يصل لمستحقه','Every penny reaches the needy')}</div>
        <div style='text-align:center; color:#7f8c8d;'> - </div>
        <div style='text-align:center; color:#7f8c8d;'> {t('شفافية كاملة','Full Transparency')}</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ─── MAIN APP ─────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def scroll_to_top():
    st.markdown(
        """<script>
        window.parent.document.querySelector('[data-testid="stAppViewContainer"]')
        ?.scrollTo({top: 0, behavior: 'instant'});
        window.parent.scrollTo(0, 0);
        </script>""",
        unsafe_allow_html=True
    )

def main():
    inject_css()
    df = load_data()
    models = load_models()

    if not st.session_state.logged_in:
        if st.session_state.get("show_public_form", False):
            public_case_submission()
        else:
            login_page()
        return

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align:center; padding: 20px 0 24px;'>
            <div style='font-size:3rem;'>🎗️</div>
            <div style='font-size:1.5rem; font-weight:900; color:white; margin-top:4px;'>سند · SANAD</div>
            <div style='font-size:0.78rem; color:#95a5a6; margin-top:4px;'>
                {'نظام إدارة الأسر المحتاجة' if LANG=='ar' else 'Needy Families Management'}
            </div>
        </div>
        <hr style='border-color:#2c3e50; margin:0 0 16px;'>
        """, unsafe_allow_html=True)

        # User badge
        icon = "👤" if st.session_state.user_type == "admin" else "🤝"
        role_label = t("أدمن","Admin") if st.session_state.user_type == "admin" else t("متبرع","Donor")
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.08); border-radius:12px; padding:12px 16px; margin-bottom:16px; display:flex; align-items:center; gap:10px;'>
            <div style='font-size:1.5rem;'>{icon}</div>
            <div>
                <div style='font-weight:700; color:white; font-size:0.9rem;'>{st.session_state.username}</div>
                <div style='font-size:0.75rem; color:#95a5a6;'>{role_label}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.user_type == "admin":
            pages = {
                t(" الرئيسية "," Home"): "home",
                t(" إدارة الحالات "," Case Management"): "cases",
                t(" إضافة حالة "," Add Case"): "add_case",
            }
        else:
            pages = {
                t(" الإحصائيات"," Statistics"): "donor_stats",
                t(" خريطة الاحتياج"," Need Map"): "donor_map",
                t(" التبرع"," Donate"): "donor_donate",
            }

        selected_label = st.radio(t("القائمة","Menu"), list(pages.keys()), label_visibility="collapsed")
        selected_page = pages[selected_label]

        st.markdown("<hr style='border-color:#2c3e50; margin:20px 0 16px;'>", unsafe_allow_html=True)

        # Wrapper واحد لكل الأزرار
        st.markdown("<div style='margin-top:8px;'>", unsafe_allow_html=True)

# زر إنشاء حساب أدمن
        if st.session_state.user_type == "admin":
            if st.button(
                f" {t('إنشاء حساب أدمن','Create Admin Account')}",
                use_container_width=True,
                key="sidebar_goto_register",
                type="primary" if selected_page == "admin_register" else "secondary"
            ):
                st.session_state["selected_page_override"] = "admin_register"
                st.rerun()

# Language toggle
        if st.button(" " + ("English" if LANG == "ar" else "عربي"), use_container_width=True):
            st.session_state.lang = "en" if LANG == "ar" else "ar"
            st.rerun()

# Logout
        if st.button(t(" تسجيل الخروج"," Logout"), use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_type = None
            st.session_state.username = ""
            if "selected_page_override" in st.session_state:
                del st.session_state["selected_page_override"]
            st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)
            # DB stats mini
            if not df.empty:
                st.markdown(f"""
                <div style='margin-top:auto; background:rgba(255,255,255,0.05); border-radius:10px; padding:12px; margin-top:16px; font-size:0.78rem; color:#95a5a6;'>
                    ● {len(df):,} {t('حالة مسجلة','registered cases')}<br>
                    ● {df['governorate'].nunique() if 'governorate' in df.columns else 0} {t('محافظة','governorates')}
                </div>
                """, unsafe_allow_html=True)

    # ── Page Router ──
    # التحقق من الـ override للانتقال لصفحة إنشاء الحساب
    override = st.session_state.get("selected_page_override", None)
    if override == "admin_register":
        admin_register_page()
        # نضيف زر رجوع
        if st.button(t("← رجوع للرئيسية","← Back to Home"), key="back_from_register"):
            del st.session_state["selected_page_override"]
            st.rerun()
        return

    if selected_page == "home":
        admin_home(df, models)
    elif selected_page == "cases":
        admin_cases(df)
    elif selected_page == "add_case":
        admin_add_case(df, models)
    elif selected_page == "donor_stats":
        donor_stats(df)
    elif selected_page == "donor_map":
        donor_map(df)
    elif selected_page == "donor_donate":
        donor_donate()


if __name__ == "__main__":
    main()