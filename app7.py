from pathlib import Path
import html

import pandas as pd
import streamlit as st

# ==========================================================
# CareDesk | Medical Records Assistant
# Streamlit reference UI for a student/hackathon prototype.
# Use synthetic/de-identified data for demos.
# ==========================================================

st.set_page_config(
    page_title="CareDesk | Medical Records",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "patient_data.csv"


# ----------------------------- DATA -----------------------------

@st.cache_data(show_spinner="Loading records…")
def load_csv(path: str, modified: float) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.columns = [str(c).strip() for c in data.columns]
    return data


def normalize(value) -> str:
    return " ".join(
        str(value).strip().lower().replace("_", " ").replace("-", " ").split()
    )


def find_id_column(columns):
    accepted = {"patient id", "patientid", "patient identifier", "patient number"}
    for column in columns:
        if normalize(column) in accepted:
            return column
    return None


if not DATA_FILE.exists():
    st.error(
        "Dataset not found. Place **patient_data.csv** in the same folder as **app.py**."
    )
    st.code(f"Expected location: {DATA_FILE}")
    st.stop()

try:
    df = load_csv(str(DATA_FILE), DATA_FILE.stat().st_mtime)
except Exception as exc:
    st.error("The dataset could not be read. Check the CSV format and encoding.")
    st.caption(f"Technical detail: {exc}")
    st.stop()

if df.empty:
    st.warning("The CSV file is empty. Add records and refresh the application.")
    st.stop()

ID_COL = find_id_column(df.columns)
if ID_COL is None:
    st.error(
        "Patient ID column not found. Rename the column to `patient_id` or `Patient ID`."
    )
    st.stop()


def patient_lookup(patient_id: str) -> pd.DataFrame:
    query = str(patient_id).strip()
    if not query:
        return df.iloc[0:0].copy()

    values = df[ID_COL].astype("string").str.strip()
    result = df[values.str.casefold() == query.casefold()]

    if result.empty:
        numeric_values = pd.to_numeric(df[ID_COL], errors="coerce")
        numeric_query = pd.to_numeric(query, errors="coerce")
        if pd.notna(numeric_query):
            result = df[numeric_values == numeric_query]

    return result.copy()


def nice_label(column) -> str:
    return str(column).replace("_", " ").replace("-", " ").strip().title()


def safe_value(value) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "Not available"
    return str(value).strip()


def answer_from_record(patient_id: str, question: str) -> str:
    """Transparent field-matching assistant, not a clinical diagnostic model."""
    record = patient_lookup(patient_id)
    if record.empty:
        return "I couldn't find that patient ID. Please check it and try again."

    q = normalize(question)

    full_record_phrases = [
        "full record", "complete record", "all details",
        "show everything", "summarize", "summary", "all information",
    ]
    if any(phrase in q for phrase in full_record_phrases):
        return "\n\n".join(
            f"**{nice_label(col)}:** {safe_value(value)}"
            for _, row in record.iterrows()
            for col, value in row.items()
        )

    aliases = {
        "doctor": ["doctor", "physician", "consultant", "provider"],
        "medicine": ["medicine", "medication", "drug", "prescription", "tablet"],
        "diagnosis": ["diagnosis", "disease", "condition", "illness"],
        "department": ["department", "specialty", "speciality", "unit"],
        "appointment": ["appointment", "visit", "follow up"],
        "lab": ["lab", "laboratory", "test result", "test results"],
        "age": ["age", "how old"],
        "gender": ["gender", "sex"],
        "name": ["patient name", "full name", "name of patient"],
        "contact": ["phone", "contact", "email", "address"],
        "date": ["date", "admission", "visit date"],
    }

    requested = [
        category for category, terms in aliases.items()
        if any(term in q for term in terms)
    ]

    matched_columns = []
    for col in record.columns:
        col_normalized = normalize(col)
        if len(col_normalized) >= 3 and col_normalized in q:
            matched_columns.append(col)

    field_terms = {
        "doctor": ["doctor", "physician", "consultant", "provider"],
        "medicine": ["medicine", "medication", "drug", "prescription"],
        "diagnosis": ["diagnosis", "disease", "condition"],
        "department": ["department", "specialty", "speciality"],
        "appointment": ["appointment", "visit"],
        "lab": ["lab", "test", "result"],
        "age": ["age"],
        "gender": ["gender", "sex"],
        "name": ["patient name", "full name", "name"],
        "contact": ["phone", "contact", "email", "address"],
        "date": ["date", "admission"],
    }

    for col in record.columns:
        col_normalized = normalize(col)
        for category in requested:
            if any(term in col_normalized for term in field_terms.get(category, [])):
                matched_columns.append(col)
                break

    matched_columns = list(dict.fromkeys(matched_columns))
    if not matched_columns:
        return (
            "I couldn't match that question to a field in this dataset. Try asking "
            "about the doctor, medicine, diagnosis, department, age, appointment, "
            "lab results, or say **“show the complete record.”**"
        )

    responses = []
    for col in matched_columns:
        vals = list(dict.fromkeys(safe_value(v) for v in record[col].tolist()))
        responses.append(f"**{nice_label(col)}:** {', '.join(vals[:10])}")
    return "\n\n".join(responses)


# ----------------------------- DESIGN SYSTEM -----------------------------

st.markdown(
    """
    <style>
    :root {
        --page: #f5f7fb;
        --surface: #ffffff;
        --ink: #17243b;
        --muted: #68788f;
        --border: #e2e9f2;
        --blue: #245fe8;
        --blue-dark: #1748bd;
        --teal: #0e9f9a;
    }

    .stApp { background: var(--page); color: var(--ink); }
    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, h4 { color: var(--ink) !important; letter-spacing: -.025em; }
    p, li { color: #3c4b61; }
    [data-testid="stCaptionContainer"] { color: var(--muted); }

    [data-testid="stSidebar"] {
        background: #101c30;
        border-right: 1px solid #233650;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #f1f5fb !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #aabbd1 !important;
    }

    .brand-row { display:flex; align-items:center; gap:12px; padding:8px 0 18px; }
    .brand-icon {
        display:flex; align-items:center; justify-content:center;
        width:44px; height:44px; border-radius:14px;
        color:white; font-size:22px;
        background:linear-gradient(135deg,#398bff,#11aaa5);
        box-shadow:0 8px 20px rgba(25,112,220,.22);
    }
    .brand-title { color:#fff; font-size:20px; font-weight:800; line-height:1.15; }
    .brand-caption { color:#aabbd1; font-size:10px; letter-spacing:.12em; margin-top:5px; }

    .hero {
        background:linear-gradient(115deg,#122d53 0%,#1856a3 62%,#0c938f 125%);
        color:#fff; border-radius:22px; padding:30px 32px; margin-bottom:22px;
        box-shadow:0 14px 35px rgba(24,75,140,.14);
    }
    .hero-kicker { color:#b9e8ff; font-size:11px; font-weight:800; letter-spacing:.15em; }
    .hero h1 { color:#fff !important; font-size:clamp(27px,3vw,37px); margin:10px 0 8px; }
    .hero p { color:#e4efff !important; max-width:800px; font-size:15px; line-height:1.65; margin:0; }
    .eyebrow { color:var(--blue); font-size:11px; font-weight:800; letter-spacing:.13em; text-transform:uppercase; margin-bottom:6px; }
    .subtitle { color:var(--muted); margin-top:-7px; margin-bottom:22px; font-size:14px; }

    .metric-card {
        background:#fff; border:1px solid var(--border); border-radius:16px;
        padding:19px 20px; min-height:115px;
        box-shadow:0 4px 16px rgba(20,42,75,.025);
    }
    .metric-name { color:#68788f; font-size:11px; text-transform:uppercase; letter-spacing:.08em; font-weight:800; }
    .metric-number { color:#17243b; font-size:28px; font-weight:850; margin-top:8px; line-height:1.2; }
    .metric-help { color:#7b899d; font-size:12px; margin-top:5px; }

    .field-row {
        display:flex; gap:14px; align-items:flex-start;
        padding:12px 2px; border-bottom:1px solid #edf1f6;
        overflow-wrap:anywhere; line-height:1.5;
    }
    .field-row:last-child { border-bottom:0; }
    .field-key { color:#68788f; font-size:13px; font-weight:650; width:40%; min-width:110px; }
    .field-value { color:#17243b; font-size:13px; font-weight:650; flex:1; }

    .stApp input, .stApp textarea,
    [data-baseweb="input"] input, [data-baseweb="textarea"] textarea {
        background:#fff !important; color:#17243b !important;
        -webkit-text-fill-color:#17243b !important; border-radius:10px !important;
    }
    [data-testid="stWidgetLabel"] p { color:#33445c !important; font-weight:650 !important; }
    .stButton > button, .stDownloadButton > button {
        min-height:42px; border-radius:10px; font-weight:700; transition:.15s ease;
    }
    .stButton > button[kind="primary"] {
        background:var(--blue); color:white; border:1px solid var(--blue);
    }
    .stButton > button[kind="primary"]:hover {
        background:var(--blue-dark); border-color:var(--blue-dark);
    }
    .stButton > button[kind="secondary"], .stDownloadButton > button {
        background:#fff; color:#263750; border:1px solid #d5dfeb;
    }
    [data-testid="stChatMessage"] {
        background:#fff; border:1px solid var(--border);
        border-radius:14px; padding:12px; margin-bottom:10px;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li { color:#263750 !important; line-height:1.65; }
    [data-testid="stChatInput"] { background:#fff; border:1px solid #ccd8e8; border-radius:13px; }
    [data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:12px; overflow:hidden; }
    [data-testid="stAlert"] { border-radius:12px; }
    hr { border-color:var(--border) !important; }

    @media(max-width:768px) {
        .block-container { padding:1rem; }
        .hero { padding:23px 20px; border-radius:16px; }
        .field-key { min-width:90px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- HELPERS -----------------------------

def header(kicker: str, title: str, subtitle: str):
    st.markdown(f'<div class="eyebrow">{html.escape(kicker)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="subtitle">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def metric(label: str, value: str, help_text: str):
    st.markdown(
        f'<div class="metric-card"><div class="metric-name">{html.escape(label)}</div>'
        f'<div class="metric-number">{html.escape(value)}</div>'
        f'<div class="metric-help">{html.escape(help_text)}</div></div>',
        unsafe_allow_html=True,
    )


def patient_card(record: pd.DataFrame):
    if record.empty:
        return
    row = record.iloc[0]
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    for col, value in row.items():
        st.markdown(
            f'<div class="field-row"><div class="field-key">{html.escape(nice_label(col))}</div>'
            f'<div class="field-value">{html.escape(safe_value(value))}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------- SESSION -----------------------------

st.session_state.setdefault("selected_patient_id", "")
st.session_state.setdefault("messages", [])


# ----------------------------- SIDEBAR -----------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-icon">✚</div>
            <div>
                <div class="brand-title">CareDesk</div>
                <div class="brand-caption">AI MEDICAL RECORDS ASSISTANT</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Medical Assistant", "Patient Search", "Records Explorer", "About"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**DATA SOURCE**")
    st.caption("Local CSV dataset")
    st.caption(f"{len(df):,} rows · {len(df.columns)} fields")
    st.warning("Demo only. Use synthetic/de-identified data.")


# ----------------------------- DASHBOARD -----------------------------

if page == "Dashboard":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">CARE OPERATIONS · OVERVIEW</div>
            <h1>AI MEDICAL RECORDS ASSISTANT</h1>
            <p>One workspace to search patient records, review available information,
            and retrieve fields with a simple record assistant.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    unique_patients = int(df[ID_COL].nunique(dropna=True))
    missing = int(df.isna().sum().sum())
    completeness = 100 * (1 - missing / max(df.size, 1))

    cols = st.columns(4, gap="medium")
    with cols[0]:
        metric("Total records", f"{len(df):,}", "Rows in the dataset")
    with cols[1]:
        metric("Unique patients", f"{unique_patients:,}", "Distinct patient IDs")
    with cols[2]:
        metric("Data fields", f"{len(df.columns):,}", "Available CSV columns")
    with cols[3]:
        metric("Completeness", f"{completeness:.1f}%", "Cells with values")

    st.write("")
    header("WORKSPACE", "Quick access", "Choose a task to get started.")

    q1, q2, q3 = st.columns(3, gap="medium")
    with q1:
        with st.container(border=True):
            st.subheader("🔎 Patient search")
            st.write("Find a record using its Patient ID.")
            st.caption("Best for looking up one patient.")
    with q2:
        with st.container(border=True):
            st.subheader("💬 Medical assistant")
            st.write("Ask questions about fields in a selected record.")
            st.caption("Rule-based lookup, not clinical advice.")
    with q3:
        with st.container(border=True):
            st.subheader("📋 Records explorer")
            st.write("Filter the CSV and export the matching rows.")
            st.caption("Useful for data review and demos.")

    st.write("")
    header("DATA PREVIEW", "Recently listed records", "Preview only; verify access before sharing records.")
    st.dataframe(df.head(8), use_container_width=True, hide_index=True)


# ----------------------------- MEDICAL ASSISTANT -----------------------------

elif page == "Medical Assistant":
    header(
        "PATIENT WORKSPACE",
        "Medical Assistant",
        "Load a patient record, review the available fields, and ask a question.",
    )

    left, right = st.columns([0.85, 1.15], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("Patient lookup")
            st.caption("Enter the Patient ID exactly as it appears in the CSV.")
            patient_input = st.text_input(
                "Patient ID",
                value=st.session_state["selected_patient_id"],
                placeholder="e.g. P0001",
                key="assistant_patient_input",
            )

            if st.button("Load patient record", type="primary", use_container_width=True):
                result = patient_lookup(patient_input)
                if result.empty:
                    st.error("Patient not found. Check the ID and try again.")
                else:
                    st.session_state["selected_patient_id"] = patient_input.strip()
                    st.session_state["messages"] = []
                    st.success("Patient record loaded.")
                    st.rerun()

            active_id = st.session_state["selected_patient_id"]
            active_record = patient_lookup(active_id) if active_id else df.iloc[0:0].copy()

            if not active_record.empty:
                st.success("A patient is selected.")
                st.markdown(f"**Patient ID:** `{html.escape(active_id)}`")
                if st.button("Clear selection", use_container_width=True):
                    st.session_state["selected_patient_id"] = ""
                    st.session_state["messages"] = []
                    st.rerun()
            else:
                st.info("Load a patient to enable the assistant.")

    with right:
        st.subheader("Patient profile")
        active_id = st.session_state["selected_patient_id"]
        active_record = patient_lookup(active_id) if active_id else df.iloc[0:0].copy()
        if not active_record.empty:
            patient_card(active_record)
        else:
            with st.container(border=True):
                st.write("Patient details will appear here.")
                st.caption("The profile is populated after a valid ID is loaded.")

    st.divider()
    chat_col, prompt_col = st.columns([1.45, 0.75], gap="large")

    with chat_col:
        st.subheader("Record chat")
        st.caption("Answers are generated from matching CSV fields; this is not a diagnostic tool.")

        with st.container(border=True):
            for message in st.session_state["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            question = st.chat_input(
                "Ask about diagnosis, medicine, doctor, lab results…",
                disabled=not bool(st.session_state["selected_patient_id"]),
            )
            if question:
                answer = answer_from_record(st.session_state["selected_patient_id"], question)
                st.session_state["messages"].extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ])
                st.rerun()

            if not st.session_state["selected_patient_id"]:
                st.caption("Select a patient above before asking a question.")

    with prompt_col:
        with st.container(border=True):
            st.subheader("Suggested questions")
            suggestions = [
                "Show the complete record",
                "Who is the doctor?",
                "What is the diagnosis?",
                "List the medicines",
                "Show lab test results",
            ]
            for idx, prompt in enumerate(suggestions):
                if st.button(prompt, key=f"prompt_{idx}", use_container_width=True):
                    patient_id = st.session_state["selected_patient_id"]
                    if patient_id:
                        st.session_state["messages"].extend([
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": answer_from_record(patient_id, prompt)},
                        ])
                        st.rerun()
                    else:
                        st.warning("Load a patient first.")
            if st.button("Clear conversation", use_container_width=True):
                st.session_state["messages"] = []
                st.rerun()


# ----------------------------- PATIENT SEARCH -----------------------------

elif page == "Patient Search":
    header(
        "PATIENT DIRECTORY",
        "Patient Search",
        "Search for a patient by their unique identifier.",
    )

    with st.container(border=True):
        search_col, button_col = st.columns([3, 1], vertical_alignment="bottom")
        with search_col:
            search_id = st.text_input(
                "Patient ID",
                placeholder="Enter ID, for example P0001",
                key="directory_patient_id",
            )
        with button_col:
            search_button = st.button("Search patient", type="primary", use_container_width=True)

    if search_button:
        result = patient_lookup(search_id)
        if result.empty:
            st.warning("No matching patient was found.")
        else:
            st.success(f"Found {len(result)} record(s).")
            st.subheader("Patient details")
            patient_card(result)
            if st.button("Select for Medical Assistant", type="primary"):
                st.session_state["selected_patient_id"] = search_id.strip()
                st.session_state["messages"] = []
                st.info("Patient selected. Open Medical Assistant from the sidebar.")

    st.write("")
    st.subheader("Directory preview")
    preview_cols = [ID_COL] + [
        col for col in df.columns
        if col != ID_COL and any(term in normalize(col) for term in ("name", "age", "department"))
    ]
    preview_cols = list(dict.fromkeys(preview_cols))
    st.dataframe(df[preview_cols].head(100), use_container_width=True, hide_index=True)
    st.caption("Showing up to 100 rows. Use Patient ID search for an exact lookup.")


# ----------------------------- RECORDS EXPLORER -----------------------------

elif page == "Records Explorer":
    header(
        "DATA MANAGEMENT",
        "Records Explorer",
        "Search the CSV, choose visible columns, and export filtered results.",
    )

    with st.container(border=True):
        search_col, columns_col = st.columns([1, 1], gap="large")
        with search_col:
            query = st.text_input("Search all fields", placeholder="Search ID or any record value…")
        with columns_col:
            visible_columns = st.multiselect(
                "Visible columns",
                options=list(df.columns),
                default=list(df.columns),
            )

    view = df.copy()
    if query.strip():
        mask = view.astype("string").apply(
            lambda series: series.str.contains(query.strip(), case=False, na=False, regex=False)
        ).any(axis=1)
        view = view[mask]

    if visible_columns:
        view = view[visible_columns]
    else:
        st.info("Choose at least one column to display.")
        view = view.iloc[:, 0:0]

    st.markdown(f"**{len(view):,} matching row(s)**")
    st.dataframe(view, use_container_width=True, hide_index=True, height=480)

    st.download_button(
        "Download filtered CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="medical_records_filtered.csv",
        mime="text/csv",
    )


# ----------------------------- ABOUT -----------------------------

elif page == "About":
    header(
        "PROJECT DETAILS",
        "About CareDesk",
        "A student project prototype for structured medical-record lookup.",
    )

    about_col, safety_col = st.columns([1.2, 0.8], gap="large")

    with about_col:
        with st.container(border=True):
            st.subheader("What it does")
            st.write(
                "CareDesk loads a local CSV dataset, looks up patients by ID, "
                "displays available fields, and uses transparent keyword matching "
                "to retrieve relevant values."
            )
            st.subheader("Technology stack")
            st.write("Python · Streamlit · Pandas · CSV")
            st.subheader("Current limitations")
            st.write(
                "- No authentication or role-based access control.\n"
                "- No database-backed audit trail.\n"
                "- The assistant is not an LLM and does not diagnose or prescribe.\n"
                "- Local CSV storage is for demonstration, not a production clinical workflow."
            )

    with safety_col:
        with st.container(border=True):
            st.subheader("Privacy and safety")
            st.warning(
                "Use synthetic or de-identified records for demos. Do not deploy real "
                "patient information publicly without appropriate security, authorization, "
                "consent, and legal/privacy review."
            )
            st.subheader("Dataset status")
            st.write(f"Rows: **{len(df):,}**")
            st.write(f"Columns: **{len(df.columns):,}**")
            st.write(f"Patient ID field: **{nice_label(ID_COL)}**")
