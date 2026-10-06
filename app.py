import re
import string

import joblib
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from nltk.corpus import stopwords

# ----------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Emotion Predictor",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# One-time setup: nltk stopwords, model + vectorizer loading (all cached)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_stopwords():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load("logistic_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


STOP_WORDS = load_stopwords()
model_lr, vector_t = load_artifacts()

# Label mapping exactly as produced in the training notebook:
# unique_emotions = df['emotion'].unique()  ->
# ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']
EMOTION_MAP = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy",
}

EMOTION_STYLE = {
    "sadness": {"emoji": "😢", "color": "#5B8DEF"},
    "anger": {"emoji": "😠", "color": "#E4572E"},
    "love": {"emoji": "❤️", "color": "#E4589A"},
    "surprise": {"emoji": "😮", "color": "#F5B841"},
    "fear": {"emoji": "😨", "color": "#7B61FF"},
    "joy": {"emoji": "😄", "color": "#3DBE64"},
}


# ----------------------------------------------------------------------------
# EXACT preprocessing pipeline from the training notebook (logic unchanged)
# ----------------------------------------------------------------------------
def remove_pun(txt: str) -> str:
    return txt.translate(str.maketrans("", "", string.punctuation))


def remove_num(txt: str) -> str:
    new = ""
    for i in txt:
        if not i.isdigit():
            new = new + i
    return new


def remove_url(txt: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", txt)


def remove_emoji(txt: str) -> str:
    new = ""
    for i in txt:
        if i.isascii():
            new = new + i
    return new


def remove_stopwords(txt: str) -> str:
    words = txt.split()
    cleaned = [i for i in words if i not in STOP_WORDS]
    return " ".join(cleaned)


def preprocess(text: str) -> str:
    text = text.lower()
    text = remove_pun(text)
    text = remove_num(text)
    text = remove_url(text)
    text = remove_emoji(text)
    text = remove_stopwords(text)
    return text


def predict_emotion(raw_text: str):
    cleaned = preprocess(raw_text)
    vec = vector_t.transform([cleaned])
    pred_class = model_lr.predict(vec)[0]
    probs = model_lr.predict_proba(vec)[0]
    return cleaned, EMOTION_MAP[pred_class], probs


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .hero {
            text-align: center;
            padding: 1.6rem 1rem 1.2rem 1rem;
        }
        .hero h1 {
            font-size: 2.3rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            background: linear-gradient(90deg, #7B61FF, #E4589A, #F5B841);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero p {
            color: #9CA3AF;
            font-size: 1.02rem;
            margin-top: 0;
        }

        .stTextArea textarea {
            border-radius: 14px;
            border: 1.5px solid #2D2F3A;
            font-size: 1.05rem;
            padding: 0.9rem;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 12px;
            padding: 0.7rem 0;
            font-weight: 700;
            font-size: 1.05rem;
            border: none;
            background: linear-gradient(90deg, #7B61FF, #E4589A);
            color: white;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(123, 97, 255, 0.35);
            color: white;
        }

        .result-card {
            border-radius: 20px;
            padding: 2rem 1.5rem;
            text-align: center;
            margin-top: 1.3rem;
            margin-bottom: 1.3rem;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .result-emoji {
            font-size: 4rem;
            line-height: 1;
        }
        .result-label {
            font-size: 1.9rem;
            font-weight: 800;
            text-transform: capitalize;
            margin-top: 0.4rem;
        }
        .result-conf {
            font-size: 1rem;
            color: #C9CCD6;
            margin-top: 0.2rem;
        }

        .cleaned-box {
            background: rgba(123, 97, 255, 0.08);
            border: 1px dashed rgba(123, 97, 255, 0.4);
            border-radius: 12px;
            padding: 0.7rem 1rem;
            font-size: 0.9rem;
            color: #B9B9C6;
            font-family: 'Courier New', monospace;
        }

        .history-item {
            padding: 0.5rem 0.8rem;
            border-radius: 10px;
            background: rgba(255,255,255,0.04);
            margin-bottom: 0.4rem;
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎭 About this app")
    st.write(
        "This app predicts the **emotion** behind a piece of text using a "
        "**Logistic Regression** model trained on **TF-IDF** features."
    )
    st.markdown("---")
    st.markdown("**Model:** Logistic Regression")
    st.markdown("**Features:** TF-IDF")
    st.markdown("**Classes:**")
    cols = st.columns(2)
    for idx, (label, style) in enumerate(EMOTION_STYLE.items()):
        with cols[idx % 2]:
            st.markdown(f"{style['emoji']} {label.capitalize()}")
    st.markdown("---")
    show_probs = st.checkbox("Show probability breakdown", value=True)
    show_cleaned = st.checkbox("Show cleaned text sent to model", value=False)
    st.markdown("---")
    if st.button("🗑️ Clear history"):
        st.session_state["history"] = []
        st.rerun()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Emotion Predictor</h1>
        <p>Type a sentence and let the model read the feeling behind it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state["history"] = []

# ----------------------------------------------------------------------------
# Input area
# ----------------------------------------------------------------------------
examples = [
    "i feel like this is such a wonderful surprise",
    "i am so angry that nobody listens to me",
    "i just feel so hopeless about everything right now",
    "i cannot stop smiling today everything feels perfect",
]

example_choice = st.selectbox(
    "Try an example, or write your own below 👇",
    ["-- pick an example --"] + examples,
)

default_text = "" if example_choice == "-- pick an example --" else example_choice

user_text = st.text_area(
    "Your text",
    value=default_text,
    height=130,
    placeholder="e.g. I can't believe how amazing today turned out to be!",
    label_visibility="collapsed",
)

predict_clicked = st.button("✨ Predict Emotion")

# ----------------------------------------------------------------------------
# Prediction + display
# ----------------------------------------------------------------------------
if predict_clicked:
    if not user_text or not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing text..."):
            cleaned_text, label, probs = predict_emotion(user_text)

        style = EMOTION_STYLE[label]
        confidence = float(np.max(probs)) * 100

        st.markdown(
            f"""
            <div class="result-card" style="background: {style['color']}1A;
                        border-color: {style['color']}55;">
                <div class="result-emoji">{style['emoji']}</div>
                <div class="result-label" style="color: {style['color']};">{label}</div>
                <div class="result-conf">Confidence: {confidence:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if show_cleaned:
            st.markdown("**Text after preprocessing (what the model actually sees):**")
            st.markdown(
                f'<div class="cleaned-box">{cleaned_text if cleaned_text else "<em>(empty after cleaning)</em>"}</div>',
                unsafe_allow_html=True,
            )

        if show_probs:
            st.markdown("#### Probability breakdown")
            labels = [EMOTION_MAP[i] for i in range(len(probs))]
            colors = [EMOTION_STYLE[l]["color"] for l in labels]
            order = np.argsort(probs)[::-1]

            fig = go.Figure(
                go.Bar(
                    x=[probs[i] * 100 for i in order],
                    y=[f"{EMOTION_STYLE[labels[i]]['emoji']} {labels[i].capitalize()}" for i in order],
                    orientation="h",
                    marker_color=[colors[i] for i in order],
                    text=[f"{probs[i]*100:.1f}%" for i in order],
                    textposition="outside",
                )
            )
            fig.update_layout(
                xaxis_title="Probability (%)",
                yaxis=dict(autorange="reversed"),
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(range=[0, 100]),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.session_state["history"].insert(
            0, {"text": user_text.strip(), "label": label, "emoji": style["emoji"], "confidence": confidence}
        )
        st.session_state["history"] = st.session_state["history"][:10]

# ----------------------------------------------------------------------------
# History
# ----------------------------------------------------------------------------
if st.session_state["history"]:
    st.markdown("#### 🕘 Recent predictions")
    for item in st.session_state["history"]:
        st.markdown(
            f"""
            <div class="history-item">
                {item['emoji']} <b>{item['label'].capitalize()}</b>
                ({item['confidence']:.1f}%) &nbsp;—&nbsp;
                <span style="color:#9CA3AF;">"{item['text'][:80]}{'...' if len(item['text']) > 80 else ''}"</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
