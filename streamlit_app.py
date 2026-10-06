"""Public Streamlit interface for the trained student mental health score model."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parent / "Mental_Health_Model.pkl"
COUNTRIES = ["India", "USA", "Canada", "Australia", "UK", "Germany", "Mexico", "Turkey", "France", "Other"]
PLATFORMS = [
    "Facebook", "Instagram", "Snapchat", "Twitter", "YouTube", "TikTok",
    "LinkedIn", "LINE", "KakaoTalk", "VKontakte", "WhatsApp", "WeChat",
]


@st.cache_resource(show_spinner="Loading prediction model…")
def load_model():
    return joblib.load(MODEL_PATH)


def make_input(age, gender, country, academic_level, platform, purpose,
               screen_time, unlocks, study, activity, sleep, stress):
    return pd.DataFrame([{
        "Study_Hours": study,
        "Age": age,
        "Avg_Daily_Usage_Hours": screen_time,
        "Daily_Unlocks": unlocks,
        "Physical_Activity_Hours": activity,
        "Sleep_Hours_Per_Night": sleep,
        "Stress_Level": stress,
        "Gender": gender,
        "Academic_Level": academic_level,
        "Most_Used_Platform": platform,
        "Purpose_Of_Use": purpose,
        "Grouped_country": country if country in COUNTRIES[:-1] else "Other",
    }])


st.set_page_config(page_title="Mental Health Signal", page_icon="🌿", layout="wide")
st.markdown("""
<style>
  .stApp {background: radial-gradient(900px 450px at 85% 0%, #f8fbf6 0%, transparent 70%), #eef2ee; color: #182420;}
  .block-container {max-width: 1160px; padding-top: 2.4rem; padding-bottom: 3rem;}
  .hero-tag {display:inline-block; padding: .4rem .8rem; border-radius: 999px; background:#e2eee6; color:#21594a; font-size:.74rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;}
  .hero-title {font-family: Georgia,serif; font-size:clamp(2.5rem,5vw,4rem); line-height:1.06; letter-spacing:-.035em; color:#163d33; margin:.8rem 0 .5rem;}
  .hero-title em {color:#397760; font-weight:400;}
  .hero-copy {max-width:650px; color:#4a5850; font-size:1.05rem; margin-bottom:1.8rem;}
  .result-card {background:linear-gradient(155deg,#163d33,#21594a); color:#eef8f0; border-radius:20px; padding:2rem; min-height:250px; box-shadow:0 12px 28px rgba(22,61,51,.15);}
  .result-kicker {font-size:.75rem; letter-spacing:.12em; text-transform:uppercase; opacity:.8; font-weight:700;}
  .result-number {font-size:4rem; font-weight:750; line-height:1.15; margin:.65rem 0;}
  .result-number span {font-size:1.4rem; opacity:.65; font-weight:400;}
  .result-note {font-size:.91rem; opacity:.84; line-height:1.5;}
  .detail-note {font-size:.85rem; color:#52645a; line-height:1.5;}
  div[data-testid="stForm"] {background:#fff; border:1px solid #d6ded5; border-radius:20px; padding:1.3rem 1.4rem; box-shadow:0 10px 26px rgba(24,36,32,.05);}
  div[data-testid="stMetric"] {background:#fff; border:1px solid #d6ded5; border-radius:12px; padding:.7rem 1rem;}
  .stButton button, div[data-testid="stFormSubmitButton"] button {background:#21594a; color:white; border:none; border-radius:9px; font-weight:700;}
  .stButton button:hover, div[data-testid="stFormSubmitButton"] button:hover {background:#163d33; color:white; border:none;}
</style>
<div class="hero-tag">Student wellness analytics</div>
<div class="hero-title">Mental Health <em>Signal</em></div>
<div class="hero-copy">Explore a model estimate based on your daily habits, social media use, and self-reported stress. This is a student project, not a clinical assessment.</div>
""", unsafe_allow_html=True)

form_col, result_col = st.columns([1.55, 1], gap="large")

with form_col:
    st.subheader("Your daily rhythm")
    with st.form("student_inputs"):
        st.markdown("**01 · Profile**")
        a, b, c = st.columns(3)
        age = a.number_input("Age", min_value=10, max_value=100, value=21, step=1)
        gender = b.selectbox("Gender", ["Female", "Male"])
        country = c.selectbox("Country", COUNTRIES, index=2, help="Countries outside this list are grouped as Other in the model.")

        st.markdown("**02 · Academic and digital habits**")
        a, b = st.columns(2)
        academic_level = a.selectbox("Academic level", ["High School", "Undergraduate", "Graduate"], index=1)
        platform = b.selectbox("Most-used platform", PLATFORMS, index=1)
        purpose = a.selectbox("Main purpose of use", ["Networking", "Education", "Entertainment", "News"], index=2)
        screen_time = b.number_input("Daily social media use (hours)", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
        unlocks = st.number_input("Daily phone unlocks", min_value=0, value=60, step=1)

        st.markdown("**03 · Lifestyle and stress**")
        a, b, c = st.columns(3)
        study = a.number_input("Study hours / day", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
        activity = b.number_input("Activity hours / day", min_value=0.0, max_value=24.0, value=1.0, step=0.5)
        sleep = c.number_input("Sleep hours / night", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        stress = st.select_slider("Perceived stress level", options=["Low", "Medium", "High", "Very High"], value="Medium")
        submitted = st.form_submit_button("Read my signal", use_container_width=True)

with result_col:
    st.subheader("Your signal")
    if submitted:
        try:
            model = load_model()
            row = make_input(age, gender, country, academic_level, platform, purpose,
                             screen_time, unlocks, study, activity, sleep, stress)
            st.session_state["mental_health_score"] = float(model.predict(row)[0])
        except Exception:
            st.error("The model could not make a prediction. Please try again later.")

    if "mental_health_score" in st.session_state:
        score = st.session_state["mental_health_score"]
        st.markdown(f"""
        <div class="result-card">
          <div class="result-kicker">Predicted score</div>
          <div class="result-number">{score:.2f}<span> / 10</span></div>
          <div class="result-note">A model estimate from the information entered. It is not a measure of your mental health or a diagnosis.</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(1.0, max(0.0, score / 10)))
        st.caption("The gauge shows the model's numerical output on the dataset's 0–10 scale.")
    else:
        st.markdown("""
        <div class="result-card">
          <div class="result-kicker">Prediction preview</div>
          <div style="font-family:Georgia,serif;font-size:1.6rem;margin:1.4rem 0 .7rem;">Your score will appear here.</div>
          <div class="result-note">Complete the form and choose “Read my signal” to run the saved model.</div>
        </div>
        """, unsafe_allow_html=True)

    st.info("For learning and exploration only. If you're concerned about your wellbeing, speak with a qualified professional or someone you trust.")

st.divider()
st.subheader("Inside the model")
one, two, three = st.columns(3)
one.metric("Dataset", "5,000 students")
two.metric("Test R²", "0.878")
three.metric("Test MAE", "0.347 / 10")
st.markdown("""
<div class="detail-note">The saved model is a random forest regression pipeline. The notebook uses a 70/30 train/test split, preprocessing for numeric and categorical inputs, and compares the default random forest with linear regression and a tuned forest. These test metrics are from the project's notebook, not a clinical validation study.</div>
""", unsafe_allow_html=True)
with st.expander("Model comparison and project links"):
    st.dataframe(pd.DataFrame([
        {"Model": "Linear regression", "Test R²": 0.740, "Test MAE": 0.536},
        {"Model": "Random forest (deployed)", "Test R²": 0.878, "Test MAE": 0.347},
        {"Model": "Random forest (tuned)", "Test R²": 0.865, "Test MAE": 0.369},
    ]), hide_index=True, use_container_width=True)
    st.link_button("View the GitHub repository", "https://github.com/ShreyanshJoshi4444/Mental-Health-Project")
