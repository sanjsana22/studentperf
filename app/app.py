import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Student Academic Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATA + MODEL
# ---------------------------------------------------

df = pd.read_csv("data/Student_Performance.csv")
model = pickle.load(open("model/std_perf.pkl","rb"))

# ---------------------------------------------------
# UI STYLING
# ---------------------------------------------------

st.markdown("""
<style>

.stApp{
background-color:#0B1F3A;
color:white;
font-family: 'Segoe UI', sans-serif;
}

button[data-baseweb="tab"]{
font-size:22px !important;
font-weight:700;
color:white !important;
}

.title{
font-size:42px;
font-weight:700;
text-align:center;
margin-bottom:30px;
}

.section-title{
font-size:30px;
font-weight:700;
color:#D6E6FF;
margin-top:20px;
}

.chart-title{
font-size:20px;
font-weight:600;
color:#D6E6FF;
}

.stNumberInput input{
background-color:#EAF2FF;
color:black !important;
}

.stSelectbox div{
background-color:#EAF2FF;
color:black !important;
}

.stSelectbox span{
color:black !important;
}

/* -------- DROPDOWN ARROW COLOR FIX -------- */

[data-baseweb="select"] svg{
fill:black !important;
}

/* ------------------------------------------ */

.stButton button{
background-color:#4A90E2;
color:white;
font-weight:700;
border-radius:8px;
padding:8px 22px;
}

.stButton button:hover{
background-color:#2E6CB8;
}

.card{
background:#1F3A5F;
padding:25px;
border-radius:12px;
box-shadow:0px 6px 16px rgba(0,0,0,0.3);
height:160px;
margin-bottom:30px;
transition:0.3s;
}

.card:hover{
background:#4A90E2;
transform:scale(1.03);
}

.chart-desc{
color:#C7DAFF;
font-size:14px;
margin-bottom:30px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown('<div class="title">Student Academic Performance Prediction</div>', unsafe_allow_html=True)

home_tab, pred_tab = st.tabs(["Home","Prediction"])

# ===================================================
# HOME PAGE
# ===================================================

with home_tab:

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
        <h4>Dataset Overview</h4>
        Student academic dataset including study hours,
        sleep hours, and exam scores.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <h4>Machine Learning Model</h4>
        Linear Regression model predicting
        academic performance index.
        </div>
        """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        <div class="card">
        <h4>Input Features</h4>
        Hours Studied, Previous Scores,
        Extracurricular Activities,
        Sleep Hours, Practice Papers.
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="card">
        <h4>Prediction Goal</h4>
        Estimate student performance index
        using study patterns.
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------
# DATA INSIGHTS
# ---------------------------------------------------

    st.markdown('<div class="section-title">Data Insights</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown('<div class="chart-title">Study Hours vs Performance</div>', unsafe_allow_html=True)

        hours_perf = df.groupby("Hours Studied")["Performance Index"].mean().reset_index()

        fig1 = px.bar(hours_perf,x="Hours Studied",y="Performance Index")
        st.plotly_chart(fig1,use_container_width=True)

        st.markdown('<div class="chart-desc">Students who study more hours tend to score higher.</div>', unsafe_allow_html=True)

    with col2:

        st.markdown('<div class="chart-title">Sleep Hours vs Performance</div>', unsafe_allow_html=True)

        sleep_perf = df.groupby("Sleep Hours")["Performance Index"].mean().reset_index()

        fig2 = px.line(sleep_perf,x="Sleep Hours",y="Performance Index",markers=True)
        st.plotly_chart(fig2,use_container_width=True)

        st.markdown('<div class="chart-desc">Balanced sleep helps maintain academic performance.</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:

        st.markdown('<div class="chart-title">Previous Scores vs Performance</div>', unsafe_allow_html=True)

        fig3 = px.scatter(df,x="Previous Scores",y="Performance Index")
        st.plotly_chart(fig3,use_container_width=True)

        st.markdown('<div class="chart-desc">Students with higher previous scores tend to maintain performance.</div>', unsafe_allow_html=True)

    with col4:

        st.markdown('<div class="chart-title">Practice Papers vs Performance</div>', unsafe_allow_html=True)

        paper_perf = df.groupby("Sample Question Papers Practiced")["Performance Index"].mean().reset_index()

        fig4 = px.line(paper_perf,x="Sample Question Papers Practiced",y="Performance Index",markers=True)
        st.plotly_chart(fig4,use_container_width=True)

        st.markdown('<div class="chart-desc">Practicing more papers improves exam readiness.</div>', unsafe_allow_html=True)


# ===================================================
# PREDICTION PAGE
# ===================================================

with pred_tab:

    col1, col2 = st.columns(2)

    with col1:
        hours = st.number_input("Hours Studied",1.0,9.0,5.0,key="hours_input")
        prev = st.number_input("Previous Scores",40.0,100.0,70.0,key="prev_scores")
        extra = st.selectbox("Extracurricular Activities",["Yes","No"],key="extra_activity")

    with col2:
        sleep = st.number_input("Sleep Hours",4.0,10.0,7.0,key="sleep_hours")
        papers = st.number_input("Practice Papers Solved",0.0,9.0,3.0,key="practice_papers")

    st.write("")

    predict = st.button("Predict")

    if predict:

        extra_val = 1 if extra=="Yes" else 0
        features = np.array([[hours,prev,extra_val,sleep,papers]])
        prediction = model.predict(features)[0]

        st.success(f"Predicted Student Performance: {prediction:.2f}")