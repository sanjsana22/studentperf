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

df = pd.read_csv("../data/Student_Performance.csv")
model = pickle.load(open("../model/std_perf.pkl","rb"))

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.title{
font-size:42px;
font-weight:700;
text-align:center;
margin-bottom:30px;
}

.card{
background:white;
padding:25px;
border-radius:10px;
box-shadow:0px 4px 12px rgba(0,0,0,0.1);
height:160px;
color:black;
margin-bottom:30px;
}

.card h4{
color:black;
margin-bottom:10px;
}

.card p{
color:black;
font-size:15px;
}

.chart-label{
font-weight:600;
font-size:18px;
margin-top:10px;
}

.chart-desc{
color:gray;
margin-bottom:40px;
}

.note-card{
background:#fff5e6;
padding:20px;
border-radius:10px;
color:black;
font-size:16px;
margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown('<div class="title">Student Academic Performance Prediction</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# TABS
# ---------------------------------------------------

home_tab, pred_tab = st.tabs(["Home","Prediction"])

# ===================================================
# HOME PAGE
# ===================================================

with home_tab:

    st.subheader("Project Overview")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="card">
        <h4>Dataset Overview</h4>
        <p>Contains student academic data including study habits, sleep hours,
        extracurricular participation, and previous scores.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <h4>Machine Learning Model</h4>
        <p>A Linear Regression model trained to learn relationships
        between study patterns and academic performance.</p>
        </div>
        """, unsafe_allow_html=True)

    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.markdown("""
        <div class="card">
        <h4>Key Input Features</h4>
        <p>Hours Studied, Previous Scores, Extracurricular Activities,
        Sleep Hours and Practice Papers Solved.</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="card">
        <h4>Prediction Objective</h4>
        <p>Estimate the expected student performance index based
        on academic behaviour and study patterns.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")



# ---------------------------------------------------
# DATA VISUALIZATION
# ---------------------------------------------------

with home_tab:

    st.header("Data Insights from Student Dataset")

    # ---------------------------------------------
    # 1 Hours Studied vs Performance
    # ---------------------------------------------

    st.subheader("Performance vs Hours Studied")

    hours_perf = df.groupby("Hours Studied")["Performance Index"].mean().reset_index()

    fig1 = px.bar(
        hours_perf,
        x="Hours Studied",
        y="Performance Index",
        title="Average Performance Index for Different Study Hours",
        color="Performance Index"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.caption(
        "Insight: Students who study more hours tend to achieve higher performance scores."
    )

    st.markdown("---")

    # ---------------------------------------------
    # 2 Sleep Hours vs Performance
    # ---------------------------------------------

    st.subheader("Performance vs Sleep Hours")

    sleep_perf = df.groupby("Sleep Hours")["Performance Index"].mean().reset_index()

    fig2 = px.line(
        sleep_perf,
        x="Sleep Hours",
        y="Performance Index",
        markers=True,
        title="Effect of Sleep Hours on Student Performance"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Insight: Students with balanced sleep hours often maintain stable academic performance."
    )

    st.markdown("---")

    # ---------------------------------------------
    # 3 Previous Scores vs Performance
    # ---------------------------------------------

    st.subheader("Previous Scores vs Performance")

    
    fig3 = px.scatter(
    df,
    x="Previous Scores",
    y="Performance Index",
    title="Relationship Between Previous Scores and Final Performance"
)

    st.plotly_chart(fig3, use_container_width=True)

    st.caption(
        "Insight: Students who performed well previously are more likely to achieve higher performance again."
    )

    st.markdown("---")

    # ---------------------------------------------
    # 4 Practice Papers vs Performance
    # ---------------------------------------------

    st.subheader("Practice Papers vs Performance")

    paper_perf = df.groupby("Sample Question Papers Practiced")["Performance Index"].mean().reset_index()

    fig4 = px.line(
        paper_perf,
        x="Sample Question Papers Practiced",
        y="Performance Index",
        markers=True,
        title="Effect of Practice Papers on Performance"
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.caption(
        "Insight: Solving more practice papers helps students improve their academic performance."
    )

# ===================================================
# PREDICTION PAGE
# ===================================================

with pred_tab:

    st.subheader("Prediction System Overview")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="card">
        <h4>Dataset Overview</h4>
        <p>Student dataset containing study behaviour and academic performance indicators.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <h4>Machine Learning Model</h4>
        <p>Linear Regression model trained to predict performance index.</p>
        </div>
        """, unsafe_allow_html=True)

    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.markdown("""
        <div class="card">
        <h4>Key Input Features</h4>
        <p>Hours studied, sleep hours, previous scores, activities and practice papers.</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="card">
        <h4>Prediction Objective</h4>
        <p>Estimate expected academic performance based on study patterns.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.header("Predict Student Performance")

    st.markdown("""
    <div class="note-card">
    Enter student academic details below. The trained machine learning model will predict the expected performance index.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# INPUTS WITH VALIDATION
# ---------------------------------------------------
#####Hours studied ≈ 1 – 9
###Sleep hours ≈ 4 – 10
    hours = st.number_input("Hours Studied", min_value=0.0, max_value=12.0)

    prev = st.number_input("Previous Scores", min_value=0.0, max_value=100.0, step=0.5)

    extra = st.selectbox("Extracurricular Activities",["Select","Yes","No"])

    sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0)

    papers = st.number_input("Practice Papers Solved", min_value=0.0, max_value=9.0)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

    if st.button("Predict"):

        if extra == "Select":

            st.warning("Please select extracurricular activities")

        else:

            extra_val = 1 if extra=="Yes" else 0

            features = np.array([[hours, prev, extra_val, sleep, papers]])

            prediction = model.predict(features)

            prediction = max(0, prediction[0])

            st.success(f"Predicted Performance Index: {prediction:.2f}")



####| Feature                            | Valid Range |
##| ---------------------------------- | ----------- |
#| Hours Studied                      | 1 – 9       |
#| Previous Scores                    | 40 – 100    |
#| Extracurricular Activities         | Yes / No    |
#| Sleep Hours                        | 4 – 10      |
#| Practice Papers                    | 0 – 9       |
#| **Performance Index (prediction)** | **0 – 100** |
