import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from datetime import datetime, timedelta

st.set_page_config(page_title="EduViz-Q (Streamlit)", layout="wide")

# --- Helpers 
@st.cache_data
def load_data(path="mock_data.json"):
    df = pd.read_json(path)
    # normalize types
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["engagement_score"] = pd.to_numeric(df["engagement_score"], errors="coerce")
    df["time_spent_min"] = pd.to_numeric(df["time_spent_min"], errors="coerce")
    return df

def avg_engagement_by_date(df):
    out = df.groupby("date")["engagement_score"].mean().reset_index()
    return out.sort_values("date")

def avg_by_concept(df):
    out = df.groupby("concept")["engagement_score"].mean().reset_index()
    return out.sort_values("engagement_score", ascending=False)

def heatmap_matrix(df, students=None, concepts=None):
    if students is None:
        students = sorted(df["student_id"].unique())
    if concepts is None:
        concepts = sorted(df["concept"].unique())
    pivot = df.pivot_table(index="student_id", columns="concept", values="engagement_score", aggfunc="mean")
    pivot = pivot.reindex(index=students, columns=concepts)
    return pivot

def get_concept_insights(df, threshold=0.4, pct_limit=30):
    out = []
    byc = df.groupby("concept")["engagement_score"].agg(list).to_dict()
    for c, arr in byc.items():
        arr = [v for v in arr if not np.isnan(v)]
        if len(arr) == 0:
            continue
        low = sum(1 for v in arr if v < threshold)
        pct = (low / len(arr)) * 100
        if pct >= pct_limit:
            out.append(f"{pct:.0f}% of interactions for '{c}' have engagement < {threshold} — consider adding a visual demo or peer activity.")
    return out

def get_student_insights(df, drop_delta=0.25):
    out = []
    bys = df.sort_values("timestamp").groupby("student_id")["engagement_score"].apply(list).to_dict()
    for s, arr in bys.items():
        arr = [v for v in arr if not np.isnan(v)]
        if len(arr) < 2:
            continue
        avg = np.mean(arr)
        recent = arr[-1]
        if avg - recent >= drop_delta:
            out.append(f"Student {s} shows engagement drop (avg {avg:.2f} → recent {recent:.2f}). Consider check-in.")
    return out

def to_csv_bytes(df):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode('utf-8')

# --- Load data 
DATA_PATH = r"C:\Users\POWER\source\repos\PythonApplication1\mock_data.json"
df_all = load_data(DATA_PATH)

# --- Sidebar controls 
st.sidebar.title("Filters & Settings")
view = st.sidebar.selectbox("View", ["Teacher", "Student"])
# time filter default: last 14 days in mock data
min_date = df_all["date"].min()
max_date = df_all["date"].max()
default_from = max_date - timedelta(days=14) if (max_date - min_date).days >= 14 else min_date
date_range = st.sidebar.date_input("Date range", [default_from, max_date], min_value=min_date, max_value=max_date)
start_date, end_date = date_range if isinstance(date_range, (list, tuple)) else (min_date, max_date)
concepts_all = sorted(df_all["concept"].unique())
selected_concepts = st.sidebar.multiselect("Concepts", concepts_all, default=concepts_all)
students_all = sorted(df_all["student_id"].unique())
selected_student = st.sidebar.selectbox("Student (Student view)", ["All"] + students_all)

# insight parameters
st.sidebar.markdown("---")
threshold = st.sidebar.slider("Low engagement threshold", 0.1, 0.6, 0.4, 0.05)
pct_limit = st.sidebar.slider("Percent low interactions to trigger suggestion", 10, 60, 30, 5)
drop_delta = st.sidebar.slider("Student drop delta (avg - recent)", 0.1, 0.5, 0.25, 0.05)

# apply filters
mask = (
    (df_all["date"] >= start_date)
    & (df_all["date"] <= end_date)
    & (df_all["concept"].isin(selected_concepts))
)
df = df_all[mask].copy()

# --- Page header 
st.header("EduViz-Q — Visualizing Engagement in Quantum Computing")
st.markdown("Lightweight prototype for teachers and students to identify engagement patterns in quantum computing lessons.")

# --- Teacher View 
if view == "Teacher":
    col_l, col_r = st.columns([3, 1])

    with col_l:
        st.subheader("Engagement Pulse")
        avg = avg_engagement_by_date(df)
        if avg.empty:
            st.info("No data for selected filters.")
        else:
            fig_line = px.line(avg, x="date", y="engagement_score", markers=True,
                               title="Average engagement (by date)",
                               labels={"engagement_score": "Avg engagement", "date": "Date"})
            fig_line.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_line, use_container_width=True)

        st.subheader("Concept Averages")
        cavg = avg_by_concept(df)
        fig_bar = px.bar(cavg, x="engagement_score", y="concept", orientation="h",
                         title="Average engagement by concept", labels={"engagement_score": "Avg engagement", "concept": "Concept"})
        fig_bar.update_xaxes(range=[0, 1])
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.subheader("Insights")
        concept_ins = get_concept_insights(df, threshold=threshold, pct_limit=pct_limit)
        student_ins = get_student_insights(df, drop_delta=drop_delta)
        if not concept_ins and not student_ins:
            st.success("No critical issues detected for selected filters.")
        else:
            for it in concept_ins:
                st.warning(it)
            for it in student_ins:
                st.info(it)

        st.markdown("---")
        st.subheader("Export Data")
        csv = to_csv_bytes(df)
        st.download_button("Download filtered CSV", data=csv, file_name="eduviz_filtered.csv", mime="text/csv")

    st.markdown("### Concept × Student Heatmap")
    pivot = heatmap_matrix(df)
    if pivot.isnull().all().all():
        st.info("Heatmap empty for selected filters.")
    else:
        fig_heat = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale="RdYlGn_r",
                             labels=dict(x="Concept", y="Student", color="Engagement"))
        fig_heat.update_coloraxes(cmin=0, cmax=1)
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    st.markdown("**Quick notes:** The insights above are rule-based and explainable. For a production system, replace with authenticated LMS integration and privacy controls.")

# --- Student View 
else:
    st.subheader("Student Dashboard")
    # allow selecting a student
    student = st.selectbox("Select student", students_all, index=0)
    sdata = df[df["student_id"] == student].copy()
    if sdata.empty:
        st.info("No data for this student in the selected date range / concepts.")
    else:
        # Weekly engagement bar
        s_week = sdata.groupby("date")["engagement_score"].mean().reset_index()
        fig_sbar = px.bar(s_week, x="date", y="engagement_score", title=f"Weekly engagement — {student}")
        fig_sbar.update_yaxes(range=[0,1])
        st.plotly_chart(fig_sbar, use_container_width=True)

        # Concept radar-like (polar)
        comp = sdata.groupby("concept")["engagement_score"].mean().reset_index()
        fig_polar = px.line_polar(comp, r="engagement_score", theta="concept", line_close=True,
                                  title="Concept proficiency (approx.)")
        fig_polar.update_traces(fill="toself")
        fig_polar.update_polars(radialaxis=dict(range=[0,1]))
        st.plotly_chart(fig_polar, use_container_width=True)

        st.markdown("### Reflection")
        reflection = st.text_area("Write a short reflection: what concept felt hardest? Any questions?", height=120)
        if st.button("Save reflection (local only)"):
            st.success("Reflection saved locally (demo). You can copy it for notes.")

        st.markdown("---")
        st.markdown("**Personal insights**")
        myins = get_student_insights(sdata)
        if not myins:
            st.write("No significant drop detected for this student.")
        else:
            for m in myins:
                st.warning(m)

# --- Footer 
st.markdown("---")
st.caption("EduViz-Q prototype — mock data. Built for hackathon demo. (Streamlit edition)")

