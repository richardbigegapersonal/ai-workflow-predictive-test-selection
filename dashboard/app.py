import json, pandas as pd, streamlit as st

st.title("Predictive Test Selection — Dashboard")
res = pd.read_csv("artifacts/per_commit_eval_all.csv")
with open("artifacts/eval_summary.json") as f: summary=json.load(f)

st.subheader("Summary")
st.json(summary)

st.subheader("Coverage distribution")
for label, g in res.groupby("strategy"):
    st.line_chart(g["fail_coverage"].sort_values().reset_index(drop=True), height=200)

st.subheader("Per-commit table (sample)")
st.dataframe(res.sample(100, random_state=1))
