import os, json, joblib, pandas as pd
from pathlib import Path

ART = "artifacts"
DATA = "data/commit_test_pairs.csv"

def select_for_commit(df, model, commit_id, k=300, pmin=0.03):
    sub = df[df.commit_id == commit_id].copy()
    if sub.empty: 
        return pd.DataFrame(columns=sub.columns.tolist()+["_score"])
    feats = ["files_changed","lines_added","lines_deleted","author_risk",
             "subsystem","test_area","test_hist_fail","author"]
    sub["_score"] = model.predict_proba(sub[feats])[:,1]
    # guardrails: ensure a minimum of 150 tests to reduce risk of false skips
    sel = sub.sort_values("_score", ascending=False)
    sel = sel[(sel["_score"] >= pmin) | (sel.index < 150)].head(k)
    return sel[["commit_id","test_id","_score"]]

if __name__ == "__main__":
    model = joblib.load(f"{ART}/model.joblib")
    df = pd.read_csv(DATA)
    commit_id = df.commit_id.iloc[0]  # demo
    out = select_for_commit(df, model, commit_id, k=300, pmin=0.05)
    out.to_csv(f"{ART}/selected_tests_{commit_id}.csv", index=False)
    print("Wrote", f"{ART}/selected_tests_{commit_id}.csv", "rows:", len(out))
