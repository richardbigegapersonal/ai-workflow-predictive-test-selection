import json, os, joblib, pandas as pd
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

DATA = "data/commit_test_pairs.csv"
ART = "artifacts"
os.makedirs(ART, exist_ok=True)

FEATS = ["files_changed","lines_added","lines_deleted","author_risk",
         "subsystem","test_area","test_hist_fail","author"]

def model_select(grp, model, k=300, pmin=0.05):
    proba = model.predict_proba(grp[FEATS])[:,1]
    g = grp.assign(_p=proba).sort_values("_p", ascending=False)
    return g[(g["_p"] >= pmin) | (g.index < 150)].head(k)

def baseline_hist(grp, k=300):
    g = grp.sort_values("test_hist_fail", ascending=False)
    sel = g.head(k)
    # if fewer than k due to ties/NA, pad with integration/e2e then random
    if len(sel) < k:
        pad = g[g["test_area"].isin(["integration","e2e"])]
        sel = pd.concat([sel, pad.head(k-len(sel))]).drop_duplicates("test_id")
    if len(sel) < k:
        sel = pd.concat([sel, g.sample(n=k-len(sel), replace=False)]).drop_duplicates("test_id")
    return sel.head(k)

def baseline_random(grp, k=300):
    return grp.sample(n=min(k, len(grp)), replace=False, random_state=7)

def eval_strategy(df, selector, label="model", **kwargs):
    rows=[]
    for cid, grp in tqdm(df.groupby("commit_id"), desc=f"eval[{label}]"):
        sel = selector(grp, **kwargs)
        total_fails = int(grp["label_fail"].sum())
        covered_fails = int(grp.loc[sel.index, "label_fail"].sum())
        coverage = covered_fails/total_fails if total_fails>0 else 1.0
        rows.append({"commit_id":cid,
                     "tests_selected": int(len(sel)),
                     "total_fails": total_fails,
                     "covered_fails": covered_fails,
                     "fail_coverage": float(coverage)})
    out = pd.DataFrame(rows)
    out["strategy"]=label
    return out

if __name__ == "__main__":
    df = pd.read_csv(DATA)

    # global PR-AUC on a random split (sanity)
    model = joblib.load(f"{ART}/model.joblib")
    Xtr, Xva, ytr, yva = train_test_split(df[FEATS], df["label_fail"].astype(int),
                                          test_size=0.2, stratify=df["label_fail"], random_state=42)
    ap = average_precision_score(yva, model.predict_proba(Xva)[:,1])

    # compare strategies
    res_model   = eval_strategy(df, lambda g: model_select(g, model, k=300, pmin=0.05), label="model_k300_p05")
    res_hist    = eval_strategy(df, lambda g: baseline_hist(g, k=300), label="hist_k300")
    res_random  = eval_strategy(df, lambda g: baseline_random(g, k=300), label="random_k300")

    all_res = pd.concat([res_model, res_hist, res_random], ignore_index=True)
    all_res.to_csv(f"{ART}/per_commit_eval_all.csv", index=False)

    # summaries
    def summarize(d):
        return {
            "n_commits": int(len(d)),
            "mean_tests_selected": float(d["tests_selected"].mean()),
            "mean_fail_coverage": float(d["fail_coverage"].mean())
        }

    s_model  = summarize(res_model)
    s_hist   = summarize(res_hist)
    s_random = summarize(res_random)

    # compute "compute_savings" vs 12k baseline
    for s in (s_model, s_hist, s_random):
        s["compute_savings_vs_12k"] = 1.0 - (s["mean_tests_selected"]/12000.0)

    summary = {
        "avg_precision_val" : float(ap),
        "model"  : s_model,
        "hist"   : s_hist,
        "random" : s_random
    }
    with open(f"{ART}/eval_summary.json","w") as f: json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
