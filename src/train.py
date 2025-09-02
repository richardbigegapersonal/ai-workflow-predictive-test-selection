import json, os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import xgboost as xgb

DATA = "data/commit_test_pairs.csv"
ART = "artifacts"
os.makedirs(ART, exist_ok=True)

def load():
    df = pd.read_csv(DATA)
    # keep a realistic subset of features
    feats = ["files_changed","lines_added","lines_deleted","author_risk",
             "subsystem","test_area","test_hist_fail","author"]
    X = df[feats]
    y = df["label_fail"].astype(int)
    return df, X, y

def build_pipeline():
    num = ["files_changed","lines_added","lines_deleted","author_risk","test_hist_fail"]
    cat = ["subsystem","test_area","author"]
    pre = ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat)
    ])
    # XGBoost model
    model = xgb.XGBClassifier(
        max_depth=6, learning_rate=0.1, n_estimators=300,
        subsample=0.9, colsample_bytree=0.8, eval_metric="logloss",
        n_jobs=4, tree_method="hist"
    )
    pipe = Pipeline([("pre", pre), ("clf", model)])
    return pipe

if __name__ == "__main__":
    df, X, y = load()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe = build_pipeline()
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xva)[:,1]
    ap = average_precision_score(yva, proba)

    # Save model
    import joblib
    joblib.dump(pipe, f"{ART}/model.joblib")
    
    # Feature importance (built-in + permutation)
    try:
        # 3a) Built-in importance (XGB)
        # Extract booster and OHE feature names
        import numpy as np, joblib
        clf = pipe.named_steps["clf"]
        pre = pipe.named_steps["pre"]
        num = ["files_changed","lines_added","lines_deleted","author_risk","test_hist_fail"]
        cat = ["subsystem","test_area","author"]
        ohe = pre.named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(cat))
        feat_names = num + cat_names
        booster = clf.get_booster()
        fmap = booster.get_score(importance_type="gain")
        # Map to human-readable
        # XGB names features f0,f1,... in the order provided by the pipeline after transform
        # We approximate by sorting fmap keys and aligning (works with tree_method='hist')
        sorted_items = sorted(((int(k[1:]), v) for k,v in fmap.items()), key=lambda x: x[0])
        imp = [{"feature": feat_names[i] if i < len(feat_names) else f"f{i}",
                "gain": float(v)} for i,v in sorted_items]
        with open(f"{ART}/feature_importance_gain.json","w") as f: json.dump(imp, f, indent=2)
    except Exception as e:
        print("Feature importance (gain) skipped:", e)
    
    # 3b) Permutation importance on a small validation slice (model-agnostic)
    try:
        from sklearn.inspection import permutation_importance
        import numpy as np
        res = permutation_importance(pipe, Xva, yva, n_repeats=5, random_state=0, n_jobs=2, scoring="average_precision")
        # Build names for numeric + expanded categorical columns (approximate: show base names)
        base_names = num + cat
        pimps = [{"feature": base_names[i], "perm_importance": float(res.importances_mean[i])}
                 for i in range(len(base_names))]
        with open(f"{ART}/feature_importance_permutation.json","w") as f: json.dump(pimps, f, indent=2)
    except Exception as e:
        print("Permutation importance skipped:", e)


    # Save simple metrics
    metrics = {"average_precision": float(ap), "n_val": int(len(yva))}
    with open(f"{ART}/metrics.json","w") as f: json.dump(metrics, f, indent=2)
    print("Saved artifacts:", metrics)
