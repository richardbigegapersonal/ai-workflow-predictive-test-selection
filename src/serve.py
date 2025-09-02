import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from src.schema import SelectionRequest, SelectionResponse

app = FastAPI(title="Predictive Test Selection")

model = joblib.load("artifacts/model.joblib")

@app.post("/select", response_model=SelectionResponse)
def select(req: SelectionRequest):
    df = pd.DataFrame([p.dict() for p in req.pairs])
    feats = ["files_changed","lines_added","lines_deleted","author_risk",
             "subsystem","test_area","test_hist_fail","author"]
    # cold-start / safety: ensure min 150 tests if we have very low confidences
    df["_p"] = model.predict_proba(df[feats])[:,1]
    df = df.sort_values("_p", ascending=False)
    sel = df[(df["_p"] >= req.pmin) | (df.index < 150)].head(req.k)
    if sel.empty:
        raise HTTPException(422, "No tests selected under constraints.")
    return SelectionResponse(selected=sel["test_id"].tolist())
