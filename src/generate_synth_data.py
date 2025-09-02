"""
Generate synthetic commit–test data with realistic correlations:
- Certain subsystems + test_areas fail more often
- High author_risk / many lines_added → higher fail odds
- A small set of "flaky" tests add noise
"""
import numpy as np, pandas as pd
from pathlib import Path
rng = np.random.default_rng(42)

N_COMMITS   = 1500
N_TESTS     = 12000
N_PAIRS     = 250_000  # sparse sampling of commit×test

subsystems  = ["driver", "ui", "infra", "ml", "build", "network"]
test_areas  = ["unit", "integration", "perf", "e2e"]
authors     = [f"dev_{i:03d}" for i in range(200)]

# commit-level table
commits = pd.DataFrame({
    "commit_id": [f"c_{i:06d}" for i in range(N_COMMITS)],
    "files_changed": rng.integers(1, 32, size=N_COMMITS),
    "lines_added": rng.poisson(120, size=N_COMMITS).clip(1, 3000),
    "lines_deleted": rng.poisson(60, size=N_COMMITS).clip(0, 2000),
    "subsystem": rng.choice(subsystems, size=N_COMMITS, p=[.18,.20,.16,.18,.14,.14]),
    "author": rng.choice(authors, size=N_COMMITS),
})
# author_risk ~ historical buginess
author_hist = pd.DataFrame({"author": authors, "author_risk": rng.beta(2,10,size=len(authors))*1.6})
commits = commits.merge(author_hist, on="author", how="left")

# sample commit–test pairs
pair_idx = pd.DataFrame({
    "commit_id": rng.choice(commits["commit_id"], size=N_PAIRS),
    "test_id": [f"t_{i:06d}" for i in rng.integers(0, N_TESTS, size=N_PAIRS)]
})

# join commit features
df = pair_idx.merge(commits, on="commit_id", how="left")
df["test_area"] = rng.choice(test_areas, size=len(df), p=[.55,.25,.12,.08])

# test historical fail rate
test_hist = pd.DataFrame({
    "test_id": [f"t_{i:06d}" for i in range(N_TESTS)],
    "test_hist_fail": rng.beta(1.5, 60, size=N_TESTS)  # mostly low rates
})
df = df.merge(test_hist, on="test_id", how="left")

# make some tests "flaky"
flaky_mask = rng.random(N_TESTS) < 0.03
flaky = pd.Series(flaky_mask, index=test_hist["test_id"].values)
df["is_flaky"] = df["test_id"].map(flaky).fillna(False)

# failure logit
# base by area and subsystem
area_w = {"unit": -4.0, "integration": -3.3, "perf": -3.6, "e2e": -3.0}
sub_w  = {"driver": 0.35, "ui": 0.15, "infra": 0.25, "ml": 0.30, "build": 0.10, "network": 0.20}

logit = (
    -4.2
    + df["files_changed"]*0.015
    + np.log1p(df["lines_added"])*0.08
    + np.log1p(df["lines_deleted"])*0.04
    + df["author_risk"]*0.9
    + df["test_hist_fail"]*4.0
    + df["test_area"].map(area_w)
    + df["subsystem"].map(sub_w)
    + df["is_flaky"].astype(float)*0.6
)
p = 1/(1+np.exp(-logit))
y = rng.binomial(1, p)

df_out = df[[
    "commit_id","test_id","files_changed","lines_added","lines_deleted",
    "author_risk","subsystem","test_area","test_hist_fail","author"
]].copy()
df_out["label_fail"] = y

# write
Path("data").mkdir(exist_ok=True)
df_out.to_csv("data/commit_test_pairs.csv", index=False)
commits.to_csv("data/commits.csv", index=False)
print("Wrote data/commit_test_pairs.csv and data/commits.csv")
