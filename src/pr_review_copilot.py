import json
import os
import subprocess
import textwrap
from typing import List

import requests

# -----------------------------
# Config
# -----------------------------
REPO = os.environ["GITHUB_REPOSITORY"]
EVENT_PATH = os.environ["GITHUB_EVENT_PATH"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
API_BASE = "https://api.github.com"

# Use a smaller summarization model to keep CI fast.
# You can switch to a bigger model later.
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# -----------------------------
# Helpers
# -----------------------------
def run(cmd: List[str]) -> str:
    out = subprocess.check_output(cmd, text=True)
    return out.strip()

def get_pr_context():
    with open(EVENT_PATH, "r") as f:
        ev = json.load(f)
    pr = ev["pull_request"]
    base_sha = pr["base"]["sha"]
    head_sha = pr["head"]["sha"]
    number = pr["number"]
    return number, base_sha, head_sha

def get_diff(base_sha: str, head_sha: str) -> str:
    # Unified diff with minimal context, smaller input to model
    return run(["git", "diff", "--unified=0", f"{base_sha}...{head_sha}"])

def chunk_text(txt: str, max_chars: int = 1800):
    # naive chunking to avoid hitting model max len
    for i in range(0, len(txt), max_chars):
        yield txt[i : i + max_chars]

def risk_flag(diff: str) -> List[str]:
    rules = [
        ("Removed authorization or checks", ["remove", "deleted", "bypass", "disable", "drop check"]),
        ("Touched error handling", ["except", "try:", "catch", "logger.error", "rollback"]),
        ("Changed configuration/infra", ["terraform", "helm", "Dockerfile", "ingress", "secret", "env:"]),
    ]
    found = []
    low = diff.lower()
    for label, patterns in rules:
        if any(p.lower() in low for p in patterns):
            found.append(f"- {label}")
    return found

def post_pr_comment(repo: str, pr_number: int, body: str):
    url = f"{API_BASE}/repos/{repo}/issues/{pr_number}/comments"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    resp = requests.post(url, headers=headers, json={"body": body})
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to post comment: {resp.status_code} {resp.text}")

# -----------------------------
# Summarizer (with graceful fallback)
# -----------------------------
def summarize_texts(chunks: List[str]) -> List[str]:
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization", model=MODEL_NAME)
        outs = []
        for c in chunks:
            s = summarizer(c, max_length=120, min_length=10, do_sample=False)[0]["summary_text"]
            outs.append(s)
        return outs
    except Exception as e:
        # Fallback if model download fails; we don't fail PRs
        return [f"(Fallback) Summary unavailable in CI: {e}"]

# -----------------------------
# Main
# -----------------------------
def main():
    pr_number, base_sha, head_sha = get_pr_context()
    diff = get_diff(base_sha, head_sha)

    if not diff.strip():
        post_pr_comment(REPO, pr_number, "PR Review Copilot: No diff detected.")
        return

    # Keep only hunks, strip noisy metadata to improve summaries
    useful_lines = []
    for line in diff.splitlines():
        if line.startswith(("+++", "---", "index", "diff --git")):
            continue
        useful_lines.append(line)
    slim_diff = "\n".join(useful_lines).strip()

    # Chunk & summarize
    chunks = list(chunk_text(slim_diff, max_chars=1600))
    summaries = summarize_texts(chunks)

    # Risk heuristics
    risks = risk_flag(slim_diff)
    risk_block = "\n".join(risks) if risks else "- No obvious risks detected."

    body = textwrap.dedent(f"""
    ### 🤖 PR Review Copilot — Code Change Summary

    **Model:** `{MODEL_NAME}`  
    **Commits:** `{base_sha[:7]}...{head_sha[:7]}`

    **Summaries:**
    {"".join(f"- {s}\n" for s in summaries)}

    **Risk Hints:**
    {risk_block}

    _Note: This is an assistant for human reviewers; please verify conclusions._
    """)

    post_pr_comment(REPO, pr_number, body)

if __name__ == "__main__":
    main()
