from pydantic import BaseModel, Field
from typing import List

class CommitTestPair(BaseModel):
    commit_id: str
    test_id: str
    files_changed: int
    lines_added: int
    lines_deleted: int
    subsystem: str
    author: str
    author_risk: float
    test_area: str
    test_hist_fail: float

class SelectionRequest(BaseModel):
    pairs: List[CommitTestPair]
    k: int = Field(300, ge=50, le=1000)        # guardrails
    pmin: float = Field(0.05, ge=0.0, le=0.5)  # guardrails

class SelectionResponse(BaseModel):
    selected: List[str]  # list of test_id
