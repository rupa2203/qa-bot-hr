"""
Question by gpt : What do you think FastAPI does and why we need it?
Answer: it is a kind of business layer where we build the code present in API so that the code can be consumed by any
gpt answer: Exactly. FastAPI exposes your classifier as an HTTP endpoint so any frontend, app, or service can call it.

Without FastAPI:
Python script only → you run it manually
With FastAPI:
POST /classify → any app sends a complaint → gets JSON back
command to rn the fastapi : uvicorn app.main:app --reload
"""
"""
    sessions is a dict. sessions[request.session_id] = [] creates an empty history list for that session ID 
    if it doesn't exist yet.
        Overall block:
        First request from "user_001" → creates sessions["user_001"] = []
        Second request from "user_001" → finds existing list, uses it
        Request from "user_002" → creates separate sessions["user_002"] = []
    Each session ID gets its own isolated history.
    """

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.llm_fewshot_prompt import chain
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.rag import answer_question
from app.vectorstore import load_vectorstore


app=FastAPI()
# Store per-session history

# CORS — allows browser JS to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # dev only — lock down in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve HTML frontend at /static/index.html
app.mount("/static", StaticFiles(directory="app/static"), name="static")

#----------Week 1 work — complaint classifier with memory--------------------#

sessions={}

class CompliantRequst(BaseModel):
    username:str
    complaint:str
    session_id:str

@app.post("/classify")

def classify(request: CompliantRequst):
    # Get or create history for this session
    if request.session_id not in sessions:
        sessions[request.session_id]=[]
        
    history = sessions[request.session_id]
            
    try:
        result = chain.invoke({
            "complaint": request.complaint, 
            "history": history
        })
         # Update history
        history.append(HumanMessage(content=request.complaint))
        history.append(AIMessage(content=str(result)))
        sessions[request.session_id] = history[-4:]

        print(f"Session {request.session_id} history: {sessions[request.session_id]}")  # debug
         
        return {
            "username":request.username, 
            "complaint":request.complaint,
            "classification":result
        }
    except Exception as e:
            return {"error": str(e)}    

#======================(Week 2 work — HR RAG bot) =======================#
hr_sessions = {}  # separate from classify sessions

class HRRequest(BaseModel):
    question: str
    session_id: str = "default"  # optional — defaults to single session

@app.post("/hr/ask")
def ask_hr(request: HRRequest):
    if(request.session_id not in hr_sessions):
        hr_sessions[request.session_id]=[]
    history = hr_sessions[request.session_id]
    # Get or create history for this session
    return answer_question(request.question)

#======================(Retrieval only — no LLM call) =======================#
# Same retrieval step /hr/ask uses internally (see app/rag.py answer_question:
# same similarity_search_with_score call, same page+1/distance-to-score
# math) but returned on its own, with no LLM call. Lets retrieval quality be
# checked in isolation — cheap, no LLM cost — whenever documents/embeddings
# change, without also needing a full end-to-end answer graded at the same
# time. Not part of the normal user-facing chat flow.
# chroma_db rebuilt fresh 2026-09-28 from the current documents/ folder
# (HR Policy Manual 2023 (8).pdf + the canary file) — the old persisted
# store had stale metadata from a since-deleted "hr_policy.pdf".
class RetrieveRequest(BaseModel):
    query: str
    k: int = 2

@app.post("/hr/retrieve")
def retrieve_chunks(request: RetrieveRequest):
    vectorstore = load_vectorstore()
    docs_with_scores = vectorstore.similarity_search_with_score(request.query, k=request.k)
    return {
        "query": request.query,
        "chunks": [
            {
                "text": doc.page_content,
                "page": doc.metadata['page'] + 1,
                "score": 1 / (1 + distance),
                # Per-chunk document identity — langchain's loaders already put
                # this in metadata, it's just never been surfaced over HTTP
                # before (/hr/ask has no need for it, since it never has to
                # tell two source documents apart).
                "source": doc.metadata.get('source_file') or doc.metadata.get('source', 'unknown'),
            }
            for doc, distance in docs_with_scores
        ],
    }

#======================(Feedback loop — real user thumbs up/down) =======================#
# Writes into the SAME feedback-log.json that qaeval-playwright's qaFeedback.ts
# (gradeFeedback/convertFeedbackToTestCases/manageGoldenSet) reads via its own
# paths(outputDir) helper — this is the FeedbackEvent JSON shape, produced from
# Python since the RAG bot and the QA tooling are separate runtimes. Path is
# overridable via env var since the two projects live in separate repos.
FEEDBACK_LOG_PATH = Path(os.environ.get(
    "QAEVAL_FEEDBACK_LOG_PATH",
    r"C:/Users/Pranav/OneDrive/Desktop/qa-eval-plugin-hr-bot-test/feedback/feedback-log.json",
))

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str  # 'up' or 'down'
    sourceChunks: Optional[List[Any]] = None
    userComment: Optional[str] = None
    sessionId: Optional[str] = None

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    if request.rating not in ("up", "down"):
        return {"error": "rating must be 'up' or 'down'"}

    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = {"events": []}
    if FEEDBACK_LOG_PATH.exists():
        log = json.loads(FEEDBACK_LOG_PATH.read_text(encoding="utf-8"))

    event = {
        "id": f"FB{str(len(log['events']) + 1).zfill(5)}-{uuid.uuid4().hex[:6]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": request.question,
        "answer": request.answer,
        "sourceChunks": request.sourceChunks,
        "rating": request.rating,
        "userComment": request.userComment,
        "sessionId": request.sessionId,
        "environment": "production",
    }
    log["events"].append(event)
    FEEDBACK_LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")

    return {"status": "recorded", "id": event["id"]}

