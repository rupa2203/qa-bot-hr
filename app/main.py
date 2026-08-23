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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.llm_fewshot_prompt import chain
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.rag import answer_question


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

