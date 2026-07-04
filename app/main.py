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
from fastapi import FastAPI
from pydantic import BaseModel
from app.W1D1llm_fewshot_prompt import chain

app=FastAPI()

class CompliantRequst(BaseModel):
    username:str
    complaint:str

@app.post("/classify")
def classify(request: CompliantRequst):
    try:
        result = chain.invoke({"complaint": request.complaint})
        return {"username":request.username, "complaint":request.complaint, "classification":result}

    except Exception as e:
        return {"error": str(e)}