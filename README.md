# HR QA Bot

AI-powered support classifier and HR policy assistant.

## Stack
- LangChain v1.3.10
- FastAPI + uvicorn
- OpenAI gpt-3.5-turbo
- Python 3.12

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`:OPENAI_API_KEY=your_key_here 
## Run
```bash
python -m uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs

## Endpoints
- `POST /classify` — classifies complaint into category, severity, action

## Week 1 — Completed
- LangChain LCEL chains
- ChatPromptTemplate + system/user roles
- JsonOutputParser + Pydantic Enum validation
- Few-shot prompting
- Error handling + fallbacks
- Session-based conversation memory
- FastAPI wrapper

## Roadmap
- Week 2: RAG pipeline + ChromaDB
- Week 3: Ragas evaluation
- Week 4: LangSmith tracing
- Week 5: PromptFoo regression tests
- Week 6: Playwright test automation
- Week 7: n8n workflows
- Week 8: Full CI/CD pipeline
- Week 9: Deployment