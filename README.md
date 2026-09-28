# HR QA Bot
AI-powered support classifier and HR policy assistant.

# HR Policy QA Bot — AI Evaluation Pipeline

An end-to-end RAG application with a full evaluation 
suite built on top. Built to demonstrate production-grade 
AI evaluation practices.

## What this project demonstrates
- RAG pipeline — PDF ingestion → ChromaDB → retrieval → answer
- LangChain LCEL chains with structured JSON output
- FastAPI endpoints with session memory
- Evaluation pipeline — 16 test cases, pass/fail scoring, 
  confidence rating
- Safety testing — out-of-scope question detection (100% accuracy)

## Eval results
| Metric | Score |
|---|---|
| Overall pass rate | 16/16 (100%) |
| Out-of-scope detection | 2/2 (100%) |
| In-scope accuracy | 14/14 (100%) |

## Stack
| Layer | Tools |
|---|---|
| LLM | OpenAI gpt-3.5-turbo |
| Orchestration | LangChain LCEL |
| Vector DB | ChromaDB |
| API | FastAPI + uvicorn |
| Evaluation | Custom eval pipeline |
| CI/CD | GitHub Actions (coming) |

## Project structure
ai-bot-hr/
├── app/          ← LangChain chains, RAG pipeline, FastAPI
├── evals/        ← eval dataset, scoring logic, results
├── tests/        ← pytest suite (in progress)
└── documents/    ← source HR policy PDF

### Detailed project structure 

qa-bot-hr/
│
├── app/
│   ├── llm.py                    ← LangChain chain v1
│   │                               - ChatPromptTemplate
│   │                               - JsonOutputParser + Pydantic
│   │                               - CategoryEnum, SeverityEnum
│   │                               - chain = prompt | llm | parser
│   │
│   ├── llm_fewshot_prompt.py     ← LangChain chain v2 (production)
│   │                               - MessagesPlaceholder (memory)
│   │                               - Few-shot prompting
│   │                               - SeverityEnum validation
│   │                               - Used by FastAPI /classify
│   │
│   ├── rag.py                    ← RAG pipeline
│   │                               - load_vectorstore()
│   │                               - similarity_search (k=2)
│   │                               - answer_question()
│   │                               - Fallback detection (in_scope)
│   │                               - Source page citation
│   │
│   ├── ingest.py                 ← PDF/TXT loader
│   │                               - PyPDFLoader / TextLoader
│   │                               - RecursiveCharacterTextSplitter
│   │                               - chunk_size=200, overlap=50
│   │                               - load_and_chunk_file() — single file
│   │                               - load_and_chunk_folder() — whole folder
│   │
│   ├── vectorstore.py            ← ChromaDB
│   │                               - build_vectorstore() / _from_folder()
│   │                               - sync_vectorstore_from_folder() —
│   │                                 incremental add/update/remove by
│   │                                 content hash, no full rebuild needed
│   │                                 (see "Updating documents" below)
│   │                               - load_vectorstore()
│   │                               - retrieve()
│   │                               - print_embeddings()
│   │
│   ├── main.py                   ← FastAPI app
│   │                               - POST /classify (complaint bot)
│   │                               - POST /hr/ask (HR RAG bot)
│   │                               - POST /hr/retrieve (retrieval only,
│   │                                 no LLM call — for eval tooling)
│   │                               - POST /feedback (thumbs up/down log)
│   │                               - Session memory (sessions dict)
│   │                               - CORS middleware
│   │                               - Static HTML frontend
│   │
│   ├── ConversationBufferWindowMemory.py  ← memory experiments
│   │
│   ├── test.py                   ← quick test scripts
│   │
│   └── static/
│       └── index.html            ← frontend UI
│
├── evals/
│   ├── test_dataset.json         ← 16 test questions + expected topics
│   ├── run_evals.py              ← eval pipeline
│   │                               - loops through all 16 questions
│   │                               - verdict: PASS/FAIL per question
│   │                               - confidence: high/low/none
│   │                               - summary: score %
│   │                               - saves results.json
│   │
│   └── results.json              ← 16/16 PASS, Score 100%
│
├── documents/
│   └── *.pdf / *.txt             ← source documents embedded into ChromaDB
│
├── ingest_manifest.json          ← filename → content-hash map used by
│                                    sync_vectorstore_from_folder() to know
│                                    what's already embedded
│
├── chroma_db/                    ← vector embeddings (gitignored — not
│                                    committed; regenerate locally, see
│                                    "Updating documents" below)
│
├── tests/                        ← empty — Week 2 pytest goes here
├── workflows/                    ← empty — Week 9 CI/CD goes here
│
├── vectorstore.py                ← root level (duplicate — can delete)
├── .env                          ← OPENAI_API_KEY
├── .gitignore
├── requirements.txt
└── README.md

## How to run
# 1. Clone and install
git clone https://github.com/rupa2203/qa-bot-hr.git
cd qa-bot-hr
pip install -r requirements.txt

# 2. Add your API key
cp .env.example .env
# edit .env and add OPENAI_API_KEY=your_key

# 3. Build vector store (first time only — embeds everything in documents/)
python -m app.vectorstore sync documents

# 4. Run the API
uvicorn app.main:app --reload

# 5. Run eval suite
python -m evals.run_evals

## Updating documents
Drop a new file into documents/, or edit an existing one, then re-run:
    python -m app.vectorstore sync documents
This is incremental — it only embeds what's new or changed (by content
hash, tracked in ingest_manifest.json). A brand-new file gets embedded and
added; an edited file has just its own old chunks deleted and replaced;
an unchanged file is skipped entirely (no re-embedding, no API cost); a
file removed from documents/ has its chunks deleted too. No full
chroma_db wipe/rebuild needed for routine document changes — that's only
ever required if the store itself gets corrupted or you want to start over,
via build_vectorstore_from_folder().

## Eval report
Q1:  How many annual leave days...  → PASS (confidence: high)
Q2:  Can unused leave be carried... → PASS (confidence: high)
...
Q15: What is the company stock...   → PASS (correctly rejected)
Q16: How should be a mattress...    → PASS (correctly rejected)

Score: 16/16 | 100%
-----steps to build the project------
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
- `POST /hr/ask` — HR policy RAG chatbot; answers a question using retrieved
  chunks, returns `answer`, `chunks`, `sources`, `latency_ms`, `model`
- `POST /hr/retrieve` — retrieval only, no LLM call; returns the top-k
  chunks + scores for a query. Used by eval tooling to grade retrieval
  quality in isolation, without generation cost.
- `POST /feedback` — records a real user's thumbs up/down on an `/hr/ask`
  answer into a feedback log (path configurable via
  `QAEVAL_FEEDBACK_LOG_PATH`)

