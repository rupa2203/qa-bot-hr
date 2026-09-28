import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.vectorstore import load_vectorstore

load_dotenv()  # Load environment variables from .env file

MODEL_NAME = "gpt-3.5-turbo"
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an HR policy assistant. Answer questions using only the context provided.
    If the answer is not in the context, say "I don't have that information in the HR policy."
    Always mention the page number your answer comes from."""),
    ("user","Context:\n {context}\n\n Question:{question}")
])


chain = prompt | llm | StrOutputParser()

def answer_question(query: str) -> dict:
    vectorstore = load_vectorstore()
    # with_score returns (doc, distance) tuples — Chroma's default index here
    # is L2 distance (lower = more similar), not a bounded 0-1 similarity.
    # 1/(1+distance) is a standard monotonic conversion to a "higher = more
    # similar" score in (0, 1] — treat it as an approximation, not a
    # precisely calibrated confidence value; adjust minRetrievalScore in
    # qaeval.config.ts based on the real values you see, not by assumption.
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=2)
    docs = [doc for doc, _ in docs_with_scores]

    context = "\n\n".join([
        f"[Page {doc.metadata['page']+1}]: {doc.page_content}"
        for doc in docs
    ])

    start = time.time()
    answer = chain.invoke({
        "context": context,
        "question": query
    })
    latency_ms = int((time.time() - start) * 1000)

    # Fallback detection
    if "don't have that information" in answer.lower():
        return {
            "question": query,
            "answer": answer,
            "sources": [],
            "chunks": [],
            "in_scope": False,
            "latency_ms": latency_ms,
            "model": MODEL_NAME
        }

    return {
        "question": query,
        "answer": answer,
        "sources": [doc.metadata['page']+1 for doc in docs],
        "chunks": [
            {
                "text": doc.page_content,
                "page": doc.metadata['page']+1,
                "score": 1 / (1 + distance),
            }
            for doc, distance in docs_with_scores
        ],
        "in_scope": True,
        "latency_ms": latency_ms,
        "model": MODEL_NAME
    }

if __name__ == "__main__":
    result = answer_question("what is the company's revenue?")
    print(result)