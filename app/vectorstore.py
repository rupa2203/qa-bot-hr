from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from app.ingest import load_and_chunks
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def build_vectorstore(pdf_path: str):
    chunks = load_and_chunks(pdf_path)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore

def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

def retrieve(query:str, k:int=2):
    vectorstore=load_vectorstore()
    results = vectorstore.similarity_search(query,k=k)
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.page_content[:200]}")  # Print first 200 characters of each result
        print(f"Source: page {doc.metadata['page']}\n")

if __name__ == "__main__":
    retrieve("how many annual leave days do employees get?")

    