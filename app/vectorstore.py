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

        
def print_embeddings(query: str):
    vectorstore = load_vectorstore()
    
    # Get the raw embedding vector for the query
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    query_vector = embeddings.embed_query(query)
    
    print(f"Query: '{query}'")
    print(f"Vector dimensions: {len(query_vector)}")
    print(f"First 10 values: {query_vector[:10]}")
    print(f"Last 10 values: {query_vector[-10:]}")
    print()
    
    # Get stored chunk embeddings directly from ChromaDB
    collection = vectorstore._collection
    results = collection.get(include=["embeddings", "documents", "metadatas"])
    
    print(f"Total chunks stored: {len(results['documents'])}")
    for i, (doc, embedding, meta) in enumerate(zip(
        results['documents'][:3],       # first 3 chunks only
        results['embeddings'][:3],
        results['metadatas'][:3]
    )):
        print(f"\nChunk {i+1} (page {meta.get('page', '?')}):")
        print(f"  Text preview: {doc[:100]}")
        print(f"  Vector dims: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")       
        
if __name__ == "__main__":
    retrieve("how many annual leave days do employees get?")
    print("\n--- EMBEDDING INSPECTION ---\n")
    print_embeddings("how many annual leave days do employees get?")
    