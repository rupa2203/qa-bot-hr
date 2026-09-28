import hashlib
import json
import os
import threading
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from app.ingest import load_and_chunks, load_and_chunk_folder, load_and_chunk_file
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

MANIFEST_PATH = "ingest_manifest.json"
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

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

def build_vectorstore_from_folder(folder_path: str):
    """Same as build_vectorstore, but ingests every supported file (.pdf, .txt)
    found in folder_path instead of a single hardcoded PDF."""
    chunks = load_and_chunk_folder(folder_path)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB from {folder_path}")
    return vectorstore

def _file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def _load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_manifest(manifest: dict):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def sync_vectorstore_from_folder(folder_path: str):
    """Incrementally syncs chroma_db with folder_path instead of rebuilding it
    from scratch every time: a new file is embedded and appended, a modified
    file (its content hash changed since the last sync) has only its own
    chunks deleted and re-embedded, an unchanged file is skipped entirely (no
    embedding call, no cost), and a file removed from folder_path has its
    chunks deleted so it doesn't linger as stale/orphaned data. Tracked via a
    filename -> content-hash manifest (MANIFEST_PATH), persisted alongside
    chroma_db so repeat runs know what's already embedded."""
    manifest = _load_manifest()
    current_files = {
        filename: _file_hash(os.path.join(folder_path, filename))
        for filename in sorted(os.listdir(folder_path))
        if os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS
    }

    added = [f for f in current_files if f not in manifest]
    modified = [f for f in current_files if f in manifest and manifest[f] != current_files[f]]
    unchanged = [f for f in current_files if f in manifest and manifest[f] == current_files[f]]
    removed = [f for f in manifest if f not in current_files]

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

    for filename in removed:
        vectorstore._collection.delete(where={"source_file": filename})
        del manifest[filename]
        print(f"  - removed: {filename} (chunks deleted, file no longer in {folder_path})")

    for filename in modified:
        vectorstore._collection.delete(where={"source_file": filename})
        chunks = load_and_chunk_file(os.path.join(folder_path, filename))
        vectorstore.add_documents(chunks)
        manifest[filename] = current_files[filename]
        print(f"  ~ modified: {filename} ({len(chunks)} chunks re-embedded)")

    for filename in added:
        chunks = load_and_chunk_file(os.path.join(folder_path, filename))
        vectorstore.add_documents(chunks)
        manifest[filename] = current_files[filename]
        print(f"  + added: {filename} ({len(chunks)} chunks embedded)")

    for filename in unchanged:
        print(f"  = unchanged: {filename} (skipped, no re-embedding)")

    _save_manifest(manifest)
    print(
        f"\nSync complete: {len(added)} added, {len(modified)} modified, "
        f"{len(unchanged)} unchanged, {len(removed)} removed."
    )
    return vectorstore

_vectorstore_cache = None
_vectorstore_lock = threading.Lock()

def load_vectorstore():
    # Cached and lock-guarded. FastAPI runs each sync endpoint in a
    # threadpool, so several /hr/ask requests can call this concurrently.
    # A bare "if cache is None: build it" still lets multiple threads all
    # see None at once and race to build their own Chroma(persist_directory=
    # ...) client against the SAME on-disk store — chromadb keeps a
    # class-level registry of "systems" keyed by that path
    # (SharedSystemClient._identifier_to_system), and concurrent construction
    # against the same key corrupts that bookkeeping: one thread's client
    # fails tenant validation ("AttributeError: 'RustBindingsAPI' object has
    # no attribute 'bindings'" -> "Could not connect to tenant
    # default_tenant"), its cleanup path deregisters the shared system out
    # from under the others, and a third thread then hits
    # "KeyError: 'chroma_db'". The lock ensures only the first request
    # actually builds the client; every other thread just waits and reuses
    # the same instance once it exists.
    global _vectorstore_cache
    if _vectorstore_cache is None:
        with _vectorstore_lock:
            if _vectorstore_cache is None:
                embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
                _vectorstore_cache = Chroma(
                    persist_directory="chroma_db",
                    embedding_function=embeddings
                )
    return _vectorstore_cache

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
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        folder = sys.argv[2] if len(sys.argv) > 2 else "documents"
        sync_vectorstore_from_folder(folder)
    else:
        retrieve("how many annual leave days do employees get?")
        print("\n--- EMBEDDING INSPECTION ---\n")
        print_embeddings("how many annual leave days do employees get?")
