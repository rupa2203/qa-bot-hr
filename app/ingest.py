import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
}

def load_and_chunks(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)
    return chunks

def load_and_chunk_file(file_path: str):
    """Loads and chunks a single supported file (.pdf, .txt), tagging every
    chunk with source_file metadata (used to delete-and-replace just this
    file's chunks during an incremental sync). Non-PDF files have no page
    metadata, so 'page' is defaulted to 0. Returns [] for unsupported
    extensions instead of raising, so a caller scanning a folder can skip
    them silently."""
    ext = os.path.splitext(file_path)[1].lower()
    loader_cls = LOADERS.get(ext)
    if loader_cls is None:
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    documents = loader_cls(file_path).load()
    chunks = splitter.split_documents(documents)
    filename = os.path.basename(file_path)
    for chunk in chunks:
        chunk.metadata.setdefault("page", 0)
        chunk.metadata["source_file"] = filename
    return chunks

def load_and_chunk_folder(folder_path: str):
    """Loads every supported file (.pdf, .txt) in folder_path and chunks them."""
    all_chunks = []
    for filename in sorted(os.listdir(folder_path)):
        all_chunks.extend(load_and_chunk_file(os.path.join(folder_path, filename)))
    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunks("documents/hr_policy.pdf")

    print(f"Total_chunks: {len(chunks)}")

    for i in range(len(chunks)):
        if i >= 5:  # limit output to first 5 chunks
            break
        print(f"Chunk {i+1}: {chunks[i].page_content}")
        print(f"Metadata: {chunks[i].metadata}")
