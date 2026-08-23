from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunks(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    
    chunks = splitter.split_documents(documents)
    return chunks
    

if __name__ == "__main__":
    chunks = load_and_chunks("documents/hr_policy.pdf")
    
    print(f"Total_chunks: {len(chunks)}")
    
    for i in range(len(chunks)):
        if i >= 5:  # limit output to first 5 chunks
            break
        print(f"Chunk {i+1}: {chunks[i].page_content}")
        print(f"Metadata: {chunks[i].metadata}")
                
        
    
    