from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.vectorstore import load_vectorstore

load_dotenv()  # Load environment variables from .env file

llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an HR policy assistant. Answer questions using only the context provided.
    If the answer is not in the context, say "I don't have that information in the HR policy."
    Always mention the page number your answer comes from."""),
    ("user","Context:\n {context}\n\n Question:{question}")
])


chain = prompt | llm | StrOutputParser()

def answer_question(query:str)->dict:
    vectorstore=load_vectorstore()
    docs = vectorstore.similarity_search(query, k=2)
    
    context = "\n\n".join([
        f"[Page {doc.metadata['page']}]:{doc.page_content}" 
        for doc in docs
    ])
    
    answer = chain.invoke({
        "context": context,
        "question": query
    })
    
    return {
        "question": query,
        "answer": answer,
        "sources": [doc.metadata['page']+1 for doc in docs]
    }

if __name__ == "__main__":
    result = answer_question("How many annual leave days do employees get?")
    print(result)