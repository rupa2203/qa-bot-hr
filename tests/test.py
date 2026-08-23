from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(model_name =" gpt-3-5-turbo", temperature=0.0, max_tokens = 1000)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful support assisstant"),
    MessagesPlaceholder(variable_name = "history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    history = []
    
    while True:
        user_input = input("You:")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        response = chain.invoke({
            "input": user_input,
            "history": history
        })
        
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response))
        history = history [-4: ] # Keep only the last 4 messages in history
        
        print(f"Bot: {response}\n")

