#Problem: Your classifier works for single complaints. But real support needs conversation:
#Solution: Memory. Store past messages, feed them to the LLM so it has context.

#ConversationBufferWindowMemory
#Keeps last N messages in memory. pip install langchain-community
#Memory in newer LangChain versions moved away from ConversationBufferWindowMemory. 

"""
Write a multi-turn chat loop using ConversationBufferWindowMemory. Test with:

"My invoice is wrong"
"What's the due date?"
"Can you refund me?"

"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful HR support assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    history = []  # stores last N messages
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        response = chain.invoke({
            "input": user_input,
            "history": history
        })
        
        # Store last 4 messages only (k=2 exchanges)
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response))
        history = history[-4:]
        
        print(f"Bot: {response}\n")


"""
Fair. Let me explain the final working code cleanly.

What We Built (Modern Approach)
pythonhistory = []  # manual list — replaces ConversationBufferWindowMemory

response = chain.invoke({
    "input": user_input,
    "history": history  # past messages injected here
})

history.append(HumanMessage(content=user_input))
history.append(AIMessage(content=response))
history = history[-4:]  # keep last 2 exchanges only
Three moving parts:
1. history list — stores HumanMessage and AIMessage objects. This IS the memory.
2. MessagesPlaceholder — the slot in the prompt where history gets inserted. Without it, history list exists but LLM never sees it.
3. history[-4:] — window control. Keeps last 4 messages (2 human + 2 AI). Older ones dropped.

Question asked: Why we abandoned the deprecated approach:
RunnableWithMessageHistory did all 3 automatically — but it's deprecated in v1.3.10. The manual approach does the same thing explicitly. Actually better for interviews — you can explain every line.

One question: In the manual approach, what happens to memory when you type "quit"?
Answered: as the session ends , memory will be lostCorrect. Gone completely — the list exists only in RAM during the session.
"""
        
        
"""
Question: what does this part do 

```python
history = []  # stores last N messages
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        response = chain.invoke({
            "input": user_input,
            "history": history
        })
```

history = [] — empty list, starts fresh each session.
while True — keeps the chat loop running until user types exit/quit.
chain.invoke({"input": user_input, "history": history}) — sends two things to the chain:

input → current message (fills {input} in prompt)
history → past messages (fills MessagesPlaceholder in prompt)

First run: history is empty, LLM sees only current message.
Second run: history has 2 messages, LLM sees context.
Third run: history has 4 messages, LLM sees more context.

Question asked: okay then for every run , will the history loads , wil lthat not eat lot of tokens?Yes. That's the exact reason for history[-4:].

Without the window limit, history grows every turn — more tokens, more cost, slower responses.
With history[-4:], max 4 messages ever sent to LLM regardless of conversation length. Token cost stays flat.
Trade-off: LLM forgets older context. For a support classifier that's fine — you rarely need context from 10 messages ago.then how will i know for which type of problem statement i have to use what number?  where does this history saved in this context?Which k value to use:

Simple FAQ bot → k=2 (2 exchanges enough)
Support chat → k=4-6 (need more context)
Complex reasoning → k=10+ (but costs more)

Rule of thumb: use the minimum that gives coherent answers.

Where history is saved:
In RAM only. The history list lives in Python's memory — your computer's RAM. Not disk, not database.
pythonhistory = []  # RAM
history.append(HumanMessage(...))  # RAM
history.append(AIMessage(...))  # RAM
# program exits → RAM cleared → gone
"""
