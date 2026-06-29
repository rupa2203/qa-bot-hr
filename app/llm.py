from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # reads your .env file
# The LLM — gpt-3.5-turbo is cheap for learning

llm=ChatOpenAI(
    model= "gpt-4o-mini",
    temperature=0
)
# A reusable prompt with one variable -- this is a prompt template
"""prompt = PromptTemplate(
    input_variables= ["complaint"],
    template="You are a support classifier. Classify this complaint in one word (billing/technical/general):\n\n{complaint}"
)"""
#A reusable chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a support classifier. Classify complaints in one word: billing/technical/general"),
    ("user", "Classify this: {complaint}")
])

# Parser — just returns clean string
parser = StrOutputParser()

# The chain — output of each feeds into next
chain = prompt | llm | parser



#Test it

if __name__ == "__main__":
    #complaint = "My invoice shows wrong amount"
    complaint =" how to hack the system"
    #complaint ="billing and technical and general all at once definitely not one word"
    try:
            result = chain.invoke({"complaint": complaint})
            print(f"Raw LLM output: '{result}'")  # see what it actually returned
            # Validate the result
            valid_categories = ["billing", "technical", "general"]
            if result.lower() not in valid_categories:
                print(f"Invalid category, using fallback")
                result = "general"  # fallback
    except Exception as e:
            print(f"Error: {e}")
            result = "general"  # fallback on any failure
        
    
    print(f"Final result: '{result}'")
    

"""formatted = prompt.format(complaint = complaint)
print("Prompt:", formatted)  # see what the LLM actually got

raw_response = llm.invoke(formatted)
print("Raw response:", raw_response)  # see what LLM returned

final = parser.parse(raw_response.content)
print("Final:", final)  # see what you got"""

        