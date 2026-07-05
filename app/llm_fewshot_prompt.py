from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

from pydantic import BaseModel, Field 
from enum import Enum

load_dotenv()  # reads your .env file
# The LLM — gpt-3.5-turbo is cheap for learning

llm=ChatOpenAI(
    model= "gpt-3.5-turbo",
    #model = "gpt-4o-mini",
    temperature=0
)

#Define the schema
class CategoryEnum(str, Enum):
    billing="billing"
    technical="technical"
    general = "general"

class SeverityEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    
class ComplaintClassification(BaseModel):
    category:CategoryEnum
    severity: SeverityEnum
    action: str = Field(description = "Suggested action")
    

#parser in json format
parser = JsonOutputParser(pydantic_object=ComplaintClassification)

# A reusable prompt with one variable -- this is a prompt template
"""prompt = PromptTemplate(
    input_variables= ["complaint"],
    template="You are a support classifier. Classify this complaint in one word (billing/technical/general):\n\n{complaint}"
)"""
#A reusable chat prompt template, 
"""Ex: In production, you want to separate system instructions from user input. This is how real LLM systems work:
#System message — rules the AI must follow (stays the same)
#User message — the actual input (changes every time)

The LLM sees:
System: [classifier rules]
User: [the compl-turboint]
Why does this matter?
System messages have higher priority. The LLM respects them more reliably than rules buried in user text. Interview question: "How do you ensure the model follows instructions?" Answer: "System prompt."
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a support classifier. Classify into category, severity, and action. Severity must reflect urgency: security threats are always HIGH."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "Classify this complaint: {complaint}\n\n{format_instructions}")
])


prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# Parser — just returns clean string
#parser = StrOutputParser()

# The chain — output of each feeds into next
chain = prompt | llm | parser


#Test it

if __name__ == "__main__":
    #complaint = "My invoice shows wrong amount"
    #complaint =" how to hack the system"
    #complaint ="billing and technical and general all at once definitely not one word"
    #complaint = " I was charged twice"
    #complaint = "I forgot my password"
    complaint = " my system creashed"
    
    try:
            result = chain.invoke({"complaint": complaint})
            #print(f"Raw LLM output: '{result}'")  # see what it actually returned
            # Validate the result
            valid_categories = ["billing", "technical", "general"]
            if result["category"].lower() not in valid_categories:
                result["category"] = "general"  # fallback
            print(result)
    except Exception as e:
            print(f"Error: {e}")
            #result = "general"  # fallback on any failure
        
   # print(f"Final result: '{result}'")
    

"""formatted = prompt.format(complaint = complaint)
print("Prompt:", formatted)  # see what the LLM actually got

raw_response = llm.invoke(formatted)
print("Raw response:", raw_response)  # see what LLM returned

final = parser.parse(raw_response.content)
print("Final:", final)  # see what you got"""

        