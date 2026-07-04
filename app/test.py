from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Add this to see raw LLM output
llm_raw = ChatOpenAI(model="gpt-4o-mini", temperature=0)

complaint = "My invoice shows the wrong amount"
result_raw = llm_raw.invoke([
    {"role": "system", "content": "Return only valid JSON with category, severity, action"},
    {"role": "user", "content": f"Classify: {complaint}"}
])
print("Raw LLM output:")
print(result_raw.content)