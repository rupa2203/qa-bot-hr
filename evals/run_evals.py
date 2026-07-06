import json
from app.rag import answer_question

with open("evals/test_dataset.json") as f:
    test_data = json.load(f)
    
results=[]
for test in test_data:
    result = answer_question(test['question'])
    result["expected_topic"] = test['expected_topic']
    results.append(result)
    
    print(f"Q:{test['question']}")
    print(f"A: {result['answer'][:100]}")
    print(f"In scope: {result['in_scope']}\n")
    
with open("evals/results.json", "w") as f:
    json.dump(results, f, indent=2)
    