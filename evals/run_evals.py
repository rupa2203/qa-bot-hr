import json
from app.rag import answer_question

with open("evals/test_dataset.json") as f:
    test_data = json.load(f)

results = []
passed = 0
failed = 0

for i, test in enumerate(test_data):
    result = answer_question(test['question'])
    result["expected_topic"] = test['expected_topic']

    if test['expected_topic'] in ["out_of_scope", "general"]:
        verdict = "PASS" if result['in_scope'] == False else "FAIL"
    else:
        verdict = "PASS" if result['in_scope'] == True else "FAIL"
        
    if len(result['sources']) == 4:
        confidence = "high"
    elif len(result['sources']) == 2:
        confidence = "low"
    else:
        confidence = "none"

    result["verdict"] = verdict
    result["confidence"] = confidence

    if verdict == "PASS":
        passed += 1
    else:
        failed += 1

    results.append(result)

    print(f"Q{i+1}: {test['question']}")
    print(f"A   : {result['answer'][:100]}")
    print(f"Verdict: {verdict} | in_scope={result['in_scope']} | confidence={result['confidence']}\n")

total = len(results)
print("=" * 50)
print("EVAL SUMMARY")
print("=" * 50)
print(f"Total  : {total}")
print(f"Passed : {passed}")
print(f"Failed : {failed}")
print(f"Score  : {round((passed/total)*100)}%")
print("=" * 50)

with open("evals/results.json", "w") as f:
    json.dump(results, f, indent=2)