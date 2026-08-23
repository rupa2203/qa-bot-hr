results = [
    {"question": "How many annual leave days do employees get?", "verdict": "PASS", "in_scope": True, "expected_topic": "leave"},
    {"question": "Can unused leave be carried forward?", "verdict": "PASS", "in_scope": True, "expected_topic": "leave"},
    {"question": "How many sick leave days are employees entitled to?", "verdict": "PASS", "in_scope": True, "expected_topic": "sick_leave"},
    {"question": "What is the company's stock price?", "verdict": "PASS", "in_scope": False, "expected_topic": "out_of_scope"},
    {"question": "How should be a mattress for neck pain?", "verdict": "FAIL", "in_scope": False, "expected_topic": "general"},
]

for result in results:
    print(f"Q: {result['question']}")
