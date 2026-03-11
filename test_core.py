"""Automated test for core RAG logic (no API key required)."""
import sys
sys.path.insert(0, ".")

from main import KnowledgeBase, QueryProcessor, Retriever, PromptBuilder

passed = 0
failed = 0

def test(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name}")
        failed += 1

print("=" * 50)
print("TEST 1: Knowledge Base Loading")
print("=" * 50)
kb = KnowledgeBase()
kb.load()
test("Loaded entries > 0", kb.get_sentence_count() > 0)
test("Loaded at least 50 entries", kb.get_sentence_count() >= 50)
test("Raw text is not empty", len(kb.raw_text) > 0)
print(f"  (Loaded {kb.get_sentence_count()} entries)")
print()

print("=" * 50)
print("TEST 2: Query Processing")
print("=" * 50)
qp = QueryProcessor()
q1 = qp.process("Plan a 2-day trip in Kashmir")
test("Original preserved", q1["original"] == "Plan a 2-day trip in Kashmir")
test("Cleaned is lowercase", q1["cleaned"] == q1["cleaned"].lower())
test("Keywords extracted", len(q1["keywords"]) > 0)
test("Stop words removed", "a" not in q1["keywords"] and "in" not in q1["keywords"])
test("Kashmir in keywords", "kashmir" in q1["keywords"])
print(f"  Keywords: {q1['keywords']}")
print()

print("=" * 50)
print("TEST 3: Retrieval - Trip Query")
print("=" * 50)
r = Retriever(kb)
results1 = r.retrieve(q1)
test("Retrieved > 0 results", len(results1) > 0)
test("Results have scores", all(score > 0 for _, score in results1))
test("Results sorted by score", all(results1[i][1] >= results1[i+1][1] for i in range(len(results1)-1)))
print(f"  Retrieved {len(results1)} results")
for i, (sent, score) in enumerate(results1[:3], 1):
    print(f"    {i}. [{score:.2f}] {sent[:70]}...")
print()

print("=" * 50)
print("TEST 4: Retrieval - Hidden Places")
print("=" * 50)
q2 = qp.process("hidden places in Kashmir")
results2 = r.retrieve(q2)
test("Retrieved > 0 results", len(results2) > 0)
has_hidden = any("hidden" in s.lower() or "offbeat" in s.lower() or "remote" in s.lower() for s, _ in results2)
test("Contains hidden/offbeat/remote results", has_hidden)
print(f"  Retrieved {len(results2)} results")
print()

print("=" * 50)
print("TEST 5: Retrieval - Food Query")
print("=" * 50)
q3 = qp.process("Kashmiri food and cuisine")
results3 = r.retrieve(q3)
test("Retrieved > 0 results", len(results3) > 0)
has_food = any("food" in s.lower() or "cuisine" in s.lower() or "wazwan" in s.lower() or "rogan" in s.lower() for s, _ in results3)
test("Contains food-related results", has_food)
print(f"  Retrieved {len(results3)} results")
print()

print("=" * 50)
print("TEST 6: Retrieval - Gulmarg Skiing")
print("=" * 50)
q4 = qp.process("skiing in Gulmarg")
results4 = r.retrieve(q4)
test("Retrieved > 0 results", len(results4) > 0)
has_gulmarg = any("gulmarg" in s.lower() for s, _ in results4)
test("Contains Gulmarg results", has_gulmarg)
print(f"  Retrieved {len(results4)} results")
print()

print("=" * 50)
print("TEST 7: Prompt Construction")
print("=" * 50)
pb = PromptBuilder()
prompt = pb.build("Plan a trip", results1)
test("Prompt contains context", "TRAVEL KNOWLEDGE BASE" in prompt)
test("Prompt contains query", "Plan a trip" in prompt)
test("Prompt is non-empty", len(prompt) > 100)
print()

print("=" * 50)
print("TEST 8: Flask App Import")
print("=" * 50)
try:
    from flask import Flask
    test("Flask importable", True)
except ImportError:
    test("Flask importable", False)

try:
    import google.generativeai
    test("google-generativeai importable", True)
except ImportError:
    test("google-generativeai importable", False)
print()

print("=" * 50)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
print("=" * 50)

if failed == 0:
    print("ALL TESTS PASSED!")
else:
    print(f"WARNING: {failed} test(s) failed")
    sys.exit(1)
