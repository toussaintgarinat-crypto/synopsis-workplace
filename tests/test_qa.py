from lib.qa import build_qa_prompt, answer_question

def test_prompt_contains_parts():
    p = build_qa_prompt("T", "R", "Q?")
    assert "T" in p and "R" in p and "Q?" in p

def test_answer_uses_llm():
    assert answer_question("T", "R", "Q?", lambda p: "ok") == "ok"
