import asyncio

from rag_from_scratch.evaluation.dataset import EvalCase
from rag_from_scratch.evaluation.judge import LLMJudge, build_judge_prompt, parse_verdict
from tests.fakes import FakeChat

CASE = EvalCase(
    id="q1",
    question="How fast?",
    reference_answer="200 km/h",
    evidence=["200 km/h"],
    category="factual",
)


def test_parse_plain_json():
    v = parse_verdict('{"faithful": true, "correct": false, "reason": "wrong number"}')
    assert (v.faithful, v.correct, v.error) == (True, False, False)


def test_parse_json_inside_code_fence():
    raw = 'Here you go:\n```json\n{"faithful": true, "correct": true, "reason": "ok"}\n```'
    v = parse_verdict(raw)
    assert v.correct and not v.error


def test_unparseable_output_is_a_visible_failure():
    v = parse_verdict("I think the answer is fine.")
    assert v.error
    assert not v.faithful and not v.correct


def test_prompt_contains_everything_the_judge_needs():
    prompt = build_judge_prompt(CASE, "It goes 200 km/h", [])
    assert "How fast?" in prompt
    assert "200 km/h" in prompt
    assert "(nothing was retrieved)" in prompt


def test_judge_calls_the_model():
    chat = FakeChat('{"faithful": true, "correct": true, "reason": "matches"}')
    verdict = asyncio.run(LLMJudge(chat).evaluate(CASE, "200 km/h", []))
    assert verdict.correct
    assert len(chat.calls) == 1


def test_thinking_text_before_the_json_is_ignored():
    raw = '<think>The answer says {maybe} 200...</think>\n{"faithful": true, "correct": true, "reason": "ok"}'
    v = parse_verdict(raw)
    assert v.correct and not v.error
