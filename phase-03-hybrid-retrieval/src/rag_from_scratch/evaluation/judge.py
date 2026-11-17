"""LLM-as-a-judge: a second LLM grades each answer.

Two questions, both answered with true/false (binary verdicts are more stable
than 1-10 scores):
- faithful: is every claim in the answer supported by the retrieved context?
  This catches hallucinations, even when the answer happens to be right.
- correct:  does the answer agree with the reference answer?

Known limits: the judge can be wrong too, and it tends to favour long answers.
That's why the reasons are saved in the report, to be reviewed by a human.
"""

import json
import re

from pydantic import BaseModel, ValidationError

from rag_from_scratch.domain.models import RetrievedChunk
from rag_from_scratch.domain.ports import ChatModel
from rag_from_scratch.evaluation.dataset import EvalCase

JUDGE_PROMPT_VERSION = "v1"

JUDGE_SYSTEM_PROMPT = """\
You are a strict evaluator of a question answering system.
You receive a question, a reference answer, the context the system retrieved,
and the system's answer. Return ONLY a JSON object with these keys:

- "faithful": true if every factual claim in the answer is supported by the
  context. An answer that says the information is not available is faithful.
- "correct": true if the answer agrees with the reference answer on the facts
  that matter. Extra correct details are fine. If the reference says the
  information is not available, the answer is correct only if it declines to
  answer or says it doesn't know, instead of inventing an answer.
- "reason": one short sentence explaining the verdict.
"""


class JudgeVerdict(BaseModel):
    faithful: bool
    correct: bool
    reason: str
    error: bool = False  # True when the judge's output could not be parsed


def build_judge_prompt(case: EvalCase, answer: str, retrieved: list[RetrievedChunk]) -> str:
    context = "\n\n".join(f"[{i}] {r.chunk.text}" for i, r in enumerate(retrieved, 1))
    return (
        f"Question: {case.question}\n\n"
        f"Reference answer: {case.reference_answer}\n\n"
        f"Retrieved context:\n{context or '(nothing was retrieved)'}\n\n"
        f"System answer: {answer}"
    )


def parse_verdict(raw: str) -> JudgeVerdict:
    """Extract the JSON object even if the model wraps it in text or code fences."""
    # local reasoning models may write their thinking first, inside <think> tags
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        if not match:
            raise ValueError("no JSON object found")
        return JudgeVerdict.model_validate(json.loads(match.group()))
    except (ValueError, ValidationError) as e:
        # a broken verdict counts as a failure and is visible in the report
        return JudgeVerdict(
            faithful=False, correct=False, reason=f"unparseable judge output: {e}", error=True
        )


class LLMJudge:
    def __init__(self, chat: ChatModel) -> None:
        self.chat = chat

    async def evaluate(
        self, case: EvalCase, answer: str, retrieved: list[RetrievedChunk]
    ) -> JudgeVerdict:
        raw = await self.chat.complete(
            JUDGE_SYSTEM_PROMPT, build_judge_prompt(case, answer, retrieved)
        )
        return parse_verdict(raw)
