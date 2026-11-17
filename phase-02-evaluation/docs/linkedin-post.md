In phase 1 of my RAG from scratch, I ended with a confession: I had no idea if the answers were actually good.

Phase 2 fixes that. 👇

The usual way to test a chatbot is to ask it 3 or 4 questions and go "looks fine". Then you change the chunk size or the prompt, ask 3 or 4 other questions, and go "looks better?". That's not testing, that's vibes.

So I gave my RAG an exam, with an answer key (diagram below):

→ 25 questions about a fictional train operator, each with the expected answer and the exact sentence in the documents that proves it
→ 5 kinds of questions: direct ones, paraphrased ones, part numbers and fault codes, questions that need two facts, and questions with no answer at all (can it say "I don't know"?)

The part I like most: the search and the answer are graded separately.

For the search, no AI is needed. Did the retrieved pieces contain the proof sentence, and in which position? That gives Hit, Recall, Precision and MRR, and it costs nothing to run.

For the answer, a second LLM acts as the examiner and answers two yes/no questions. Is everything in the answer supported by what was retrieved (faithfulness)? Does it match the answer key (correctness)?

Why separately? When an answer is wrong, it's either because the search didn't find the right piece, or because the LLM didn't use it well. Knowing which one tells you what to fix.

First results (baseline):
[Hit@3: X · MRR: X · Faithfulness: X · Correctness: X]
[One sentence about the weakest category, e.g. "Part numbers like BRK-4471 are the weak spot, as expected for pure vector search."]

Every change from now on gets compared with these numbers. Phase 3 is hybrid search, and now I'll be able to show whether it really helps instead of just saying it.

Code: https://github.com/luigus/rag-from-scratch

How do you evaluate your RAG systems? Do you trust LLM-as-a-judge?

#RAG #GenAI #LLM #Python #Evaluation
