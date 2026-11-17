Phase 3 of my RAG from scratch started with an uncomfortable discovery: my evaluation was too easy.

In phase 2 I built an exam with an answer key for my RAG. I expected part numbers and fault codes to be the weak spot of vector search. The first run gave 100% hit rate everywhere, codes included. 🤔

Not because the search was perfect, but because the exam was. 4 documents on 4 very different topics: finding the right page among 19 is easy. A perfect score tells you nothing, and it leaves no room to prove that the next change helps.

So before touching the search, I made the exam harder:
→ a second, older train fleet with the same kinds of facts but different numbers
→ a parts catalogue full of look-alikes: BRK-4471, BRK-4417, BRK-4710
→ fault codes like F-217 and F-271, which differ by one swapped digit
→ 15 new questions, 40 in total

Then the actual topic: hybrid search (diagram below).

Vector search finds meaning, so "bike" finds "bicycle", but to an embedding F-217 and F-271 look almost the same. Keyword search (BM25) finds exact words, so it nails F-271, but "bike" never matches "bicycle". Hybrid runs both and merges the two rankings with Reciprocal Rank Fusion: every chunk gets 1/(60 + position) from each list. Only the positions matter, which solves the problem of adding a BM25 score of 12.7 to a cosine similarity of 0.83.

The results, same 40 questions for every strategy:
[paste the comparison table or 3 numbers here: MRR vector X · keyword X · hybrid X, and MRR on exact codes]
[one honest sentence: did hybrid win? did the reranker earn its extra LLM call?]

Lesson of the week: before improving a system, check that your measurement can actually see the improvement.

Code (each phase in its own folder): https://github.com/luigus/rag-from-scratch

#RAG #GenAI #LLM #Python #Search
