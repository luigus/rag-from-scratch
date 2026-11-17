I've built a few RAG chatbots at work, but always with frameworks doing a lot of the heavy lifting. So I decided to build one from scratch, in public, and improve it step by step.

Here's phase 1. 👇

The idea of RAG is pretty simple: instead of hoping the LLM "knows" the answer, you give it the right pieces of your own documents and ask it to answer based only on that.

In practice it's two flows (diagram below):
→ you upload a document, it gets split into chunks, each chunk becomes a vector and goes into OpenSearch
→ you ask a question, it becomes a vector too, the closest chunks are found and sent to the LLM, and you get an answer with the sources

A few things I did on purpose:

No LangChain this time. Just Python, FastAPI and Pydantic. More code to write, but I can see and test every step.

The core logic doesn't know it's talking to OpenAI or OpenSearch. In the tests I swap them for fake in-memory versions, so the tests run in milliseconds, cost nothing and give the same result every time.

If the search finds nothing relevant, the LLM isn't even called. Better to say "I don't know" than to let the model make something up.

The big problem with this version? I have no idea if the answers are actually good. 😅 That's what phase 2 is about: building a small evaluation dataset and measuring it with real numbers, so every change after that is a measured improvement and not just a feeling.

Each phase is its own folder in the repo, so you can compare one with the next: https://github.com/luigus/rag-from-scratch

If you've built RAG in production, I'd love to hear what you'd do differently.

#RAG #GenAI #Python #LLM
