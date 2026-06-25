Academic RAG with Entropy-Regularized MADDPG
Built an advanced academic Retrieval-Augmented Generation (RAG) system for citation-aware question answering over research papers.
The system combines hybrid retrieval (dense embeddings + BM25 sparse search), role-based RAG agents (retriever, rewriter, grader, generator, verifier),
and a stage-constrained MADDPG-style multi-agent reinforcement learning controller to adaptively manage retrieval, evidence grading, answer generation,
and verification. Introduced entropy regularization to improve retrieval diversity and prevent policy collapse, leading to better evidence coverage
and source grounding. Achieved 93.3% accuracy on ARC Challenge, with improved ROUGE-L, source precision, and source recall on a custom academic
benchmark while maintaining strong answer quality.

Tech Stack: Python, Qdrant, BM25, BAAI/bge-m3, Groq/OpenAI APIs, Reinforcement Learning, MADDPG.
