# Project Summary

Problem: Build a chatbot agentic system that the agent can answer user questions
based on the provided documents. There are mainly 2 pages:

1. Upload Page: A dedicated page to let user upload files in various formats:

- Documents (PDF/Doc): Manuals and guides (with screenshots and diagrams)
- Spreadsheets (Excel/CSV): Data sheets and term definitions (potentially
  thousands of rows)

2. Chat Page: A chat interface where user can ask questions in English or
   Japanese. The agent should be able to:

- Understand and process user queries in both English & Japanese
- Retrieve and present relevant information from the provided documents
- Generate text-based responses and extract images from documents
- When unable to answer a question, call tool `transferToSupport` instead of
  responding directly

## Implementation notes:

- Use only ONE OpenAI-compatible model as the main LLM for the agent
  (`qwen3-vl:8b-instruct` from Ollama). No other LLMs are allowed.

- Use only ONE OpenAI-compatible embedding model for vectorization
  (`embeddinggemma:300m` from Ollama). No other embedding models are allowed.

- Before retrieving information from documents, make sure the user query is
  rewritten into Japanese and English queries so that the retrieval can be done
  in both languages to maximize the chances of finding relevant information.

- For informal chat (greetings, small talk, etc.), the agent can respond
  directly without retrieving documents. For questions required knowledge from
  documents (e.g., specific data points, definitions, explanations, tutorials),
  the agent should always try to retrieve relevant documents first before
  responding. NEVER hallucinate answers.

- Agent does not always retrieve documents for every query. Treat RAG as a tool
  that the agent can decide to use or not based on the situation. There are ONLY
  TWO tools: `retrieveDocument` & `transferToSupport`.

- Agent should be able to decide by itself when to use the tool
  `transferToSupport` based on the `retrieveDocument` results and the user
  query. For example, if the retrieved documents are not relevant to the user
  query, call `transferToSupport` directly instead of responding with the
  retrieved documents.

- When retrieving documents, implement caching in redis to optimize for cost and
  performance. Specifically, each rewritten query should be cached with its
  retrieval results (set of document IDs) for a certain period of time (e.g., 1
  hour) to avoid redundant retrievals for the same query.

- Documents immediately after upload should be converted to text using
  `markitdown` and chunked into smaller pieces using `RecursiveChunker` from
  `chonkie` for vectorization and storage in the vector database.

- NEVER write any raw SQL in python code. Use `sqlmodel` as the ORM for all
  database operations.

Preferred Technologies/Frameworks:

- Agentic framework: [Pydantic AI](https://ai.pydantic.dev/)
- Document processing: [markitdown](https://github.com/microsoft/markitdown)
- Chunking: [Chonkie](https://chonkie.ai/)
- File storage: [Minio](https://min.io/)
- Caching: [Redis](https://redis.io/)
- ORM: [SQLModel](https://sqlmodel.tiangolo.com/)
- Chat database: [PostgreSQL](https://postgresql.org/)
- Vector Database: [Milvus](https://milvus.io/)
- Backend: [FastAPI](https://fastapi.tiangolo.com/)
- Frontend: [NextJS 16 + Tailwind CSS](see
  https://ai.pydantic.dev/examples/chat-app/#example-code)
- LLM: [Ollama](https://ollama.com/) (OPENAI_BASE_URL=http://localhost:11434)

Frontend notes:

- We should have 2 dedicated pages: / is for chat, /upload is for upload and
  files management (user can upload/delete files)
- We should have a toggle dark mode button, keep the foreground/background
  colors simple (black/white)
- Chat history should be persist on the sidebar, user can browse between
  conversation, conversation should be named as the first user message,
  conversation can be rename or delete
- Make the UI to look just like chatgpt.com (sidebar and main conversation area)
- Make LLM response streaming

---

# Implementation Plan

`/Users/o/micromamba/bin/python` is the Python interpreter to be used for this
project. Use this interpreter for all Python scripts and installations. When
testing the implementation, if you find any missing libraries, just run
`pip install <library>` to install them. No need to confirm with me.

YOU MUST search for latest documentation of each library to find the best way to
use it before implementing any functionality. Follow the official documentation
as much as possible.

For every functionality, write a simple test script to make sure it works before
integrating it into the main backend.

Before starting implementation, write docker-compose.yml (load credentials from
.env) and .env (with default values) that can run all the services needed for
this project, including Minio, Redis, PostgreSQL, and Milvus, then test the
connections from a simple Python script using appropriate clients to make sure
everything is working. The .env file will be used by both docker and the
backend. Python and Ollama is not needed in docker-compose. Also include testing
2 ollama models (`qwen3-vl:8b-instruct` and `embeddinggemma:300m`) in the test
script to ensure they can be accessed correctly.

Once the docker-compose is ready, let me double-check all services can run
together smoothly. Wait for my confirmation before proceeding to implement the
actual features.

Make sure every function has clear argument types and return types. After
finishing, use python typechecking tool: mypy and pyright to make sure all
classes, functions and variables has proper member access.
