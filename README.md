# RAG Code Generation System

A modular Retrieval-Augmented Generation (RAG) pipeline that takes a natural language programming task description, retrieves similar coding examples from the HumanEval dataset, and generates a complete Python function using an open-source LLM via OpenRouter.

---

## Project Structure

```
Task0/
├── main.py                  # Entry point — orchestrates the full pipeline
├── config.py                # Centralized configuration and environment variables
├── .env                     # API keys (not committed to version control)
├── requirements.txt         # Python dependencies
├── chroma_humaneval/        # Persistent ChromaDB storage (auto-generated)
└── src/
    ├── __init__.py
    ├── dataset.py           # Loads and parses the HumanEval dataset
    ├── embeddings.py        # Embeds prompts and manages ChromaDB
    ├── retriever.py         # Retrieves similar examples from ChromaDB
    └── generator.py         # Builds the LLM and generates code
```

---

## Features

- System prompting to define LLM role, tone, and behavior
- Conversational memory using LangChain's `RunnableWithMessageHistory`
- Persistent vector storage with ChromaDB
- RAG pipeline over the HumanEval dataset
- Modular, clean architecture with separation of concerns
- Free LLM inference via OpenRouter

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Task0
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root directory:

```bash
echo OPENROUTER_API_KEY=your-key-here > .env
```

Get your free API key from [https://openrouter.ai](https://openrouter.ai).

---

## Usage

Run the full RAG pipeline:

```bash
python main.py
```

The pipeline will:
1. Load the HumanEval dataset (164 examples)
2. Embed and store prompts in ChromaDB (skips if already stored)
3. Retrieve the top 3 most similar examples for each task
4. Generate a complete Python function using the LLM

---

## Configuration

All settings are managed in `config.py`:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `openrouter/free` | OpenRouter model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `CHROMA_PATH` | `./chroma_humaneval` | Local ChromaDB storage path |
| `TOP_K` | `3` | Number of similar examples to retrieve |
| `DATASET_SPLIT` | `test` | HumanEval dataset split |

---

## Requirements

```
langchain
langchain-core
langchain-openai
langchain-chroma
langchain-huggingface
openai
chromadb
sentence-transformers
datasets
python-dotenv
```

---

## Dataset

This project uses the [OpenAI HumanEval dataset](https://huggingface.co/datasets/openai/openai_humaneval), which contains 164 Python programming challenges. The following fields are extracted:

- `task_id` — unique identifier for each problem
- `prompt` — the function signature and docstring (used for embedding)
- `canonical_solution` — the reference solution (used as RAG context)

---

## Notes

- ChromaDB embeddings persist locally across sessions. Delete the `chroma_humaneval/` folder to force a fresh reload.
- The free OpenRouter tier routes to available providers automatically. If a model fails, update `LLM_MODEL` in `config.py`.
- Embedding is done locally using `sentence-transformers` — no additional API key required.
