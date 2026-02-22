from src.dataset import load_humaneval
from src.embeddings import build_vectorstore
from src.retriever import retrieve_similar_examples
from src.generator import build_llm, generate_code


def rag_code_pipeline(task_description: str) -> str:
    """
    Full RAG pipeline:
    1. Load HumanEval dataset
    2. Build / load ChromaDB vectorstore
    3. Retrieve similar examples
    4. Generate code with LLM
    """
    print("\n" + "=" * 65)
    print(f"Task: {task_description}")
    print("=" * 65 + "\n")

    # Step 1 — Load dataset
    examples = load_humaneval()

    # Step 2 — Build vectorstore
    vectorstore = build_vectorstore(examples)

    # Step 3 — Retrieve similar examples
    print("Retrieving similar examples from ChromaDB...")
    retrieved_docs = retrieve_similar_examples(vectorstore, task_description)

    # Step 4 — Generate code
    print("Generating code with LLM...\n")
    llm = build_llm()
    generated_code = generate_code(llm, task_description, retrieved_docs)

    print("=" * 65)
    print("Generated Code:")
    print("=" * 65)
    print(generated_code)

    return generated_code


if __name__ == "__main__":
    test_tasks = [
        "Write a function that checks if a list of numbers has any two numbers closer to each other than a given threshold.",
        "Write a function that returns the largest prime factor of a given number.",
        "Write a function that counts the number of vowels in a given string.",
    ]

    for task in test_tasks:
        rag_code_pipeline(task)
        print("\n")
