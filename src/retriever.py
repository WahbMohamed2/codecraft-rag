from langchain_chroma import Chroma
from config import TOP_K


def retrieve_similar_examples(vectorstore: Chroma, query: str, k: int = TOP_K) -> list:
    """
    Retrieve top-k most similar examples from ChromaDB
    based on the input query (task description).
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )

    results = retriever.invoke(query)

    print(f"Retrieved {len(results)} similar examples:\n")
    for doc in results:
        print(f"  • {doc.metadata['task_id']}: {doc.page_content[:60].strip()}...")
    print()

    return results
