from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import EMBEDDING_MODEL, CHROMA_PATH, CHROMA_COLLECTION


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the sentence-transformer embedding model."""
    print("Loading embedding model...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore(examples: list) -> Chroma:
    """
    Build or load ChromaDB vectorstore from HumanEval examples.
    Skips embedding if data already exists.
    """
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

    existing = vectorstore.get()
    if len(existing["ids"]) > 0:
        print(
            f"ChromaDB already has {len(existing['ids'])} embeddings. Skipping reload.\n"
        )
        return vectorstore

    print("Embedding and storing HumanEval prompts into ChromaDB...")
    docs = [
        Document(
            page_content=ex["prompt"],
            metadata={
                "task_id": ex["task_id"],
                "canonical_solution": ex["canonical_solution"],
            },
        )
        for ex in examples
    ]

    vectorstore.add_documents(docs)
    print(f"Stored {len(docs)} embeddings in ChromaDB at '{CHROMA_PATH}'\n")
    return vectorstore
