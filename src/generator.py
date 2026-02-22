from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

FREE_MODELS = [
    "arcee-ai/trinity-large-preview:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "google/gemma-3-27b-it:free",
    "google/gemma-3-12b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen2.5-vl-72b-instruct:free",
]


def build_llm() -> ChatOpenAI:
    """Try free models in order until one responds successfully."""
    for model in FREE_MODELS:
        try:
            llm = ChatOpenAI(
                model=model, api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL
            )
            llm.invoke("hi")
            print(f"Using model: {model}\n")
            return llm
        except Exception as e:
            print(f"Skipping {model}: {str(e)[:60]}")
    raise RuntimeError("No available free models found on OpenRouter.")


def format_context(retrieved_docs: list) -> str:
    """Format retrieved docs into a readable context string."""
    context = ""
    for i, doc in enumerate(retrieved_docs):
        context += f"""
--- Example {i + 1} (task_id: {doc.metadata["task_id"]}) ---
Prompt:
{doc.page_content}

Solution:
{doc.metadata["canonical_solution"]}
"""
    return context


def generate_code(llm: ChatOpenAI, task_description: str, retrieved_docs: list) -> str:
    """Generate a Python function using the LLM guided by retrieved examples."""
    context = format_context(retrieved_docs)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert Python code generation assistant.
You will be given a programming task description and similar coding examples
retrieved from the HumanEval dataset as context.

Use the examples as reference to understand patterns and style,
then generate a complete, correct, and clean Python function for the given task.

Rules:
- Always return a complete Python function
- Include the function signature and docstring
- Follow Python best practices
- Do not include test cases in your output
""",
            ),
            (
                "human",
                """## Task Description:
{task_description}

## Retrieved Similar Examples (for context):
{context}

## Your Task:
Generate a complete Python function that solves the task described above.
""",
            ),
        ]
    )

    chain = prompt_template | llm
    response = chain.invoke({"task_description": task_description, "context": context})
    return response.content
