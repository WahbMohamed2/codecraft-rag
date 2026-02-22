from datasets import load_dataset
from config import DATASET_NAME, DATASET_SPLIT


def load_humaneval() -> list:
    """
    Load HumanEval dataset and extract:
    - task_id
    - prompt
    - canonical_solution
    """
    print("Loading HumanEval dataset...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    examples = [
        {
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "canonical_solution": row["canonical_solution"],
        }
        for row in dataset
    ]

    print(f"Loaded {len(examples)} examples from HumanEval.\n")
    return examples
