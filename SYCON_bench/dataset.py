from pathlib import Path

from inspect_ai.dataset import Dataset, MemoryDataset, Sample

DATA_DIR = Path(__file__).parent / "data"

def load_debate_dataset(data_dir: Path = DATA_DIR) -> Dataset:
    questions_path = data_dir / "questions.txt"
    arguments_path = data_dir / "arguments.txt"

    with open(questions_path) as f:
        questions = [line.strip() for line in f if line.strip()]

    with open(arguments_path) as f:
        arguments = [line.strip() for line in f if line.strip()]

    if len(questions) != len(arguments):
        raise ValueError(
            f"questions.txt has {len(questions)} lines but arguments.txt has "
            f"{len(arguments)} lines — they must match."
        )

    samples = [
        Sample(
            id=i + 1,
            input=question,
            target=argument,
            metadata={"argument": argument},
        )
        for i, (question, argument) in enumerate(zip(questions, arguments))
    ]

    return MemoryDataset(samples=samples)
