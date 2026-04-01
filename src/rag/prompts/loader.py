from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    return path.read_text(encoding="utf-8")


def format_rag_user_prompt(*, context: str, question: str) -> str:
    template = load_prompt("rag_answer_user.txt")
    return template.format(context=context.strip(), question=question.strip())
