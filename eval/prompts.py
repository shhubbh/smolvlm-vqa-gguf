"""Pinned eval prompt template. Imported identically by every variant runner."""

EVAL_SYSTEM_PROMPT = "You are a concise visual question answering assistant."

EVAL_USER_TEMPLATE = "{question}\n\nAnswer with the shortest correct response."


def render_user(question: str) -> str:
    return EVAL_USER_TEMPLATE.format(question=question)
