from __future__ import annotations

import os

from google import genai


def main() -> None:
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    client = genai.Client()

    prompt = """
You are a software repair assistant.

Return only a valid unified diff.
Do not include markdown fences.
Do not include explanations.
Do not include prose.

Buggy file: stack.py

class BoundedMemoryStack:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items = []

    def push(self, item):
        self.items.append(item)

Failing test:

def test_capacity_limit():
    stack = BoundedMemoryStack(capacity=1)
    stack.push("a")
    with pytest.raises(MemoryError):
        stack.push("b")

Return a unified diff patch only.
"""

    interaction = client.interactions.create(
        model=model,
        input=prompt,
    )

    print(interaction.output_text)


if __name__ == "__main__":
    main()