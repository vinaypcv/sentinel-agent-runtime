from __future__ import annotations

import os
import sys

from google import genai


def main() -> int:
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
        print('Run: $env:GEMINI_API_KEY="your_real_gemini_key"')
        return 1

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

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
    except Exception as exc:
        print(f"Gemini request failed: {type(exc).__name__}: {exc}")
        return 1

    text = getattr(response, "text", None)
    if not text:
        print("Gemini returned an empty response.")
        return 1

    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())