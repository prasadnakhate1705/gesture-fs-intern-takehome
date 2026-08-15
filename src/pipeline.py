"""
Document Q&A Pipeline.

Answers client questions from the agency docs: the question is embedded and
matched against the FAISS index, and the closest chunks are handed to
flan-t5 as context.

Usage:
    python -m src.pipeline
    python -m src.pipeline --query "How much is the Growth package?"
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable, Sequence

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.knowledge_base import build_knowledge_base

LLM = Callable[[str], Sequence[dict[str, Any]]]


# Provided: local LLM (no API key needed)
def get_llm():
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# Provided: prompt template
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


TOP_K = 3
MAX_CONTEXT_CHARS = 1400
SOURCE_PREVIEW_CHARS = 200
NO_ANSWER = "I don't have enough information to answer that."
EXIT_COMMANDS = frozenset({"quit", "exit", "q"})


def _build_context(sources: Sequence[str], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Join the retrieved chunks into a single context string.

    get_llm() cuts the prompt off at 512 tokens, and the question sits after
    the context in PROMPT_TEMPLATE. A context that runs too long therefore
    pushes the question out of the prompt completely, so it is kept under a
    character budget.

    Chunks arrive ordered by relevance, so the budget is filled from the top
    down and whole chunks are dropped rather than cut in half. The exception
    is a first chunk that is too big on its own, which gets truncated so the
    closest match is never lost.
    """
    kept: list[str] = []
    used = 0

    for chunk in sources:
        text = chunk.strip()
        if not text:
            continue

        cost = len(text) + (2 if kept else 0)
        if used + cost <= max_chars:
            kept.append(text)
            used += cost
        elif not kept:
            kept.append(text[:max_chars])
            break

    return "\n\n".join(kept)


def _extract_generated_text(result: Any) -> str:
    """Pull the generated string out of an LLM result, or raise ValueError."""
    try:
        text = result[0]["generated_text"]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"Unexpected LLM result shape: {result!r}") from exc

    if not isinstance(text, str):
        raise ValueError(f"Expected 'generated_text' to be a str, got {type(text).__name__}")

    return text.strip()


def ask_question(
    vector_store: Any,
    llm: LLM,
    question: str,
    k: int = TOP_K,
) -> dict[str, Any]:
    """Retrieve the k most relevant chunks and generate an answer from them.

    If nothing is retrieved there is no context to work from, so the LLM is
    skipped rather than being asked to answer on an empty context block.

    Args:
        vector_store: FAISS vector store from knowledge_base.py
        llm: Callable from get_llm()
        question: The user's question string
        k: How many chunks to retrieve

    Returns:
        dict with two keys:
            "answer"  -> str: the generated answer
            "sources" -> list[str]: the chunk texts that were retrieved

    Raises:
        ValueError: if the question is empty, or the LLM returns a bad shape.
    """
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    docs = vector_store.similarity_search(question, k=k)
    sources = [doc.page_content for doc in docs]

    if not sources:
        return {"answer": NO_ANSWER, "sources": []}

    prompt = PROMPT_TEMPLATE.format(context=_build_context(sources), question=question)
    answer = _extract_generated_text(llm(prompt)) or NO_ANSWER

    return {"answer": answer, "sources": sources}


def _preview(text: str, limit: int = SOURCE_PREVIEW_CHARS) -> str:
    """Collapse a chunk onto one line so it fits in the terminal."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "..."


def format_result(result: dict[str, Any]) -> str:
    """Render an ask_question() result for printing."""
    lines = ["", "📄 Sources:"]

    if result["sources"]:
        for i, source in enumerate(result["sources"], start=1):
            lines.append(f"  {i}. {_preview(source)}")
    else:
        lines.append("  (none)")

    lines += ["", f"💬 Answer: {result['answer']}", ""]
    return "\n".join(lines)


def _enable_unicode_output() -> None:
    """Allow the emoji in the output to print on a non-UTF-8 console.

    Windows terminals often default to cp1252, which raises
    UnicodeEncodeError partway through a session.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Ask questions about the agency's services, pricing, and process.",
    )
    parser.add_argument(
        "--query",
        metavar="TEXT",
        help="answer a single question and exit, instead of starting a session",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="directory of .txt documents to index (default: ./data)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        metavar="N",
        help=f"how many chunks to retrieve per question (default: {TOP_K})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Q&A CLI.

    Returns 0 on success, 1 if something failed at runtime, 2 on bad usage.
    """
    args = _parse_args(argv)
    _enable_unicode_output()

    # Checked before startup so a typo fails straight away instead of after
    # the index build and the model load.
    if args.top_k < 1:
        print("Error: --top-k must be at least 1.", file=sys.stderr)
        return 2

    if args.query is not None and not args.query.strip():
        print("Error: --query must not be empty.", file=sys.stderr)
        return 2

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    # Built once and reused, since both are slow to set up.
    try:
        vector_store = build_knowledge_base(data_dir)
        print("Loading language model (first run downloads ~1GB)...")
        llm = get_llm()
        print("  Ready!\n")
    except Exception as exc:
        print(f"Error: failed to start up: {exc}", file=sys.stderr)
        return 1

    if args.query is not None:
        try:
            print(format_result(ask_question(vector_store, llm, args.query, k=args.top_k)))
        except Exception as exc:
            print(f"Error: could not answer that question: {exc}", file=sys.stderr)
            return 1
        return 0

    print("Ask about our services, pricing, or process.")
    print(f"Type {'/'.join(sorted(EXIT_COMMANDS))} to exit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("Goodbye!")
            return 0

        # A failed question shouldn't drop the user out of the session.
        try:
            print(format_result(ask_question(vector_store, llm, question, k=args.top_k)))
        except Exception as exc:
            print(f"Sorry, something went wrong answering that: {exc}\n", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
