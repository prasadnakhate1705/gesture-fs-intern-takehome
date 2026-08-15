"""
Extra tests for the Q&A pipeline.

test_pipeline.py builds a real FAISS index and loads flan-t5, so it is slow
and its assertions have to stay loose. These use fakes instead, which makes
them fast and lets them cover the cases that are hard to trigger against a
real model: empty input, an empty index, oversized context, a broken LLM.

Run: pytest tests/test_pipeline_extra.py -v
"""

import pytest

from src.pipeline import (
    MAX_CONTEXT_CHARS,
    NO_ANSWER,
    _build_context,
    _preview,
    ask_question,
    format_result,
    main,
)


class FakeDoc:
    """Stand-in for a LangChain Document."""

    def __init__(self, page_content: str):
        self.page_content = page_content


class FakeVectorStore:
    """Records its calls and returns canned chunks."""

    def __init__(self, chunks=None):
        self._chunks = chunks if chunks is not None else ["chunk one", "chunk two", "chunk three"]
        self.calls = []

    def similarity_search(self, query, k=3):
        self.calls.append((query, k))
        return [FakeDoc(text) for text in self._chunks[:k]]


class FakeLLM:
    """Captures the prompt it was given and returns a canned answer."""

    def __init__(self, answer="the answer"):
        self.answer = answer
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return [{"generated_text": self.answer}]


class TestAskQuestionContract:
    def test_returns_answer_and_sources(self):
        result = ask_question(FakeVectorStore(), FakeLLM("$5,500 per month"), "How much?")
        assert result["answer"] == "$5,500 per month"
        assert result["sources"] == ["chunk one", "chunk two", "chunk three"]

    def test_retrieves_three_chunks_by_default(self):
        store = FakeVectorStore()
        ask_question(store, FakeLLM(), "How much?")
        assert store.calls == [("How much?", 3)]

    def test_k_is_forwarded_to_the_vector_store(self):
        store = FakeVectorStore()
        ask_question(store, FakeLLM(), "How much?", k=5)
        assert store.calls[0][1] == 5

    def test_answer_is_stripped(self):
        result = ask_question(FakeVectorStore(), FakeLLM("  spaced out \n"), "How much?")
        assert result["answer"] == "spaced out"

    def test_sources_are_full_chunks_not_previews(self):
        """The CLI shortens chunks for display, the returned data should not."""
        chunk = "S" * 800
        result = ask_question(FakeVectorStore([chunk]), FakeLLM(), "How much?")
        assert result["sources"] == [chunk]


class TestPromptConstruction:
    def test_prompt_contains_context_and_question(self):
        llm = FakeLLM()
        ask_question(FakeVectorStore(), llm, "What is the Growth package?")
        prompt = llm.prompts[0]
        assert "chunk one" in prompt
        assert "chunk two" in prompt
        assert "What is the Growth package?" in prompt

    def test_question_survives_an_oversized_context(self):
        """An unbounded context would push the question past the 512 token
        cutoff, leaving the model with a prompt that has no question in it.
        """
        llm = FakeLLM()
        huge = ["X" * 5000, "Y" * 5000, "Z" * 5000]
        ask_question(FakeVectorStore(huge), llm, "Can I cancel my contract?")
        prompt = llm.prompts[0]
        assert "Can I cancel my contract?" in prompt
        assert len(prompt) < 2000


class TestAskQuestionEdgeCases:
    @pytest.mark.parametrize("question", ["", "   ", "\n\t"])
    def test_empty_question_raises(self, question):
        with pytest.raises(ValueError):
            ask_question(FakeVectorStore(), FakeLLM(), question)

    def test_empty_index_says_it_does_not_know(self):
        result = ask_question(FakeVectorStore([]), FakeLLM(), "How much?")
        assert result == {"answer": NO_ANSWER, "sources": []}

    def test_empty_index_does_not_call_the_llm(self):
        llm = FakeLLM()
        ask_question(FakeVectorStore([]), llm, "How much?")
        assert llm.prompts == []

    def test_blank_generation_falls_back_to_no_answer(self):
        result = ask_question(FakeVectorStore(), FakeLLM("   "), "How much?")
        assert result["answer"] == NO_ANSWER

    @pytest.mark.parametrize("bad", [[], [{}], [{"generated_text": 42}], None])
    def test_malformed_llm_result_raises_a_clear_error(self, bad):
        with pytest.raises(ValueError):
            ask_question(FakeVectorStore(), lambda prompt: bad, "How much?")


class TestBuildContext:
    def test_keeps_every_chunk_when_they_fit(self):
        assert _build_context(["a", "b", "c"]) == "a\n\nb\n\nc"

    def test_skips_blank_chunks(self):
        assert _build_context(["a", "   ", "c"]) == "a\n\nc"

    def test_respects_the_budget(self):
        context = _build_context(["A" * 400, "B" * 400, "C" * 400], max_chars=850)
        assert len(context) <= 850

    def test_drops_low_ranked_chunks_first(self):
        """Chunks come back ranked, so the least relevant is the one to lose."""
        context = _build_context(["A" * 400, "B" * 400, "C" * 400], max_chars=850)
        assert "A" * 400 in context
        assert "B" * 400 in context
        assert "C" not in context

    def test_truncates_a_single_oversized_chunk(self):
        """The closest match should be trimmed, not dropped."""
        context = _build_context(["A" * 5000], max_chars=100)
        assert context == "A" * 100

    def test_default_budget_leaves_room_for_the_question(self):
        assert MAX_CONTEXT_CHARS < 2048


class TestFormatting:
    def test_preview_collapses_whitespace(self):
        assert _preview("a\n\n  b\tc") == "a b c"

    def test_preview_truncates_long_text(self):
        preview = _preview("W" * 500, limit=50)
        assert preview.endswith("...")
        assert len(preview) == 53

    def test_preview_leaves_short_text_alone(self):
        assert _preview("short") == "short"

    def test_format_result_shows_numbered_sources_and_answer(self):
        output = format_result({"answer": "It costs $2,500.", "sources": ["one", "two"]})
        assert "1. one" in output
        assert "2. two" in output
        assert "It costs $2,500." in output

    def test_format_result_handles_no_sources(self):
        output = format_result({"answer": NO_ANSWER, "sources": []})
        assert "(none)" in output


@pytest.fixture
def stub_startup(monkeypatch):
    """Swap out the slow startup (index build and model load)."""
    monkeypatch.setattr("src.pipeline.build_knowledge_base", lambda data_dir: FakeVectorStore())
    monkeypatch.setattr("src.pipeline.get_llm", lambda: FakeLLM("It costs $2,500 per month."))


class TestCLI:
    def test_query_mode_answers_once_and_exits(self, stub_startup, tmp_path, capsys):
        code = main(["--query", "How much is Starter?", "--data-dir", str(tmp_path)])
        assert code == 0
        assert "It costs $2,500 per month." in capsys.readouterr().out

    def test_missing_data_dir_exits_with_an_error(self, tmp_path, capsys):
        missing = str(tmp_path / "nope")
        assert main(["--data-dir", missing]) == 1
        assert "not found" in capsys.readouterr().err

    def test_empty_query_is_rejected_before_startup(self, tmp_path):
        """No startup stub here, so reaching startup at all would be the bug."""
        assert main(["--query", "   ", "--data-dir", str(tmp_path)]) == 2

    def test_invalid_top_k_is_rejected_before_startup(self, tmp_path):
        """No startup stub here, so reaching startup at all would be the bug."""
        assert main(["--top-k", "0", "--data-dir", str(tmp_path)]) == 2

    def test_interactive_loop_exits_on_quit(self, stub_startup, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda prompt="": "quit")
        assert main(["--data-dir", str(tmp_path)]) == 0
        assert "Goodbye!" in capsys.readouterr().out

    def test_interactive_loop_answers_then_exits(self, stub_startup, tmp_path, monkeypatch, capsys):
        replies = iter(["What is SEO?", "quit"])

        monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))
        assert main(["--data-dir", str(tmp_path)]) == 0
        assert "It costs $2,500 per month." in capsys.readouterr().out

    def test_blank_input_is_skipped_not_answered(self, stub_startup, tmp_path, monkeypatch):
        replies = iter(["", "   ", "quit"])
        llm = FakeLLM()
        monkeypatch.setattr("src.pipeline.get_llm", lambda: llm)
        monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))

        assert main(["--data-dir", str(tmp_path)]) == 0
        assert llm.prompts == []

    def test_ctrl_d_exits_cleanly(self, stub_startup, tmp_path, monkeypatch):
        def raise_eof(prompt=""):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        assert main(["--data-dir", str(tmp_path)]) == 0

    def test_a_failed_question_does_not_end_the_session(self, stub_startup, tmp_path, monkeypatch, capsys):
        replies = iter(["boom", "quit"])

        def exploding_llm(prompt):
            raise RuntimeError("model exploded")

        monkeypatch.setattr("src.pipeline.get_llm", lambda: exploding_llm)
        monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))

        assert main(["--data-dir", str(tmp_path)]) == 0
        assert "model exploded" in capsys.readouterr().err
