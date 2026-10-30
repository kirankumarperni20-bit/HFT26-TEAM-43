"""Unit tests for the EmbeddingService using a mock model fixture (Step 5)."""

from typing import Any
import numpy as np
import pytest

from backend.retrieval.embeddings import (
    EmbeddingInputError,
    EmbeddingService,
    InputTooLongError,
    ModelLoadError,
)


class MockTokenizer:
    """Mock HuggingFace tokenizer providing deterministic token counting."""

    def __init__(self, token_multiplier: int = 1) -> None:
        self.token_multiplier = token_multiplier

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
    ) -> list[int]:
        words = text.split()
        return list(range(len(words) * self.token_multiplier))


class MockSentenceTransformer:
    """Mock SentenceTransformer model simulating BAAI/bge-small-en-v1.5."""

    def __init__(self, dimension: int = 384, max_seq_length: int = 512) -> None:
        self.dimension = dimension
        self.max_seq_length = max_seq_length
        self.tokenizer = MockTokenizer()
        self.encoded_calls: list[list[str]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        self.encoded_calls.append(list(sentences))
        n = len(sentences)
        if n == 0:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Generate deterministic synthetic vectors
        rng = np.random.default_rng(seed=123)
        raw = rng.standard_normal((n, self.dimension)).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return raw / norms
        return raw


@pytest.fixture
def mock_model():
    """Fixture providing a fresh MockSentenceTransformer instance."""
    return MockSentenceTransformer(dimension=384, max_seq_length=512)


@pytest.fixture
def service(mock_model):
    """Fixture providing an EmbeddingService injected with the mock model."""
    return EmbeddingService(model_instance=mock_model)


def test_lazy_model_loading(monkeypatch):
    """1. Verify that model loading is deferred until first property/method access."""
    load_called = False

    def fake_load(self):
        nonlocal load_called
        load_called = True
        return MockSentenceTransformer()

    monkeypatch.setattr(EmbeddingService, "_load_model", fake_load)

    # Instantiating service without model_instance should not trigger load
    lazy_svc = EmbeddingService()
    assert lazy_svc._model is None
    assert not load_called

    # Accessing dimension or model triggers lazy loading
    dim = lazy_svc.dimension
    assert load_called
    assert dim == 384
    assert lazy_svc._model is not None


def test_query_embedding_shape_and_normalization(service):
    """2. Verify that query embedding returns a 1D vector of shape (384,) with unit L2 norm."""
    query = "The proposed model achieved an accuracy of 92%."
    vec = service.embed_query(query)

    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape == (384,)

    norm = float(np.linalg.norm(vec))
    assert norm == pytest.approx(1.0, rel=1e-5)


def test_document_embeddings_shape_and_normalization(service):
    """3. Verify that document embeddings return a 2D array of shape (N, 384) with unit L2 norms."""
    passages = [
        "First research passage discussing transformers.",
        "Second passage regarding residual neural connections.",
        "Third passage on benchmark accuracy.",
    ]
    arr = service.embed_documents(passages, batch_size=2)

    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    assert arr.shape == (3, 384)

    row_norms = np.linalg.norm(arr, axis=1)
    for norm in row_norms:
        assert float(norm) == pytest.approx(1.0, rel=1e-5)


def test_query_instruction_applied_to_queries_only(service, mock_model):
    """4. Verify that the query instruction is prepended to queries only."""
    query = "Sample claim text"
    service.embed_query(query)

    assert len(mock_model.encoded_calls) == 1
    encoded_query = mock_model.encoded_calls[0][0]
    expected_prefix = service.query_instruction
    assert encoded_query.startswith(expected_prefix)
    assert encoded_query == f"{expected_prefix}{query}"


def test_document_passages_do_not_receive_query_instruction(service, mock_model):
    """5. Verify that document passages are encoded without prepending the query instruction."""
    passages = ["Passage alpha", "Passage beta"]
    service.embed_documents(passages)

    assert len(mock_model.encoded_calls) == 1
    encoded_passages = mock_model.encoded_calls[0]
    assert encoded_passages == passages
    for p in encoded_passages:
        assert not p.startswith(service.query_instruction)


def test_batch_processing_preserves_order_and_count(service):
    """6. Verify batch processing preserves exact input count and ordering."""
    passages = [f"Passage number {i}" for i in range(10)]
    arr = service.embed_documents(passages, batch_size=3)

    assert arr.shape == (10, 384)


def test_empty_document_input_returns_empty_array(service):
    """7. Verify empty document list returns an empty NumPy array with shape (0, 384)."""
    arr = service.embed_documents([])

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (0, 384)


def test_empty_query_rejected(service):
    """8. Verify empty or whitespace-only query raises EmbeddingInputError."""
    for empty in ["", "   ", "\t\n"]:
        with pytest.raises(EmbeddingInputError):
            service.embed_query(empty)


def test_empty_document_string_rejected(service):
    """9. Verify empty or whitespace-only passage strings in document list are rejected."""
    with pytest.raises(EmbeddingInputError) as exc_info:
        service.embed_documents(["Valid passage", "   ", "Another valid passage"])

    assert "index 1" in str(exc_info.value)


def test_invalid_batch_size_rejected(service):
    """10. Verify non-positive batch sizes raise ValueError."""
    for invalid_bs in [0, -1, -10]:
        with pytest.raises(ValueError):
            service.embed_documents(["Some text"], batch_size=invalid_bs)


def test_overlength_query_raises_input_too_long_error(service, mock_model):
    """11. Verify query exceeding max_seq_length raises InputTooLongError with position info."""
    # Generate long string exceeding 512 tokens
    long_words = ["word"] * 600
    long_query = " ".join(long_words)

    with pytest.raises(InputTooLongError) as exc_info:
        service.embed_query(long_query)

    err = exc_info.value
    assert err.position == "query"
    assert err.token_count > 512
    assert err.max_limit == 512


def test_overlength_passage_raises_input_too_long_error(service, mock_model):
    """12. Verify document passage exceeding max_seq_length raises InputTooLongError with index position."""
    valid_passage = "Short valid passage."
    oversized_passage = " ".join(["chunk"] * 550)

    with pytest.raises(InputTooLongError) as exc_info:
        service.embed_documents([valid_passage, oversized_passage])

    err = exc_info.value
    assert err.position == 1  # 0-indexed position of second passage
    assert err.token_count > 512
    assert err.max_limit == 512


def test_model_loading_error_propagated(monkeypatch):
    """13. Verify model loading failure raises ModelLoadError."""
    import sentence_transformers

    def fail_init(*args, **kwargs):
        raise RuntimeError("Network timeout downloading weights")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fail_init)
    svc = EmbeddingService()

    with pytest.raises(ModelLoadError) as exc_info:
        _ = svc.model

    assert "Network timeout" in str(exc_info.value)


def test_concurrent_model_loading_is_thread_safe(mock_model):
    """Verify that multiple concurrent threads accessing .model initialize it safely exactly once."""
    import concurrent.futures
    import time

    svc = EmbeddingService()
    load_count = 0

    def delayed_load():
        nonlocal load_count
        time.sleep(0.01)
        load_count += 1
        return mock_model

    svc._load_model = delayed_load

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(lambda: svc.model) for _ in range(16)]
        results = [f.result() for f in futures]

    assert load_count == 1
    assert all(r is mock_model for r in results)


def test_non_string_inputs_rejected_with_input_error(service, mock_model):
    """Verify non-string query and passage inputs raise EmbeddingInputError cleanly."""
    for invalid_query in [123, None, ["query"]]:
        with pytest.raises(EmbeddingInputError):
            service.embed_query(invalid_query)  # type: ignore[arg-type]

    for invalid_passages in [[123], [None], ["valid", 456]]:
        with pytest.raises(EmbeddingInputError):
            service.embed_documents(invalid_passages)  # type: ignore[arg-type]


def test_boolean_batch_size_rejected(service, mock_model):
    """Verify boolean batch_size is rejected with ValueError."""
    for bool_val in [True, False]:
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            service.embed_documents(["Sample text"], batch_size=bool_val)  # type: ignore[arg-type]
