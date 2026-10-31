"""Embedding generation service utilizing BAAI/bge-small-en-v1.5."""

import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

from backend.config import (
    BGE_QUERY_INSTRUCTION,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    MAX_SEQUENCE_LENGTH,
)


class EmbeddingError(Exception):
    """Base exception for all embedding-related errors."""

    pass


class ModelLoadError(EmbeddingError):
    """Raised when the embedding model cannot be loaded."""

    pass


class ModelDimensionMismatchError(ModelLoadError):
    """Raised when the discovered model dimension does not match configured database dimension."""

    pass


class EmbeddingInputError(EmbeddingError, ValueError):
    """Raised when an input query or passage is empty, whitespace-only, or invalid."""

    pass


class InputTooLongError(EmbeddingError):
    """Raised when an input exceeds the model's maximum sequence token limit without silent truncation."""

    def __init__(
        self,
        position: int | str,
        token_count: int,
        max_limit: int,
        text_preview: str = "",
    ) -> None:
        self.position = position
        self.token_count = token_count
        self.max_limit = max_limit
        self.text_preview = text_preview
        message = (
            f"Input at position '{position}' exceeds model maximum token limit of {max_limit} tokens "
            f"(actual: {token_count} tokens). Ingestion chunking must be adjusted to prevent silent truncation."
        )
        super().__init__(message)


class EmbeddingService:
    """Service generating dense vector embeddings using BAAI/bge-small-en-v1.5.

    Features:
        - Lazy model loading (loaded upon first access, not module import).
        - Dynamic discovery of embedding dimensions and sequence length.
        - Strict token-length validation to prevent silent truncation.
        - Asymmetric encoding: prepends BGE query instruction to queries only.
        - L2 normalization on all output vectors.
        - Batch processing support.
    """

    DEFAULT_MODEL_NAME = EMBEDDING_MODEL_NAME
    DEFAULT_QUERY_INSTRUCTION = BGE_QUERY_INSTRUCTION

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        expected_dimension: int = EMBEDDING_DIMENSION,
        model_instance: Any = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.query_instruction = query_instruction
        self.expected_dimension = expected_dimension
        self._model = model_instance
        self._dimension: int | None = None
        self._max_seq_length: int | None = None
        self._lock = threading.Lock()

        if model_instance is not None:
            self._init_model_attributes(model_instance)

    @property
    def model(self) -> Any:
        """Lazily load and return the SentenceTransformer model instance in a thread-safe manner."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    loaded = self._load_model()
                    self._init_model_attributes(loaded)
                    self._model = loaded
        return self._model

    def _load_model(self) -> Any:
        """Instantiate the underlying SentenceTransformer model on the specified device."""
        try:
            from sentence_transformers import SentenceTransformer

            try:
                return SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
            except Exception:
                return SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load embedding model '{self.model_name}' on device '{self.device}': {exc}"
            ) from exc

    def _init_model_attributes(self, model: Any) -> None:
        """Discover model properties (dimension and max sequence length) dynamically and verify compatibility."""
        if hasattr(model, "get_embedding_dimension"):
            self._dimension = int(model.get_embedding_dimension())
        elif hasattr(model, "get_sentence_embedding_dimension"):
            self._dimension = int(model.get_sentence_embedding_dimension())
        elif hasattr(model, "dimension"):
            self._dimension = int(model.dimension)
        else:
            self._dimension = self.expected_dimension

        # Enforce dimension compatibility against configured database vector size
        if self._dimension != self.expected_dimension:
            raise ModelDimensionMismatchError(
                f"Discovered model embedding dimension {self._dimension} does not match "
                f"expected database dimension {self.expected_dimension}."
            )

        if hasattr(model, "max_seq_length") and model.max_seq_length is not None:
            self._max_seq_length = int(model.max_seq_length)
        else:
            self._max_seq_length = MAX_SEQUENCE_LENGTH

    @property
    def dimension(self) -> int:
        """Return the discovered embedding dimensionality (e.g. 384)."""
        if self._dimension is None:
            _ = self.model  # Trigger lazy load
        return self._dimension or 384

    @property
    def max_seq_length(self) -> int:
        """Return the maximum allowed token sequence length (e.g. 512)."""
        if self._max_seq_length is None:
            _ = self.model  # Trigger lazy load
        return self._max_seq_length or 512

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text using the model's tokenizer without truncation."""
        model = self.model
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            tokens = tokenizer.encode(text, add_special_tokens=True, truncation=False)
            return len(tokens)
        # Fallback approximation for lightweight mocks that don't supply a tokenizer
        return len(text.split())

    def _validate_length(self, text: str, position: int | str) -> None:
        """Ensure the input text does not exceed the model's maximum sequence length."""
        token_count = self.count_tokens(text)
        if token_count > self.max_seq_length:
            raise InputTooLongError(
                position=position,
                token_count=token_count,
                max_limit=self.max_seq_length,
                text_preview=text[:80],
            )

    def embed_query(self, query: str) -> np.ndarray:
        """Generate a normalized 1D embedding for a single query or atomic claim.

        Applies the BGE query instruction to the model input only. Does not alter the input query string.

        Args:
            query: The query or claim text string.

        Returns:
            A 1D NumPy array of shape (dimension,) with unit L2 norm.

        Raises:
            EmbeddingInputError: If query is empty or whitespace-only.
            InputTooLongError: If the formatted query exceeds max_seq_length.
            EmbeddingError: If model inference fails.
        """
        if not isinstance(query, str) or not query.strip():
            raise EmbeddingInputError("Query cannot be empty or whitespace-only.")

        formatted_query = f"{self.query_instruction}{query}"
        self._validate_length(formatted_query, position="query")

        try:
            embeddings = self.model.encode(
                [formatted_query],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise EmbeddingError(f"Inference failed during query embedding: {exc}") from exc

        vec = np.asarray(embeddings[0], dtype=np.float32)
        return vec

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Generate normalized 2D embeddings for a sequence of document passages in batches.

        Does NOT prepend query instructions to document passages.

        Args:
            texts: Sequence of raw document passage strings.
            batch_size: Batch size for encoding (must be > 0).

        Returns:
            A 2D NumPy array of shape (len(texts), dimension) with unit L2 norm.
            Returns shape (0, dimension) if texts is empty.

        Raises:
            ValueError: If batch_size <= 0.
            EmbeddingInputError: If any passage is empty or whitespace-only.
            InputTooLongError: If any passage exceeds max_seq_length.
            EmbeddingError: If model inference fails.
        """
        if isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

        if len(texts) == 0:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Validate each passage for content and length
        for idx, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingInputError(
                    f"Document passage at index {idx} cannot be empty or whitespace-only."
                )
            self._validate_length(text, position=idx)

        try:
            embeddings = self.model.encode(
                list(texts),
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise EmbeddingError(f"Inference failed during document embedding: {exc}") from exc

        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
