"""Lexical search engine utilizing Okapi BM25 for keyword-based evidence retrieval."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from backend.config import BM25_B, BM25_K1
from backend.database.repository import ChunkRepository
from backend.schemas.chunk import DocumentChunk


class LexicalSearchError(Exception):
    """Base exception for lexical search errors."""

    pass


class LexicalInputError(LexicalSearchError, ValueError):
    """Raised when query or input parameters for lexical search are invalid."""

    pass


@dataclass(frozen=True)
class LexicalSearchResult:
    """Container representing a ranked document chunk with its BM25 score."""

    chunk: DocumentChunk
    score: float


def tokenize_text(text: str) -> list[str]:
    """Tokenize text for BM25 lexical search with full Unicode alphanumeric support.

    Features:
        - Case-insensitive normalization using casefold().
        - Extracts alphanumeric tokens with full Unicode support (e.g. Greek letters, accented characters).
        - Handles punctuation and symbols consistently by splitting on non-alphanumeric boundaries.
        - Preserves numeric terms (e.g. '92', '50').
        - Treats hyphenated expressions consistently by extracting sub-terms.
        - Preserves stopwords (e.g. 'not', 'without', 'no') which are critical to research claims.
        - Does not apply stemming (deferred for future evaluation).

    Args:
        text: The raw input string to tokenize.

    Returns:
        A list of lowercase alphanumeric token strings.
    """
    if not text:
        return []
    return re.findall(r"[^\W_]+", text.casefold())


class LexicalSearchEngine:
    """Lexical retrieval engine based on Okapi BM25.

    Uses ChunkRepository to fetch candidate document chunks for selected paper IDs.
    Calculates BM25 keyword relevance scores with explicit parameters k1 and b.
    """

    def __init__(
        self,
        repository: ChunkRepository,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        self.repository = repository
        self.k1 = k1
        self.b = b

    def search(
        self,
        query: str,
        paper_ids: Sequence[str],
        top_k: int = 20,
    ) -> list[LexicalSearchResult]:
        """Perform lexical retrieval over document chunks belonging to paper_ids.

        Args:
            query: Keyword query or claim text string.
            paper_ids: Target paper identifiers to restrict retrieval to.
            top_k: Maximum number of ranked results to return (must be > 0).

        Returns:
            A list of LexicalSearchResult objects ordered by BM25 score descending,
            with deterministic tie-breaking on chunk_id ascending.
            Returns an empty list if paper_ids is empty, no chunks exist, or no passages
            contain any query token.

        Raises:
            LexicalInputError: If query is empty or whitespace-only.
            ValueError: If top_k <= 0.
        """
        if not isinstance(query, str) or not query.strip():
            raise LexicalInputError("Query cannot be empty or whitespace-only.")

        if isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        if not paper_ids:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        query_token_set = set(query_tokens)

        # Deduplicate paper IDs preserving order
        unique_paper_ids = list(dict.fromkeys(paper_ids))

        # Retrieve eligible chunks via repository
        chunks = self.repository.get_chunks_by_paper_ids(unique_paper_ids)
        if not chunks:
            return []

        # Tokenize corpus passages
        tokenized_corpus = [tokenize_text(c.text) for c in chunks]

        # Check if query tokens occur anywhere in the candidate corpus
        corpus_has_overlap = any(
            bool(query_token_set & set(doc_tokens)) for doc_tokens in tokenized_corpus
        )
        if not corpus_has_overlap:
            return []

        # Construct BM25 index over retrieved candidate chunks
        bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        scores = bm25.get_scores(query_tokens)

        # Filter out passages containing NONE of the query tokens
        ranked_candidates: list[tuple[float, str, DocumentChunk]] = []
        for idx, chunk in enumerate(chunks):
            doc_tokens = tokenized_corpus[idx]
            if not (query_token_set & set(doc_tokens)):
                # Must match at least one query token
                continue

            bm25_score = float(scores[idx])
            ranked_candidates.append((bm25_score, chunk.chunk_id, chunk))

        if not ranked_candidates:
            return []

        # Sort by BM25 score descending, breaking ties with chunk_id ascending
        ranked_candidates.sort(key=lambda item: (-item[0], item[1]))

        return [
            LexicalSearchResult(chunk=item[2], score=item[0])
            for item in ranked_candidates[:top_k]
        ]
