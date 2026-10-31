"""
retrieval_service.py — Evidence retrieval adapter integrating the retrieval team's engine.

Integrates:
- EvidenceRetriever (database-backed hybrid search via vector and Okapi BM25 with RRF)
- Claim schema from backend.schemas.claim
- EvidenceItem and ClaimEvidenceResponse schemas from backend.schemas.evidence
- Preserves the existing EvidenceRetrievalService.retrieve() contract
- Enforces strict paper provenance filtering and safe fallback isolation
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

try:
    from backend.llm.schemas.claim import AtomicClaim
    from backend.llm.schemas.verification import RetrievedEvidence
except ModuleNotFoundError:
    from llm.schemas.claim import AtomicClaim
    from llm.schemas.verification import RetrievedEvidence

from backend.retrieval.retriever import EvidenceRetriever, EvidenceRetrieverError
from backend.schemas.claim import Claim as RetrievalClaim
from backend.schemas.evidence import ClaimEvidenceResponse, EvidenceItem

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Standard English stopwords for lexical matching in local test fallback
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at",
    "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "this",
    "that", "these", "those", "it", "its", "we", "our", "us", "they", "their",
}


def tokenize_content_words(text: str) -> list[str]:
    """Extract content words (lowercase, alphanumeric, non-stopword, stripping punctuation)."""
    words = re.findall(r"[A-Za-z0-9]+(?:\.[0-9]+)?", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def resolve_target_paper_ids(
    involved_papers: list[str] | None,
    allowed_paper_ids: list[str] | None,
) -> list[str]:
    """Resolve and filter permitted paper IDs according to strict provenance constraints.

    Rules:
    - Use claim.involved_papers when available.
    - Respect allowed_paper_ids (intersection).
    - If involved_papers is empty but allowed_paper_ids is supplied, use the allowed IDs.
    - Never search papers outside the permitted paper IDs.
    - If no valid paper IDs remain, return an empty list rather than searching the entire database.
    """
    clean_involved = [p.strip() for p in (involved_papers or []) if p and p.strip()]
    clean_allowed = [p.strip() for p in (allowed_paper_ids or []) if p and p.strip()]

    if clean_involved:
        if clean_allowed:
            allowed_set = set(clean_allowed)
            targets = [p for p in clean_involved if p in allowed_set]
        else:
            targets = clean_involved
    elif clean_allowed:
        targets = clean_allowed
    else:
        return []

    return list(dict.fromkeys(targets))


def convert_atomic_claim_to_retrieval_claim(
    claim: AtomicClaim,
    target_paper_ids: list[str],
) -> RetrievalClaim:
    """Convert an AtomicClaim into the retrieval team's Claim schema.

    Raises:
        ValueError: If target_paper_ids is empty.
    """
    if not target_paper_ids:
        raise ValueError("Cannot construct retrieval Claim without target paper_ids.")
    return RetrievalClaim(
        claim_id=claim.claim_id,
        claim_text=claim.claim_text,
        paper_ids=target_paper_ids,
    )


def convert_evidence_item_to_retrieved_evidence(
    item: EvidenceItem,
) -> RetrievedEvidence:
    """Convert a retrieval team EvidenceItem into our RetrievedEvidence schema.

    Maps all available fields:
    - evidence_id -> evidence_id
    - paper_id -> paper_id
    - chunk_id -> chunk_id
    - page_number -> page
    - text -> text
    - retrieval_score -> retrieval_score
    - retrieval_method -> retrieval_method
    - section (preserved without alteration if present) -> section
    """
    return RetrievedEvidence(
        evidence_id=item.evidence_id,
        paper_id=item.paper_id,
        chunk_id=item.chunk_id,
        text=item.text,
        page=item.page_number,
        section=getattr(item, "section", None),
        retrieval_score=float(item.retrieval_score),
        retrieval_method=item.retrieval_method,
    )


class EvidenceRetrievalService:
    """Retrieves candidate evidence passages using EvidenceRetriever adapter or local corpus."""

    def __init__(
        self,
        retriever: EvidenceRetriever | None = None,
        corpus: list[RetrievedEvidence] | None = None,
        include_synthetic_fallback: bool = False,
    ) -> None:
        """Initialize EvidenceRetrievalService.

        Args:
            retriever: Optional EvidenceRetriever instance for database-backed hybrid retrieval.
            corpus: Optional preloaded list of RetrievedEvidence for test fixtures.
            include_synthetic_fallback: Flag whether to include synthetic test data (ConvS2S).
                Default is False to prevent synthetic evidence from entering production retrieval.
        """
        self._retriever = retriever
        self._corpus = corpus
        self._include_synthetic_fallback = include_synthetic_fallback

    @property
    def is_database_backed(self) -> bool:
        """Return True if an EvidenceRetriever is configured and active for database retrieval."""
        return self._retriever is not None

    def retrieve(
        self,
        claim: AtomicClaim,
        allowed_paper_ids: list[str] | None = None,
        max_candidates: int = 5,
        candidate_pool: list[RetrievedEvidence] | None = None,
    ) -> list[RetrievedEvidence]:
        """Retrieve candidate evidence passages for an atomic claim.

        Follows strict paper ID provenance filtering:
        1. Uses claim.involved_papers when available.
        2. Intersects with allowed_paper_ids.
        3. If involved_papers is empty, uses allowed_paper_ids.
        4. If no valid paper IDs remain, returns [] immediately without querying database.
        5. If candidate_pool is provided (isolated tests), filters and ranks candidate_pool.
        6. If self._retriever is configured, calls retrieve_evidence() and converts EvidenceItem -> RetrievedEvidence.
           Database errors are raised explicitly (no silent synthetic fallback).
        7. Otherwise, falls back to local corpus matching (excluding synthetic data in production).
        """
        target_papers = resolve_target_paper_ids(
            involved_papers=claim.involved_papers,
            allowed_paper_ids=allowed_paper_ids,
        )

        if not target_papers:
            logger.debug(
                "Claim %s has no valid permitted paper IDs remaining. Returning empty evidence list.",
                claim.claim_id,
            )
            return []

        # Candidate pool override (preserves isolated unit test behavior)
        if candidate_pool is not None:
            return self._filter_and_rank_pool(
                pool=candidate_pool,
                claim=claim,
                target_papers=set(target_papers),
                max_candidates=max_candidates,
            )

        # Step 2: Database-backed hybrid retrieval using EvidenceRetriever
        if self._retriever is not None:
            logger.debug(
                "Executing database-backed hybrid retrieval for claim %s across papers: %s",
                claim.claim_id,
                target_papers,
            )
            retrieval_claim = convert_atomic_claim_to_retrieval_claim(claim, target_papers)
            try:
                evidence_response: ClaimEvidenceResponse = self._retriever.retrieve_evidence(
                    claim=retrieval_claim,
                    top_k=max_candidates,
                )
            except Exception as exc:
                logger.error(
                    "Database hybrid retrieval failed for claim %s: %s",
                    claim.claim_id,
                    exc,
                )
                # Step 4: Do not silently fall back to synthetic evidence when database retrieval fails
                raise

            return [
                convert_evidence_item_to_retrieved_evidence(item)
                for item in evidence_response.evidence
            ]

        # Step 4: Local test corpus path (only real PDF corpus; NO synthetic ConvS2S in production)
        logger.info(
            "EvidenceRetriever is inactive (no database connection). "
            "Falling back to local PDF evidence corpus for claim %s.",
            claim.claim_id,
        )
        local_pool = self._get_or_load_local_corpus()
        return self._filter_and_rank_pool(
            pool=local_pool,
            claim=claim,
            target_papers=set(target_papers),
            max_candidates=max_candidates,
        )

    def _get_or_load_local_corpus(self) -> list[RetrievedEvidence]:
        """Obtain or lazily load the local corpus."""
        if self._corpus is not None:
            return self._corpus

        passages = self._load_pdf_corpus()
        if self._include_synthetic_fallback:
            # Only loaded when explicitly enabled for isolated legacy fixtures
            passages.extend(self._load_synthetic_test_data())

        self._corpus = passages
        return self._corpus

    def _load_pdf_corpus(self) -> list[RetrievedEvidence]:
        """Load document passages from backend/data/test_paper.pdf if present."""
        passages: list[RetrievedEvidence] = []
        pdf_path = DATA_DIR / "test_paper.pdf"
        if not pdf_path.is_file():
            return passages

        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(str(pdf_path))
            heading_pat = re.compile(
                r"^(Abstract|"
                r"\d+(\.\d+)*\s+[A-Z][A-Za-z0-9\s,\'-]+|"
                r"Table\s+\d+:|"
                r"Figure\s+\d+:|"
                r"References|"
                r"Acknowledgements|"
                r"Conclusion)"
            )

            def is_heading_line(line: str) -> bool:
                line = line.strip()
                if not line:
                    return False
                if line.startswith(("Table ", "Figure ")):
                    return True
                if line == "Abstract":
                    return True
                if heading_pat.match(line) and len(line) < 70 and not line.endswith("."):
                    return True
                return False

            active_section = None

            for page_idx in range(len(pdf)):
                page_num = page_idx + 1
                raw_text = pdf[page_idx].get_textpage().get_text_range()
                text = raw_text.replace("\ufffe", "-").replace("\xad", "-")
                text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1-\2", text)
                text = text.replace("\r\n", "\n").strip()
                if not text or page_num >= 10:
                    continue

                lines = [l.strip() for l in text.split("\n") if l.strip()]
                curr_heading = f"{active_section} (cont.)" if active_section else None
                curr_lines: list[str] = []

                for line in lines:
                    if page_num == 1 and ("@" in line or "Google " in line or "University of" in line):
                        continue
                    if page_num == 1 and line.startswith("∗Equal contribution"):
                        break
                    if line.startswith(("∗", "†", "‡")) or "Conference on Neural Information Processing" in line:
                        continue

                    if is_heading_line(line):
                        if curr_lines:
                            passage_content = "\n".join(curr_lines).strip()
                            if len(passage_content) > 60 and not (page_num == 1 and curr_heading is None):
                                passages.append(
                                    RetrievedEvidence(
                                        evidence_id=f"ev_vaswani_p{page_num}_{len(passages) + 1:02d}",
                                        paper_id="Vaswani_2017_Attention",
                                        chunk_id=f"page_{page_num}",
                                        text=passage_content,
                                        page=page_num,
                                        section=curr_heading,
                                        retrieval_method="local_pdf_corpus",
                                        retrieval_score=1.0,
                                    )
                                )
                            curr_lines = []
                        curr_heading = line[:60]
                        active_section = curr_heading
                        curr_lines.append(line)
                    else:
                        curr_lines.append(line)

                if curr_lines:
                    passage_content = "\n".join(curr_lines).strip()
                    if len(passage_content) > 60 and not (page_num == 1 and curr_heading is None):
                        passages.append(
                            RetrievedEvidence(
                                evidence_id=f"ev_vaswani_p{page_num}_{len(passages) + 1:02d}",
                                paper_id="Vaswani_2017_Attention",
                                chunk_id=f"page_{page_num}",
                                text=passage_content,
                                page=page_num,
                                section=curr_heading,
                                retrieval_method="local_pdf_corpus",
                                retrieval_score=1.0,
                            )
                        )
        except Exception as exc:
            logger.warning("Could not ingest test_paper.pdf: %s", exc)

        return passages

    def _load_synthetic_test_data(self) -> list[RetrievedEvidence]:
        """Load synthetic ConvS2S extraction items strictly for isolated test fixtures when enabled."""
        passages: list[RetrievedEvidence] = []
        convs2s_path = DATA_DIR / "convs2s_extraction.json"
        if not convs2s_path.is_file():
            return passages

        try:
            data = json.loads(convs2s_path.read_text(encoding="utf-8"))
            for idx, m in enumerate(data.get("methodology", []), 1):
                ev = m.get("evidence", {})
                q = ev.get("quote")
                if q:
                    passages.append(
                        RetrievedEvidence(
                            evidence_id=f"ev_convs2s_m_{idx:02d}",
                            paper_id="Gehring_2017_ConvS2S",
                            chunk_id="section_3",
                            text=f"[SYNTHETIC TEST DATA] {m.get('technique', '')}: {m.get('description', '')} Quote: {q}",
                            page=ev.get("page") or 2,
                            section=ev.get("section") or "Section 3",
                            retrieval_method="synthetic_test_corpus",
                            retrieval_score=1.0,
                        )
                    )
            for idx, em in enumerate(data.get("evaluation_metrics", []), 1):
                ev = em.get("evidence", {})
                q = ev.get("quote")
                if q:
                    passages.append(
                        RetrievedEvidence(
                            evidence_id=f"ev_convs2s_em_{idx:02d}",
                            paper_id="Gehring_2017_ConvS2S",
                            chunk_id="section_4",
                            text=f"[SYNTHETIC TEST DATA] Metric: {em.get('metric_name')} {em.get('value')} on {em.get('task_or_benchmark')}. Quote: {q}",
                            page=ev.get("page") or 6,
                            section=ev.get("section") or "Section 4",
                            retrieval_method="synthetic_test_corpus",
                            retrieval_score=1.0,
                        )
                    )
        except Exception as exc:
            logger.warning("Could not ingest convs2s_extraction.json: %s", exc)

        return passages

    def _filter_and_rank_pool(
        self,
        pool: list[RetrievedEvidence],
        claim: AtomicClaim,
        target_papers: set[str],
        max_candidates: int = 5,
    ) -> list[RetrievedEvidence]:
        """Filter and rank candidate evidence passages from a pool."""
        filtered_pool = [p for p in pool if not target_papers or p.paper_id in target_papers]
        if not filtered_pool:
            return []

        query_tokens = set(tokenize_content_words(claim.claim_text))
        query_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim.claim_text))

        hint_quotes = [
            ref.quote.lower().strip()
            for ref in claim.source_references
            if ref and ref.quote
        ]

        clean_q_words = [
            w for w in re.sub(r"[^a-zA-Z0-9\s-]", " ", claim.claim_text.lower()).split()
            if len(w) > 2 and w not in STOPWORDS
        ]
        query_bigrams = [
            f"{clean_q_words[i]} {clean_q_words[i+1]}"
            for i in range(len(clean_q_words) - 1)
        ]

        scored_candidates: list[tuple[float, str, RetrievedEvidence]] = []

        for passage in filtered_pool:
            passage_text_lower = passage.text.lower()
            passage_tokens = set(tokenize_content_words(passage.text))

            token_score = 0.0
            if query_tokens:
                matched_tokens = query_tokens.intersection(passage_tokens)
                token_score = len(matched_tokens) / len(query_tokens)

            num_score = 0.0
            if query_numbers:
                matched_nums = sum(1 for n in query_numbers if n in passage_text_lower)
                num_score = matched_nums / len(query_numbers)

            hint_boost = 0.0
            for hq in hint_quotes:
                if hq in passage_text_lower or passage_text_lower in hq:
                    hint_boost = 0.35
                    break

            phrase_boost = 0.2 if claim.claim_text.lower() in passage_text_lower else 0.0

            bigram_boost = 0.0
            if query_bigrams:
                matched_bigrams = sum(1 for bg in query_bigrams if bg in passage_text_lower)
                bigram_boost = min(0.25, (matched_bigrams / len(query_bigrams)) * 0.25)

            total_score = (
                (0.40 * token_score)
                + (0.30 * num_score)
                + (0.15 * hint_boost)
                + (0.10 * phrase_boost)
                + (0.05 * (bigram_boost / 0.25 if bigram_boost > 0 else 0.0))
            )

            if len(passage.text.split()) < 15:
                total_score *= 0.5

            normalized_score = min(1.0, round(total_score, 4))
            method = "reference_hint" if hint_boost > 0 else "lexical_overlap"

            scored_passage = passage.model_copy(
                update={"retrieval_score": normalized_score, "retrieval_method": method}
            )
            scored_candidates.append((normalized_score, method, scored_passage))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        results: list[RetrievedEvidence] = []
        for score, _, cand in scored_candidates[:max_candidates]:
            if score > 0.05 or len(results) < 2:
                results.append(cand)

        return results


_SHARED_EMBEDDING_SERVICE: Any = None


def get_shared_embedding_service() -> Any:
    """Return a process-level cached EmbeddingService instance to avoid repeated model loading."""
    global _SHARED_EMBEDDING_SERVICE
    if _SHARED_EMBEDDING_SERVICE is None:
        try:
            from backend.retrieval.embeddings import EmbeddingService

            _SHARED_EMBEDDING_SERVICE = EmbeddingService()
        except Exception as exc:
            logger.info("Shared EmbeddingService could not be pre-initialized: %s", exc)
            return None
    return _SHARED_EMBEDDING_SERVICE


def create_retriever_from_session(session: Any, embedding_service: Any = None) -> EvidenceRetriever:
    """Construct an EvidenceRetriever bound to an active SQLAlchemy session.

    Reuses the shared process-level EmbeddingService.
    Does NOT manage session transactions; relies on the caller's session lifecycle.
    """
    from backend.database.repository import ChunkRepository
    from backend.retrieval.hybrid_search import HybridSearchEngine
    from backend.retrieval.lexical_search import LexicalSearchEngine
    from backend.retrieval.vector_search import VectorSearchEngine

    emb = embedding_service or get_shared_embedding_service()
    if emb is None:
        from backend.retrieval.embeddings import EmbeddingService

        emb = EmbeddingService()

    repo = ChunkRepository(session=session)
    v_search = VectorSearchEngine(session=session)
    l_search = LexicalSearchEngine(repository=repo)
    h_search = HybridSearchEngine(
        embedding_service=emb,
        vector_search=v_search,
        lexical_search=l_search,
    )
    return EvidenceRetriever(
        repository=repo,
        embedding_service=emb,
        hybrid_search=h_search,
    )

