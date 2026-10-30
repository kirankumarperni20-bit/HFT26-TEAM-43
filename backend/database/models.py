"""SQLAlchemy declarative models for the document chunk storage."""

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from backend.config import EMBEDDING_DIMENSION


class Base(DeclarativeBase):
    """Base declarative class for all database models."""

    pass


class ChunkRecord(Base):
    """Database model representing an ingested research paper chunk.

    Mapped strictly to the 'chunks' table following the agreed shared contract.
    Chunk IDs and extracted text originate from Member 1's ingestion pipeline.
    """

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    paper_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    paragraph_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    section: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSION).with_variant(JSON, "sqlite"),
        nullable=True,
        default=None,
    )

    __table_args__ = (
        Index(
            "ix_chunks_paper_page_para",
            "paper_id",
            "page_number",
            "paragraph_number",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<ChunkRecord(chunk_id='{self.chunk_id}', "
            f"paper_id='{self.paper_id}', page={self.page_number}, "
            f"para={self.paragraph_number}, section='{self.section}')>"
        )

    def to_pydantic(self) -> "DocumentChunk":
        """Convert this database record into a validated DocumentChunk Pydantic model."""
        from backend.schemas.chunk import DocumentChunk

        return DocumentChunk(
            chunk_id=self.chunk_id,
            paper_id=self.paper_id,
            page_number=self.page_number,
            paragraph_number=self.paragraph_number,
            text=self.text,
            section=self.section,
        )
