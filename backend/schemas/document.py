from enum import Enum
from pydantic import BaseModel, ConfigDict


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    filename: str
    status: DocumentStatus


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[DocumentMetadata]


class DocumentStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    status: DocumentStatus
