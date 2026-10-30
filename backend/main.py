import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.documents import router as documents_router, get_repository
from backend.api.research import router as research_router, get_research_job_repository

app = FastAPI(
    title="AI Research Assistant - Backend",
    description="Modular monolith backend for verifiable summaries and cross-paper synthesis.",
    version="0.1.0",
)

# Sensible localhost development defaults for frontend origins
DEFAULT_FRONTEND_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# Configurable origins via FRONTEND_ORIGINS environment variable
frontend_origins_env = os.getenv("FRONTEND_ORIGINS")
if frontend_origins_env:
    allowed_origins = [orig.strip() for orig in frontend_origins_env.split(",") if orig.strip()]
else:
    allowed_origins = DEFAULT_FRONTEND_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(documents_router, prefix="/api")
app.include_router(research_router, prefix="/api")


@app.get("/health")
def health_check():
    """Health check endpoint to verify backend service status."""
    return {"status": "healthy"}


def get_shared_repository():
    """Access to the application-level shared document repository."""
    return get_repository()


def get_shared_research_repository():
    """Access to the application-level shared research job repository."""
    return get_research_job_repository()
