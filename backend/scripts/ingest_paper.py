#!/usr/bin/env python3
"""
ingest_paper.py — CLI script to ingest a research paper PDF into PostgreSQL with pgvector.

Usage:
    PYTHONPATH=. python3 backend/scripts/ingest_paper.py [pdf_path] [--paper-id PAPER_ID]
    # Default without arguments ingests backend/data/test_paper.pdf
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
backend_root = Path(__file__).resolve().parent.parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from backend.retrieval.ingest import main

if __name__ == "__main__":
    sys.exit(main())
