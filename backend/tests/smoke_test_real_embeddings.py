"""Smoke test script for verifying real model download and inference with BAAI/bge-small-en-v1.5.

Run explicitly with:
    .\\.venv\\Scripts\\python backend/tests/smoke_test_real_embeddings.py

Note: Not executed by default during pytest unit test runs.
"""

import sys
from pathlib import Path

# Add project root to sys.path so backend package can be imported directly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

from backend.retrieval.embeddings import EmbeddingService


def run_smoke_test() -> None:
    print("=" * 60)
    print("RUNNING REAL EMBEDDING MODEL SMOKE TEST")
    print("Target Model: BAAI/bge-small-en-v1.5")
    print("=" * 60)

    print("\n1. Initializing EmbeddingService (lazy)...")
    service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5", device="cpu")
    print("   [OK] Service instantiated (model not yet loaded).")

    print("\n2. Loading real model weights from Hugging Face / cache...")
    model = service.model
    dim = service.dimension
    max_len = service.max_seq_length
    print(f"   [OK] Model loaded successfully: {type(model).__name__}")
    print(f"   [OK] Discovered embedding dimension: {dim}")
    print(f"   [OK] Discovered max sequence length: {max_len}")
    assert dim == 384, f"Expected 384 dimensions, got {dim}"
    assert max_len == 512, f"Expected 512 max tokens, got {max_len}"

    print("\n3. Testing query embedding with BGE query instruction...")
    query = "The proposed model achieved an accuracy of 92%."
    q_vec = service.embed_query(query)
    q_norm = float(np.linalg.norm(q_vec))
    print(f"   [OK] Query vector shape: {q_vec.shape}")
    print(f"   [OK] Query vector finite: {np.all(np.isfinite(q_vec))}")
    print(f"   [OK] Query vector L2 norm: {q_norm:.6f}")
    assert q_vec.shape == (384,)
    assert np.all(np.isfinite(q_vec))
    assert abs(q_norm - 1.0) < 1e-4

    print("\n4. Testing document batch embedding without query instruction...")
    passages = [
        "Transformer architectures utilize self-attention mechanisms to model global dependencies.",
        "Residual networks with 50 layers demonstrate superior convergence over vanilla networks.",
    ]
    doc_vecs = service.embed_documents(passages, batch_size=2)
    row_norms = [float(norm) for norm in np.linalg.norm(doc_vecs, axis=1)]
    print(f"   [OK] Document array shape: {doc_vecs.shape}")
    print(f"   [OK] Document vectors finite: {np.all(np.isfinite(doc_vecs))}")
    print(f"   [OK] Document vectors L2 norms: {row_norms}")
    assert doc_vecs.shape == (2, 384)
    assert np.all(np.isfinite(doc_vecs))
    for norm in row_norms:
        assert abs(norm - 1.0) < 1e-4

    print("\n" + "=" * 60)
    print("ALL REAL MODEL SMOKE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception as exc:
        print(f"\n[FAIL] Smoke test failed: {exc}", file=sys.stderr)
        sys.exit(1)
