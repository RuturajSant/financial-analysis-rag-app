"""
ChromaDB vector store — embedding, upserting, and hybrid retrieval.

Design (from references/retrieval.md):
- Content-hash deduplication: only re-embeds chunks whose hash changed.
- Stores embedding_model name in collection metadata to catch embedding drift.
- Hybrid search: dense (Chroma) + sparse (BM25) with Reciprocal Rank Fusion.
- Reranking: local bge-reranker-base cross-encoder (CPU, no paid service needed).
"""
from __future__ import annotations

from typing import Any

import chromadb
from chromadb import Collection
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from backend.core.config import settings
from backend.core.logger import get_logger
from backend.ingestion.chunker import Chunk

logger = get_logger(__name__)

EXPECTED_EMBED_MODEL = settings.embed_model

# Lazy singletons
_embedder: OpenAIEmbeddings | None = None
_reranker: CrossEncoder | None = None
_chroma_client: chromadb.PersistentClient | None = None


def _get_embedder() -> OpenAIEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbeddings(
            model=settings.embed_model,
            openai_api_key=settings.openai_api_key,
        )
    return _embedder


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("loading_reranker", extra={"model": "BAAI/bge-reranker-base"})
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker


def _get_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _chroma_client


def _get_or_create_collection(doc_id: str) -> Collection:
    """Each document gets its own Chroma collection keyed by doc_id."""
    client = _get_client()
    collection_name = f"doc_{doc_id}"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"embedding_model": EXPECTED_EMBED_MODEL, "doc_id": doc_id},
    )
    return collection


def _assert_embedding_model(collection: Collection) -> None:
    """Guard against silent embedding drift when model changes mid-project."""
    stored = collection.metadata.get("embedding_model")
    if stored and stored != EXPECTED_EMBED_MODEL:
        raise RuntimeError(
            f"Collection '{collection.name}' was embedded with '{stored}', "
            f"but current embed model is '{EXPECTED_EMBED_MODEL}'. "
            "Re-embed the collection before querying."
        )


def upsert_chunks(chunks: list[Chunk], doc_id: str) -> int:
    """
    Embed and upsert chunks into ChromaDB, skipping chunks whose hash hasn't changed.

    Returns:
        Number of chunks actually (re-)embedded and upserted.
    """
    collection = _get_or_create_collection(doc_id)
    embedder = _get_embedder()

    # Fetch existing hashes from ChromaDB to detect what changed
    existing = collection.get(include=["metadatas"])
    existing_hashes: dict[str, str] = {
        meta["chunk_id"]: meta["content_hash"]
        for meta in (existing["metadatas"] or [])
        if meta and "chunk_id" in meta
    }

    new_chunks = [
        c for c in chunks
        if existing_hashes.get(c.chunk_id) != c.content_hash
    ]

    if not new_chunks:
        logger.info("upsert_skip_all", extra={"doc_id": doc_id, "reason": "no content changed"})
        return 0

    texts = [c.text for c in new_chunks]
    embeddings = embedder.embed_documents(texts)

    collection.upsert(
        ids=[c.chunk_id for c in new_chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "page_number": c.page_number,
                "section_type": c.section_type,
                "section_title": c.section_title or "",
                "section_id": c.section_id,
                "table_id": c.table_id or "",
                "content_hash": c.content_hash,
            }
            for c in new_chunks
        ],
    )

    logger.info(
        "upsert_done",
        extra={"doc_id": doc_id, "upserted": len(new_chunks), "skipped": len(chunks) - len(new_chunks)},
    )
    return len(new_chunks)


def hybrid_search(
    query: str,
    doc_id: str,
    k: int = 8,
    rerank_top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    Hybrid dense + BM25 search with cross-encoder reranking.

    Returns list of dicts with keys: text, page_number, section_title, section_type, chunk_id, score.
    """
    collection = _get_or_create_collection(doc_id)
    _assert_embedding_model(collection)

    # Pull all docs for BM25 (small corpus per doc — affordable)
    all_docs = collection.get(include=["documents", "metadatas"])
    all_texts: list[str] = all_docs["documents"] or []
    all_metadatas: list[dict] = all_docs["metadatas"] or []
    all_ids: list[str] = all_docs["ids"] or []

    if not all_texts:
        return []

    # ── Dense retrieval (top-20 candidates) ──────────────────────────────────
    embedder = _get_embedder()
    query_embedding = embedder.embed_query(query)
    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(20, len(all_texts)),
        include=["documents", "metadatas", "distances"],
    )
    dense_ids: list[str] = dense_results["ids"][0] if dense_results["ids"] else []

    # ── Sparse BM25 retrieval ─────────────────────────────────────────────────
    tokenized = [t.lower().split() for t in all_texts]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.lower().split())
    sparse_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:20]
    sparse_ids = [all_ids[i] for i in sparse_ranked_indices]

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────
    rrf_k = 60
    rrf_scores: dict[str, float] = {}
    for rank, doc_id_hit in enumerate(dense_ids):
        rrf_scores[doc_id_hit] = rrf_scores.get(doc_id_hit, 0) + 1 / (rrf_k + rank)
    for rank, doc_id_hit in enumerate(sparse_ids):
        rrf_scores[doc_id_hit] = rrf_scores.get(doc_id_hit, 0) + 1 / (rrf_k + rank)

    fused_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[: k * 2]

    # ── Rerank with cross-encoder ─────────────────────────────────────────────
    id_to_text = dict(zip(all_ids, all_texts))
    id_to_meta = dict(zip(all_ids, all_metadatas))

    candidates = [(fid, id_to_text[fid]) for fid in fused_ids if fid in id_to_text]
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, text) for _, text in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])[:rerank_top_n]

    # ── Assemble result objects ───────────────────────────────────────────────
    # "Lost in the middle" mitigation: best chunk first, second-best last
    results = []
    for (cid, text), score in ranked:
        meta = id_to_meta.get(cid, {})
        results.append(
            {
                "chunk_id": cid,
                "text": text,
                "page_number": meta.get("page_number"),
                "section_title": meta.get("section_title", ""),
                "section_type": meta.get("section_type", "narrative"),
                "score": float(score),
            }
        )

    # Re-order: highest relevance first, second-highest last (lost-in-middle fix)
    if len(results) > 2:
        reordered = [results[0]] + results[2:] + [results[1]]
        results = reordered

    logger.info(
        "hybrid_search_done",
        extra={
            "query_preview": query[:80],
            "candidates": len(candidates),
            "returned": len(results),
        },
    )
    return results


def delete_document(doc_id: str) -> None:
    """Remove a document's collection from ChromaDB (e.g., on session cleanup)."""
    client = _get_client()
    collection_name = f"doc_{doc_id}"
    try:
        client.delete_collection(collection_name)
        logger.info("collection_deleted", extra={"doc_id": doc_id})
    except Exception as exc:
        logger.warning("collection_delete_failed", extra={"doc_id": doc_id, "error": str(exc)})
