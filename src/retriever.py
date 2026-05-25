import os
import sys
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL    = "text-embedding-3-small"
CHROMA_PATH    = "./chroma_db"
COLLECTION     = "bike_manual"
BM25_PASS_RANK = 5    # BM25 leg: top-N keyword matches count as a strong signal
SEMANTIC_FETCH = 20   # semantic candidates fetched before fusion
RRF_K          = 60   # RRF damping constant (standard default)
CANDIDATES     = 20   # how many fused candidates to return to the reranker

# ── BM25 INDEX CACHE ───────────────────────────────────────────────────────────

_bm25_index: BM25Okapi | None = None
_corpus: list[dict] | None = None


def _load_corpus() -> tuple[BM25Okapi, list[dict]]:
    """Pull all chunks from ChromaDB, build a BM25 index, and cache both."""
    global _bm25_index, _corpus
    if _bm25_index is not None:
        return _bm25_index, _corpus

    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_collection(name=COLLECTION)
    data = collection.get(include=["documents", "metadatas"])

    _corpus = [
        {
            "id":      id_,
            "text":    doc,
            "section": meta.get("section", "General"),
            "page":    meta.get("page", "?"),
        }
        for id_, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
    ]

    _bm25_index = BM25Okapi([chunk["text"].lower().split() for chunk in _corpus])
    return _bm25_index, _corpus


# ── SEMANTIC SEARCH ────────────────────────────────────────────────────────────

def _embed_query(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return response.data[0].embedding


def _semantic_search(query_embedding: list[float]) -> list[dict]:
    """Return top SEMANTIC_FETCH chunks with similarity scores. No threshold."""
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_collection(name=COLLECTION)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=SEMANTIC_FETCH,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "id":            results["ids"][0][i],
            "text":          results["documents"][0][i],
            "section":       results["metadatas"][0][i].get("section", "General"),
            "page":          results["metadatas"][0][i].get("page", "?"),
            "similarity":    round(1 - results["distances"][0][i], 4),
            "semantic_rank": i + 1,
        }
        for i in range(len(results["documents"][0]))
    ]


# ── BM25 SEARCH ────────────────────────────────────────────────────────────────

def _bm25_search(query: str) -> list[dict]:
    """Score every chunk with BM25 and return them sorted best-first."""
    bm25, corpus = _load_corpus()
    scores = bm25.get_scores(query.lower().split())
    return sorted(
        [{"corpus_idx": i, "bm25_score": float(scores[i])} for i in range(len(corpus))],
        key=lambda x: x["bm25_score"],
        reverse=True,
    )


# ── RECIPROCAL RANK FUSION ─────────────────────────────────────────────────────

def _fuse(semantic: list[dict], bm25_ranked: list[dict]) -> list[dict]:
    """
    Combine semantic and BM25 ranked lists with RRF.
    RRF score = 1/(k + semantic_rank) + 1/(k + bm25_rank)
    """
    _, corpus = _load_corpus()

    id_to_bm25 = {
        corpus[item["corpus_idx"]]["id"]: {
            "bm25_rank":  rank + 1,
            "bm25_score": item["bm25_score"],
        }
        for rank, item in enumerate(bm25_ranked)
    }

    fused = []
    for sem in semantic:
        bm25 = id_to_bm25.get(sem["id"], {"bm25_rank": len(corpus) + 1, "bm25_score": 0.0})
        rrf_score = 1 / (RRF_K + sem["semantic_rank"]) + 1 / (RRF_K + bm25["bm25_rank"])

        fused.append({
            "id":            sem["id"],
            "text":          sem["text"],
            "section":       sem["section"],
            "page":          sem["page"],
            "similarity":    sem["similarity"],
            "semantic_rank": sem["semantic_rank"],
            "bm25_rank":     bm25["bm25_rank"],
            "bm25_score":    round(bm25["bm25_score"], 4),
            "rrf_score":     round(rrf_score, 6),
        })

    return sorted(fused, key=lambda x: x["rrf_score"], reverse=True)


# ── CANDIDATE RETRIEVAL (used by reranker) ─────────────────────────────────────

def get_candidates(query: str) -> list[dict]:
    """
    Return the top CANDIDATES chunks after RRF fusion, with no threshold applied.
    This is the input list for the reranker.
    """
    if not query.strip():
        return []
    query_embedding  = _embed_query(query)
    semantic_results = _semantic_search(query_embedding)
    bm25_ranked      = _bm25_search(query)
    fused            = _fuse(semantic_results, bm25_ranked)
    return fused[:CANDIDATES]


# ── QUERY REWRITER ─────────────────────────────────────────────────────────────

_REWRITE_SYSTEM = """You are a query normaliser for a motorcycle manual Q&A system.

If the user's question contains a false assertion, embedded assumption, or conversational filler, strip it and rephrase as a neutral information-seeking question in English.

Examples:
  Input:  "The manual says to use 98 octane premium fuel right?"
  Output: "What fuel type does the Interceptor 650 require?"

  Input:  "Since the Interceptor uses diesel, how often should I change the fuel filter?"
  Output: "What fuel type does the Interceptor 650 use?"

  Input:  "How do I check the engine oil level?"
  Output: "How do I check the engine oil level?"

  Input:  "Bhai, engine oil kab change karna chahiye?"
  Output: "How often should the engine oil be changed?"

  Input:  "भाई, इंजन ऑयल कब चेंज करना चाहिए?"
  Output: "How often should the engine oil be changed?"

Rules:
1. If the question contains an unverified factual claim, neutralise it into a lookup.
2. Strip informal address words or conversational filler at the start of the question (e.g. "Bhai", "yaar", "dost", "भाई", "यार") before rephrasing.
3. If the question is in a non-English language or code-switched (e.g. Hinglish), translate it to English — the manual is in English and English queries retrieve better.
4. If it is already a neutral English question with no false premise, return it unchanged.
5. Return ONLY the reformulated question — no explanation, no prefix."""


def rewrite_query(query: str) -> str:
    """
    Neutralise false-premise queries before retrieval.
    The original query still goes to the generator so GPT-4o can correct the false premise.
    Returns the query unchanged if no rewriting is needed.
    """
    if not query.strip():
        return query
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _REWRITE_SYSTEM},
            {"role": "user",   "content": query},
        ],
        temperature=0,
        max_tokens=80,
    )
    return response.choices[0].message.content.strip()


# ── PUBLIC INTERFACE ───────────────────────────────────────────────────────────

def retrieve(query: str) -> list[dict]:
    """
    Full retrieval pipeline: hybrid search + LLM re-ranking.

    Internally:
      1. get_candidates() → top 20 chunks via semantic + BM25 + RRF
      2. reranker.rerank() → GPT-4o scores each candidate for relevance
      3. Returns top 5 by reranker score, or [] if top score < 6

    Public interface is unchanged: callers receive a list of chunk dicts
    with 'text', 'section', 'page', 'similarity'.
    """
    from src.reranker import rerank   # local import avoids circular-import risk
    candidates = get_candidates(query)
    return rerank(query, candidates)


# ── SMOKE TEST ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.reranker import rerank

    test_queries = [
        "what kind of fuel should I use",
        "how do I check the engine oil level",
        "what is the capital of France",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 72)

        candidates = get_candidates(query)
        print(f"  Top 5 RRF candidates (pre-rerank):")
        for i, c in enumerate(candidates[:5], 1):
            print(
                f"    {i}. RRF {c['rrf_score']:.5f} | "
                f"Sem {c['similarity']:.3f} (#{c['semantic_rank']:2d}) | "
                f"BM25 {c['bm25_score']:.2f} (#{c['bm25_rank']:3d}) | "
                f"{c['section'][:28]} | p{c['page']}"
            )

        final = rerank(query, candidates)
        print(f"\n  After re-ranking — {len(final)} chunk(s) returned:")
        for i, c in enumerate(final, 1):
            print(
                f"    {i}. Rerank {c['rerank_score']:4.1f}/10 | "
                f"Sem {c['similarity']:.3f} | "
                f"{c['section'][:28]} | p{c['page']}"
            )
        if not final:
            print("    => REFUSED (top rerank score < 6)")
