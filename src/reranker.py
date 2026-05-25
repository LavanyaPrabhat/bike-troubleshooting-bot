import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TOP_K              = 5    # chunks to return after reranking
RERANK_PASS_SCORE  = 6    # minimum score for the top result; below this → refusal
EXCERPT_CHARS      = 400  # how much of each chunk to show the reranker (saves tokens)

RERANKER_SYSTEM = """You are a relevance judge for a motorcycle manual Q&A system.

Rate how directly each numbered excerpt answers the user's question on a scale of 0–10.

Scoring guide:
  10  Excerpt directly and completely answers the question
  7–9 Excerpt contains the answer but requires some reading or inference.
      IMPORTANT: For symptom-description queries (e.g. "white smoke from exhaust",
      "engine knocking", "oil leak"), a troubleshooting or diagnostic page that lists
      possible causes for that exact symptom scores 7–9 — that list IS the correct
      answer to a diagnostic question.
  4–6 Excerpt is topically related but does not actually address the question
  1–3 Excerpt mentions the topic only in passing
  0   Excerpt is completely unrelated to the question

Be strict for specification queries ("what fuel type?", "what tyre pressure?") — an
excerpt that mentions the topic without giving the spec should score low.
Be generous for symptom queries ("why is X happening?", "what causes X?") — a page
listing causes for that exact symptom is directly and completely relevant.

Respond with ONLY a valid JSON array — no explanation, no markdown:
[{"id": 0, "score": 7}, {"id": 1, "score": 3}, ...]"""


def _build_reranker_prompt(query: str, candidates: list[dict]) -> str:
    """Format the candidates as a numbered list for GPT-4o to score."""
    lines = [f'User question: "{query}"\n\nExcerpts:']
    for i, chunk in enumerate(candidates):
        excerpt = chunk["text"][:EXCERPT_CHARS].replace("\n", " ").strip()
        lines.append(f"[{i}] (Section: {chunk['section']}, Page {chunk['page']})\n{excerpt}")
    return "\n\n".join(lines)


def _parse_scores(response_text: str, n: int) -> list[float]:
    """
    Parse GPT-4o's JSON score array.
    Falls back to 0.0 for any missing or malformed entry so we never crash.
    """
    try:
        parsed = json.loads(response_text)
        scores = {item["id"]: float(item["score"]) for item in parsed}
        return [scores.get(i, 0.0) for i in range(n)]
    except (json.JSONDecodeError, KeyError, TypeError):
        return [0.0] * n


# ── RETRIEVAL FAILURE CLASSIFIER ─────────────────────────────────────────────
# Thresholds calibrated empirically on this corpus (text-embedding-3-small, 130 chunks).
#
# Key finding: multi-topic queries score HIGH (0.46–0.49) because the embedding
# model still finds individually relevant pages for each topic. The dilution shows
# up as section diversity, not low scores. Genuine OOS queries score < 0.20 because
# the entire corpus is motorcycle-domain and has near-zero overlap with unrelated topics.
#
# Signal for dilution: top-5 candidates score ≥ 0.40 AND scatter across ≥ 5 sections.
# Signal for OOS:      top candidate scores < 0.20.
#
# _DILUTION_MIN_SECS raised from 3 → 5 after Malayalam diagnostic showed a
# single-topic "check engine oil" query retrieving from 4 sections (MINOR
# MAINTENANCE TIPS × 2 + WARNING INDICATIONS + RECOMMENDED LUBRICANTS +
# PERIODICAL MAINTENANCE) — all thematically related — and misfiring as
# "dilution". Requiring all top-5 candidates to be from distinct sections is a
# tighter signal that only genuinely multi-topic queries satisfy. The original
# calibration case ("engine noise + brakes + battery") produced 5 distinct
# sections and still fires correctly at the new threshold.

_DILUTION_OOS_THRESHOLD  = 0.20  # below this → genuinely not in manual
_DILUTION_SPREAD_MIN     = 0.40  # at or above this = retrieval found real content
_DILUTION_MIN_SECS       = 5     # all top-5 must be from distinct sections → true dilution


def classify_retrieval_failure(candidates: list[dict]) -> str:
    """Return 'dilution' or 'out_of_scope' given candidates that failed the reranker.

    Distinguishes multi-topic queries (information IS in the manual but the query
    is too broad for precise retrieval) from genuinely off-topic queries.
    No extra API call — uses only the cosine similarity scores already on each candidate.
    """
    if not candidates:
        return "out_of_scope"

    by_sim = sorted(candidates, key=lambda c: c.get("similarity", 0), reverse=True)
    top5   = by_sim[:5]
    top_score = top5[0].get("similarity", 0) if top5 else 0

    if top_score < _DILUTION_OOS_THRESHOLD:
        return "out_of_scope"

    above_spread  = [c for c in top5 if c.get("similarity", 0) >= _DILUTION_SPREAD_MIN]
    distinct_secs = len({c.get("section", "") for c in top5})

    if len(above_spread) >= 3 and distinct_secs >= _DILUTION_MIN_SECS:
        return "dilution"

    return "out_of_scope"


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """
    Score each candidate with GPT-4o and return the top TOP_K by relevance.

    Args:
        query:      The user's original question.
        candidates: Up to 20 chunk dicts from the hybrid retriever.

    Returns:
        Up to TOP_K chunk dicts, each with an added 'rerank_score' field (0–10).
        Returns an empty list if the top reranker score is below RERANK_PASS_SCORE —
        the generator treats an empty list as a refusal without calling GPT-4o.
    """
    if not candidates:
        return []

    prompt = _build_reranker_prompt(query, candidates)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": RERANKER_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0,      # deterministic scoring
        max_tokens=300,     # 20 scores as JSON is well under 100 tokens; 300 is generous
    )

    scores = _parse_scores(response.choices[0].message.content.strip(), len(candidates))

    # Attach scores and sort best-first
    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = score

    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    top = ranked[:TOP_K]

    # Refuse if even the best result isn't a strong match
    if not top or top[0]["rerank_score"] < RERANK_PASS_SCORE:
        return []

    return top
