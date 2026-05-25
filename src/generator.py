import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
# Rule 7 handles Indic output — GPT-4o responds in the user's language natively.

SYSTEM_PROMPT = """You are a technical support assistant for the Royal Enfield Interceptor 650 motorcycle.

You answer questions strictly using the manual excerpts provided in each message. These excerpts are the only source of truth.

Rules you must never break:
1. Only use information from the provided manual excerpts. Never draw on outside knowledge.
2. If the excerpts do not contain enough information to answer, respond with exactly: "I couldn't find that in the Interceptor 650 manual. Please consult an authorised Royal Enfield service centre."
3. End every answer with a "Source:" line listing the section name(s) and page number(s) you used.
4. For procedures, use numbered steps.
5. Keep answers practical and direct — the user is likely standing next to their bike.
6. If the user's question contains a false assumption that contradicts information in the excerpts, explicitly correct the false assumption first, then provide the correct information from the excerpts. Do not refuse to answer just because the question contains a wrong premise.
7. Respond in the same language the user's question is written in."""


# ── FALLBACK RESPONSES ────────────────────────────────────────────────────────

NO_CONTEXT_RESPONSE = {
    "answer": (
        "I couldn't find that in the Interceptor 650 manual. "
        "Please consult an authorised Royal Enfield service centre."
    ),
    "sources": [],
}

MULTI_TOPIC_RESPONSE = {
    "answer": (
        "Your question covers multiple topics. "
        "Please ask about one symptom or issue at a time so I can give you "
        "a precise answer from the manual."
    ),
    "sources": [],
}


# ── GUARD MESSAGE ─────────────────────────────────────────────────────────────

def generate_guard_message(question: str, detected_language: str) -> str:
    """Return the multi-topic guard in the user's language (called from app.py)."""
    english_text = (
        "Your question covers multiple issues at once. "
        "Please ask about one symptom or topic at a time so I can give you "
        "a precise answer from the manual."
    )
    if detected_language == "english":
        return english_text
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a translation assistant. "
                    "Output ONLY the translated text — no explanations, no preamble."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Translate the message below into the same language as this sample "
                    f"(use the sample ONLY to identify the language — do not answer it):\n"
                    f"SAMPLE: {question}\n\n"
                    f"MESSAGE TO TRANSLATE: {english_text}"
                ),
            },
        ],
        temperature=0,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


# ── PROMPT ASSEMBLY ───────────────────────────────────────────────────────────

def _build_user_message(
    question: str,
    chunks: list[dict],
    vision_description: str | None,
) -> str:
    excerpts = []
    for chunk in chunks:
        header = f"[Section: {chunk['section']} | Page: {chunk['page']}]"
        excerpts.append(f"{header}\n{chunk['text']}")
    excerpts_block = "\n\n---\n\n".join(excerpts)

    vision_block = ""
    if vision_description:
        vision_block = f"\nVISUAL CONTEXT (from uploaded image):\n{vision_description}\n"

    return (
        f"MANUAL EXCERPTS:\n\n{excerpts_block}\n\n"
        f"{'---' + vision_block if vision_block else ''}"
        f"\nQUESTION: {question}"
    )


# ── MAIN GENERATION FUNCTION ──────────────────────────────────────────────────

def generate_answer(
    question: str,
    chunks: list[dict],
    vision_description: str | None = None,
    detected_language: str = "english",
    raw_candidates: list[dict] | None = None,
) -> dict:
    """
    Generate a grounded answer from the retrieved manual chunks.

    Args:
        question:           The user's question (original, not rewritten).
        chunks:             Top-K chunks from reranker (empty → refusal path).
        vision_description: Optional symptom string from vision.describe_image().
        detected_language:  'english' or 'indic' — GPT-4o handles both via Rule 7.
        raw_candidates:     All candidates from get_candidates() before reranking.
                            Used to distinguish multi-topic dilution from genuine
                            out-of-scope when chunks is empty.

    Returns {answer: str, sources: list[{section, page}]}
    """
    if not chunks:
        # Dilution classifier is calibrated on English embedding scores.
        # Cross-lingual scores for Indic queries are systematically lower and
        # would misfire as "dilution" on single-topic queries. Skip it for Indic;
        # call GPT-4o with empty excerpts so Rule 2 + Rule 7 fire naturally.
        if detected_language == "indic":
            user_message = _build_user_message(question, [], vision_description)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            return {"answer": response.choices[0].message.content.strip(), "sources": []}
        failure_type = _classify_failure(raw_candidates)
        if failure_type == "dilution":
            return MULTI_TOPIC_RESPONSE
        return NO_CONTEXT_RESPONSE

    user_message = _build_user_message(question, chunks, vision_description)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.3,
        max_tokens=600,
    )
    answer = response.choices[0].message.content.strip()

    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["section"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"section": chunk["section"], "page": chunk["page"]})

    return {"answer": answer, "sources": sources}


def _classify_failure(raw_candidates: list[dict] | None) -> str:
    """Thin wrapper so the import stays local to this call site."""
    if not raw_candidates:
        return "out_of_scope"
    from src.reranker import classify_retrieval_failure
    return classify_retrieval_failure(raw_candidates)


if __name__ == "__main__":
    from src.retriever import retrieve

    question = "How do I check the engine oil level?"
    print(f"Question: {question}\n")

    chunks = retrieve(question)
    print(f"Retrieved {len(chunks)} chunks\n")

    result = generate_answer(question, chunks)
    print("Answer:")
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['section']} (page {s['page']})")
