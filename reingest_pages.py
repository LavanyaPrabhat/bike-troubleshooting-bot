"""
Targeted re-ingest for specific pages: extract via Vision (forced),
update ChromaDB entries in place.

Usage: python reingest_pages.py 96 97
"""
import sys
import base64
import fitz
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv()

client  = OpenAI()
PDF     = "./data/royal-enfield-interceptor-650-owners-manual-english.pdf"
CHROMA  = "./chroma_db"
COLL    = "bike_manual"
EMBED   = "text-embedding-3-small"

VISION_PROMPT = (
    "Describe the technical content of this page in detail — include part names, "
    "measurements, specifications, warning notices, and any procedures shown. "
    "Be specific and technical."
)


def vision_describe(fitz_page) -> str:
    pix    = fitz_page.get_pixmap(dpi=150)
    b64    = base64.b64encode(pix.tobytes("png")).decode("utf-8")
    resp   = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
            {"type": "text",      "text": VISION_PROMPT},
        ]}],
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()


def embed(text: str) -> list[float]:
    return client.embeddings.create(model=EMBED, input=[text]).data[0].embedding


def main(target_pages: list[int]):
    chroma     = chromadb.PersistentClient(path=CHROMA)
    collection = chroma.get_collection(name=COLL)
    pdf_fitz   = fitz.open(PDF)

    # Find existing chunks for the target pages
    existing = collection.get(include=["metadatas"])
    page_to_id = {
        meta["page"]: id_
        for id_, meta in zip(existing["ids"], existing["metadatas"])
        if meta["page"] in target_pages
    }

    print(f"Found existing chunks for pages: {list(page_to_id.keys())}")
    if len(page_to_id) != len(target_pages):
        missing = set(target_pages) - set(page_to_id.keys())
        print(f"WARNING: no existing chunk found for pages {missing} — skipping those")

    for page_num in target_pages:
        if page_num not in page_to_id:
            continue

        chunk_id = page_to_id[page_num]
        fitz_page = pdf_fitz[page_num - 1]   # 0-indexed

        print(f"\nPage {page_num} (chunk id={chunk_id}) — extracting via Vision...")
        new_text = vision_describe(fitz_page)
        print(f"  Vision text ({len(new_text)} chars):")
        print("  " + new_text[:300].replace("\n", "\n  "))
        if len(new_text) > 300:
            print("  ...")

        print(f"  Embedding...")
        new_embedding = embed(new_text)

        collection.update(
            ids        = [chunk_id],
            embeddings = [new_embedding],
            documents  = [new_text],
            metadatas  = [{"section": "TROUBLESHOOTING", "page": page_num, "source": "vision"}],
        )
        print(f"  ChromaDB updated.")

    pdf_fitz.close()
    print("\nDone. BM25 index will rebuild automatically on next query.")


if __name__ == "__main__":
    pages = [int(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else [96, 97]
    main(pages)
