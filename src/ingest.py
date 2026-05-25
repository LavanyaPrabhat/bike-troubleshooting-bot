import os
import sys
import base64
import pdfplumber
import fitz  # pymupdf
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL  = "text-embedding-3-small"
CHROMA_PATH  = "./chroma_db"
COLLECTION   = "bike_manual"
PDF_PATH     = "./data/royal-enfield-interceptor-650-owners-manual-english.pdf"

# A page is "image-heavy" if it has images AND fewer than 200 characters of text.
# Those pages get sent to GPT-4o Vision instead of plain text extraction.
IMAGE_TEXT_THRESHOLD = 200

VISION_PROMPT = (
    "Describe the technical content of this page in detail — include part names, "
    "measurements, specifications, warning notices, and any procedures shown. "
    "Be specific and technical."
)


# ── 1. PAGE CLASSIFICATION ─────────────────────────────────────────────────────

def is_image_heavy(fitz_page) -> bool:
    """
    Returns True if the page has at least one image AND very little text.
    These pages are diagrams, procedure illustrations, or spec charts that
    pdfplumber can't meaningfully extract.
    """
    if not fitz_page.get_images():
        return False
    text = fitz_page.get_text().strip()
    return len(text) < IMAGE_TEXT_THRESHOLD


# ── 2. VISION DESCRIPTION ──────────────────────────────────────────────────────

def describe_with_vision(fitz_page) -> str:
    """
    Rasterize the page at 150 DPI, encode as base64 PNG,
    and ask GPT-4o Vision to produce a technical description.
    """
    pix = fitz_page.get_pixmap(dpi=150)
    img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                        "detail": "high",
                    },
                },
                {
                    "type": "text",
                    "text": VISION_PROMPT,
                },
            ],
        }],
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


# ── 3. SECTION DETECTION ───────────────────────────────────────────────────────

def detect_section(text: str) -> str | None:
    """Return a section heading if the first few lines contain one (ALL CAPS, short)."""
    for line in text.split("\n")[:6]:
        line = line.strip()
        if line and line.isupper() and 3 <= len(line) <= 60:
            return line
    return None


# ── 4. HYBRID PAGE EXTRACTION ──────────────────────────────────────────────────

def extract_pages(pdf_path: str) -> list[dict]:
    """
    Process every page with the right method:
      - Image-heavy pages  → GPT-4o Vision description
      - Text-heavy pages   → pdfplumber text extraction

    Each page produces exactly one text block (no overlap).
    Returns a list of {page, text, section, source} dicts.
    """
    pdf_fitz = fitz.open(pdf_path)
    total_pages = len(pdf_fitz)
    pages = []
    current_section = "General"
    vision_count = 0
    text_count = 0

    with pdfplumber.open(pdf_path) as pdf_plumber:
        for i in range(total_pages):
            fitz_page = pdf_fitz[i]
            page_num = i + 1

            if is_image_heavy(fitz_page):
                print(f"  Page {page_num:3d}/{total_pages} - image-heavy -> Vision")
                text = describe_with_vision(fitz_page)
                source = "vision"
                vision_count += 1
            else:
                raw = pdf_plumber.pages[i].extract_text()
                text = (raw or "").strip()
                source = "text"
                text_count += 1

            if not text:
                continue

            heading = detect_section(text)
            if heading:
                current_section = heading

            pages.append({
                "page":    page_num,
                "text":    text,
                "section": current_section,
                "source":  source,
            })

    pdf_fitz.close()
    print(f"\n  {text_count} text pages, {vision_count} vision pages")
    return pages


# ── 5. CHUNKING ────────────────────────────────────────────────────────────────

def build_chunks(pages: list[dict]) -> list[dict]:
    """
    Each page is already one self-contained unit — no sliding window needed.
    We just number them and pass them through.
    """
    return [
        {
            "text":        page["text"],
            "section":     page["section"],
            "page":        page["page"],
            "source":      page["source"],
            "chunk_index": i,
        }
        for i, page in enumerate(pages)
    ]


# ── 6. EMBEDDING ───────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed each chunk's text via OpenAI, batching 100 at a time."""
    texts = [c["text"] for c in chunks]
    batch_size = 100

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend([e.embedding for e in response.data])
        print(f"  Embedded {min(i + batch_size, len(texts))} / {len(texts)} chunks")

    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["embedding"] = embedding

    return chunks


# ── 7. CHROMADB STORAGE ────────────────────────────────────────────────────────

def store_in_chroma(chunks: list[dict]) -> None:
    """Persist chunks + embeddings to ChromaDB on disk. Safe to re-run."""
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        chroma.delete_collection(COLLECTION)
    except Exception:
        pass

    collection = chroma.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids        = [str(c["chunk_index"]) for c in chunks],
        embeddings = [c["embedding"]        for c in chunks],
        documents  = [c["text"]             for c in chunks],
        metadatas  = [{
            "section": c["section"],
            "page":    c["page"],
            "source":  c["source"],
        } for c in chunks],
    )

    print(f"\nStored {len(chunks)} chunks in ChromaDB at '{CHROMA_PATH}'")


# ── MAIN ───────────────────────────────────────────────────────────────────────

def ingest(pdf_path: str = PDF_PATH) -> None:
    print(f"Reading PDF: {pdf_path}\n")
    pages = extract_pages(pdf_path)
    print(f"\n{len(pages)} pages extracted\n")

    print("Building chunks (1 per page)...")
    chunks = build_chunks(pages)
    print(f"{len(chunks)} chunks ready\n")

    print("Embedding chunks (calling OpenAI)...")
    chunks = embed_chunks(chunks)

    print("\nStoring in ChromaDB...")
    store_in_chroma(chunks)
    print("\nDone! The manual is indexed and ready to query.")


if __name__ == "__main__":
    ingest()
