# Bike Troubleshooting Bot — Royal Enfield Interceptor 650

A RAG-based conversational assistant that answers maintenance and troubleshooting questions about the Royal Enfield Interceptor 650, grounded strictly in the official owner's manual. Users can type questions, upload photos of parts or warning lights, or record voice queries in English or Indian languages. The bot retrieves relevant passages from the manual, re-ranks them for relevance, and generates an answer — refusing to respond when the information is not in the manual.

## Live demo
[https://bike-troubleshooting-bot-nwc3nh3jxmnwdvovzswgxa.streamlit.app/](https://bike-troubleshooting-bot-nwc3nh3jxmnwdvovzswgxa.streamlit.app/)

## Video walkthrough
[Loom link — to be added]

## What this does

- **Text input** — ask any maintenance or troubleshooting question; answers are cited to specific manual sections and pages
- **Image upload** — attach a photo of a warning light, part, or symptom; the bot describes what it sees and uses that context alongside your question
- **Voice input** — record a question using the microphone; Sarvam Saaras V3 transcribes it automatically and routes it into the same pipeline
- **Multilingual support** — questions in Hindi, Kannada, Tamil, Telugu, and other Indic languages are detected and answered in the same language; GPT-4o responds in the user's language natively, Sarvam Saaras V3 handles Indic voice input
- **Grounded answers** — the bot only uses text from the manual; it will not draw on outside knowledge or speculate
- **Refusal behaviour** — out-of-scope questions (unrelated topics, information absent from the manual, multi-topic queries) receive explicit refusals rather than hallucinated answers

## Architecture

Hybrid retrieval (semantic search + BM25, fused via Reciprocal Rank Fusion) feeds a GPT-4o reranker that selects the top five manual passages; GPT-4o generates answers in the user's language for all queries, including Indic. Sarvam Saaras V3 handles multilingual voice transcription. See `decisions-log.md` for full architectural decisions and trade-off analysis.

## Example queries

| Mode | Query | Expected behaviour |
|---|---|---|
| Text | "What engine oil does the Interceptor 650 use and how often should I change it?" | Answer with oil spec, change interval, source pages |
| Image | Upload photo of instrument cluster warning light | Bot identifies the light and explains what it indicates |
| Voice | Record: "Tyre pressure kitna hona chahiye?" | Transcribed, detected as Indic, answered in Hindi |
| Multilingual | "இன்ஜின் ஆயில் எப்படி மாற்றுவது?" (Tamil) | Tamil refusal if retrieval fails; answer in Tamil if retrieved |
| Out-of-scope | "What is the capital of France?" | Explicit refusal: not in the Interceptor 650 manual |

## How to run locally

```bash
# 1. Clone the repo
git clone https://github.com/LavanyaPrabhat/bike-troubleshooting-bot.git
cd bike-troubleshooting-bot

# 2. Create a .env file with your API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and SARVAM_API_KEY

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The `chroma_db/` vector database is included in the repo — no ingest step is required.

## Built with

| Component | Role |
|---|---|
| [OpenAI GPT-4o](https://openai.com) | Query rewriting, re-ranking, answer generation (all languages), image description |
| [ChromaDB](https://www.trychroma.com) | Local vector store for manual embeddings |
| [text-embedding-3-small](https://openai.com) | Semantic search embeddings |
| [rank-bm25](https://github.com/dorianbrown/rank_bm25) | BM25 keyword search (hybrid retrieval) |
| [Sarvam Saaras V3](https://www.sarvam.ai) | Multilingual ASR — voice transcription (all languages) |
| [Streamlit](https://streamlit.io) | Frontend |
| [pdfplumber](https://github.com/jsvine/pdfplumber) + [PyMuPDF](https://pymupdf.readthedocs.io) | PDF text and image extraction (ingest only) |
| [langdetect](https://github.com/Mimino666/langdetect) | Offline language detection for routing |

## Author

Lavanya Prabhat
