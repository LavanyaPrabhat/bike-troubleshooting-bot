# Architecture Decision Log

Decisions made during build, with the reasoning behind each. Useful for interviews and future reference.

---

## 1. Hybrid PDF Extraction: pdfplumber + GPT-4o Vision

**Decision:** Use PyMuPDF to classify each page. Pages with images AND fewer than 200 characters of extracted text are sent to GPT-4o Vision for description. All other pages use pdfplumber text extraction.

**Why:** A motorcycle owner's manual is heavily visual — diagrams, torque spec charts, procedure illustrations. pdfplumber only reads machine-readable text, so image-dominant pages would produce empty or near-empty chunks, making them unsearchable. GPT-4o Vision reads the page as a human would and produces a rich technical description (part names, measurements, procedure steps) that can be embedded and retrieved like any text chunk.

**Trade-off:** Vision calls cost ~$0.02–0.03 per page and add latency during ingest. Acceptable because ingest is a one-time operation. At runtime, Vision is only used for user-uploaded images, not for retrieval.

**Threshold chosen:** 200 characters. Pages below this with images are overwhelmingly diagram pages. Pages above it have enough text for pdfplumber to be useful even if diagrams are present.

---

## 2. One Chunk Per Page (replacing sliding-window chunking)

**Decision:** Each page produces exactly one chunk. No sliding window, no overlap.

**Why:** The original plan (800–1200 token chunks with 100-token overlap) was designed for a dense text corpus. The actual manual turned out to be ~32,000 tokens across 130 pages — roughly 246 tokens per page on average. Running a 1000-token sliding window over this produced only 30 very coarse chunks, each spanning ~4 pages and mixing multiple topics. This killed retrieval quality (best similarity score: 0.45).

Switching to one-chunk-per-page gave 130 focused chunks. Best similarity score on the same test query jumped to 0.589. Pages are already natural semantic units in a manual — each page covers a specific topic or procedure.

**Trade-off:** No cross-page overlap means a procedure that spans two pages could be split. In practice this is rare in this manual, and the top-K retrieval (returning up to 5 chunks) mitigates it by often returning adjacent pages.

---

## 3. Similarity Threshold: 0.7 → 0.4

**Decision:** Lower the cosine similarity cutoff from 0.7 to 0.4.

**Why:** The 0.7 threshold was set based on general RAG guidance, but it assumes a dense, well-matched corpus. With text-embedding-3-small on this specific manual, the highest similarity score observed was 0.589 — meaning a 0.7 threshold would reject every single result, including highly relevant ones. 0.4 was chosen empirically: it passes clearly relevant chunks while still filtering out weak matches. Anything below 0.4 on this corpus is genuinely unrelated.

**Interview note:** This is a calibration decision — the "right" threshold depends on the model, the corpus, and the query distribution. The right approach in production is to test a sample of known good and bad queries and pick the threshold that maximises precision at acceptable recall.

---

## 4. Vision Module: `detail: "low"`, `temperature: 0.2`, `max_tokens: 150`

**Decision:** User-uploaded images are processed with GPT-4o Vision at low detail, temperature 0.2, and capped at 150 tokens.

**Why (detail: low):** During ingest we use `detail: high` because we need to read fine print, measurements, and spec tables from the manual. For user-uploaded symptom photos (a warning light, a leaking hose, a worn tyre), we only need to identify and name the condition — not read text within the image. Low detail is sufficient for object/condition identification and cuts Vision token cost by ~6x per user query.

**Why (temperature: 0.2):** The symptom description feeds directly into a semantic search query. Creative or varied phrasing would make retrieval inconsistent across similar images. Low temperature produces stable, factual, repeatable output.

**Why (max_tokens: 150):** The symptom description is a search enrichment, not an answer. It needs to be specific but short — 1–3 sentences. Capping at 150 tokens prevents verbose output that would dilute the search signal.

---

## 5. Embedding Model: text-embedding-3-small

**Decision:** Use OpenAI's `text-embedding-3-small` for both indexing and query embedding.

**Why:** Cheaper and faster than `text-embedding-3-large` while producing strong retrieval quality for English technical text. Both the manual chunks and the user queries must be embedded with the same model — mixing models would break similarity comparisons. Small is sufficient for a single-domain corpus like this.

---

## 6. ChromaDB with Cosine Distance

**Decision:** Use ChromaDB as the local vector store with `hnsw:space: cosine`.

**Why:** ChromaDB is file-based, requires no server, and is easy to set up for a prototype. It persists to disk so the index survives restarts without re-embedding. Cosine distance (rather than L2/Euclidean) is the right metric for text embeddings — it measures the angle between vectors, which captures semantic similarity regardless of text length.

**Note on distances:** ChromaDB returns cosine *distance* (0 = identical, 1 = unrelated), not similarity. We convert with `similarity = 1 - distance` before applying the threshold.

---

## 7. Two-Layer Grounding in generator.py

**Decision:** Grounding is enforced at two independent layers, not one.
- Layer 1: strict system prompt — GPT-4o is told never to use outside knowledge and to return a canned response if the excerpts are insufficient.
- Layer 2: reranker threshold — if the top reranker score is below 6, `rerank()` returns an empty list and `generate_answer()` returns `NO_CONTEXT_RESPONSE` without calling GPT-4o at all. *(Originally implemented as a cosine similarity cutoff of 0.4 in the retriever; superseded by the reranker threshold when hybrid retrieval was introduced — see decisions #11, #12.)*

**Why:** A system prompt alone is not reliable — GPT-4o can still hallucinate if given weak context. The reranker threshold ensures GPT-4o only gets called when there is genuinely relevant content to anchor its answer. The two layers are complementary: the threshold handles off-topic queries, the system prompt handles edge cases where on-topic queries surface marginally relevant chunks.

**Trade-off:** The threshold can produce false negatives — a valid question the manual does cover might get the canned response if the query phrasing doesn't match well. This is preferable to a false positive (a confident-sounding wrong answer), which damages user trust more severely.

**Interview note:** This is a deliberate product decision, not just a technical one. For a safety-relevant domain like vehicle maintenance, a conservative "I don't know" is always better than a hallucinated procedure.

---

## 8. Generator Temperature: 0.3

**Decision:** GPT-4o generation uses `temperature=0.3`, not 0.

**Why:** Temperature 0 produces deterministic output but can feel robotic and occasionally gets stuck in repetitive phrasing. 0.3 keeps answers factual and consistent while allowing natural sentence variation. This is higher than vision.py's 0.2 because the generator is writing prose for a human to read, whereas vision.py is producing a search query fragment.

**Trade-off:** Slightly less deterministic than 0 — two runs on the same input may phrase the answer differently. Acceptable for a support bot; would be inappropriate for a system that needs byte-for-byte reproducibility.

---

## 9. Canned No-Context Response Bypasses GPT-4o

**Decision:** When `chunks` is empty, `generate_answer()` returns a hardcoded string and never calls the OpenAI API.

**Why:** Cost and correctness. If the retriever found nothing relevant, any answer GPT-4o produces would be drawn from training data, not the manual. Skipping the API call saves money and guarantees the response is grounded. The canned message directs the user to an authorised service centre — the right escalation path for a question the bot can't answer.

---

## 10. Streamlit UI: Sidebar Image Upload, Centered Layout, Per-Step Spinners

**Decision:** Image upload lives in the sidebar; the main area is a standard chat interface. Four labelled spinners show each pipeline stage: "Analysing your image", "Searching the manual", "Re-ranking results", "Writing answer". *(Originally three spinners; a fourth was added when the reranker was introduced.)*

**Why (sidebar):** Keeps the chat area uncluttered. The image is an optional input, not the primary interaction. Sidebar placement also lets the preview persist visually while the user types their question.

**Why (per-step spinners):** A single "Loading..." spinner gives no feedback on a pipeline with three distinct API calls (Vision, embedding, generation), each of which can take 1–3 seconds. Labelled spinners tell the user what is happening and make the pipeline legible — useful for a demo where the evaluator should see each stage.

**Why (centered layout):** Chat UIs feel cramped at full browser width. Centered layout with Streamlit's default max-width reads comfortably as a Q&A interface.

**Trade-off:** Session state stores full chat history in memory. For a long session with many image uploads this could grow large, but for a demo/prototype this is fine. A production version would paginate or summarise history.

---

## 11. Hybrid Retrieval (Semantic + BM25 + RRF) Instead of Semantic-Only

**Decision:** Replace pure cosine similarity retrieval with a hybrid pipeline: semantic search (ChromaDB) + BM25 keyword search (rank-bm25), fused via Reciprocal Rank Fusion (k=60).

**Why:** Semantic embeddings are averages over the full chunk. A spec-sheet page containing 30 different specs (fuel type, tyre size, oil grade, etc.) embeds as "technical motorcycle specs in general" — its cosine similarity for any single specific query is diluted across all 30 topics. BM25 is blind to meaning but rewards exact keyword presence, so it rescues spec-sheet chunks that semantically dilute. RRF combines both ranked lists without requiring score normalisation — it only cares about rank order, which makes it robust when the two scoring scales (cosine 0-1, BM25 0-∞) are incomparable.

**Why RRF over alternatives like score normalisation or query expansion:**
- Score normalisation requires knowing the score distribution in advance, which changes with the corpus.
- Query expansion (rewriting the query before embedding) adds an LLM call but doesn't fix BM25-frequency bias — a maintenance-schedule page that mentions "fuel" 10 times still outranks a spec page that says "Fuel type: Unleaded gasoline" once.
- RRF is parameter-light (just k=60, a community-standard default) and well-studied in the IR literature.

**Trade-off:** Two retrieval paths instead of one adds ~50ms latency per query (BM25 is in-memory so negligible; the extra overhead is the wider ChromaDB fetch). Code complexity increases moderately. Accepted because retrieval quality is the core correctness requirement of the system.

---

## 12. LLM Re-ranker (GPT-4o) After Hybrid Retrieval

**Decision:** After hybrid retrieval returns 20 candidates, a second GPT-4o call scores each candidate 0–10 for direct relevance to the query. Top 5 are returned. If the top score is below 6, the pipeline returns an empty list and the generator issues the canned refusal.

**Why re-ranking solves what BM25 frequency cannot:**
BM25 rewards raw keyword frequency. A periodic maintenance schedule page that mentions "fuel level check" six times outranks a single-line spec "Fuel type: Unleaded gasoline" by BM25. The reranker reads both in context with the query and correctly scores the spec page 10/10 and the maintenance schedule 1/10. This is something neither cosine similarity nor BM25 can do — it requires language understanding, not statistical matching.

**Why GPT-4o for re-ranking instead of a dedicated cross-encoder (e.g. ms-marco-MiniLM):**
- A cross-encoder would require a separate model download (~500MB) and inference setup.
- GPT-4o is already in the dependency stack. One extra call with truncated excerpts costs ~$0.005–0.01 and adds ~500ms.
- For a prototype on a deadline, using what's already wired in is the right call. Production would swap to a local cross-encoder for cost and latency.

**Threshold: top rerank score < 6 → refusal:**
A score of 6 means "topically related but doesn't answer the question." We only want to answer when the manual actually contains the answer, so 6 is the correct floor. Tested: "what is the capital of France" gets a top rerank score of 0 and correctly refuses.

**Trade-off:** One extra LLM call per user query (~500ms, ~$0.005–0.01). The reranker call uses `temperature=0` for deterministic, consistent scoring. Accepted because it eliminates the class of false-positives (confident-sounding wrong answers) that undermine user trust.

**Interview note:** This is the production-grade pattern in hybrid RAG pipelines — retrieve broadly, re-rank precisely. The canonical stack is BM25 + dense retrieval + cross-encoder reranker. We're using GPT-4o as the cross-encoder, which is slightly expensive but architecturally identical.

---

## 13. Reranker Threshold Is Binary on the Top Result, Not Per-Chunk

**Decision:** The reranker's pass/fail check is applied only to the top-scoring chunk. If chunk #1 scores ≥ 6, all top-5 chunks are passed to the generator — including any that scored 0.

**Why:** The refusal threshold exists to answer one question: "does this corpus contain a useful answer at all?" That's a batch-level question, best answered by the strongest match. A low score on chunks #4 and #5 doesn't mean the query is unanswerable — it means those chunks are less useful. Filtering them individually would require a second threshold to tune and could silently drop a chunk the reranker underscored.

The generator's system prompt already instructs GPT-4o to use only relevant excerpts. It will naturally ignore 0-scoring chunks when composing the answer — it has the full question and all chunks simultaneously, making it better placed to judge relevance than a blind numeric cutoff.

**Alternative considered:** A per-chunk floor (e.g. drop any chunk scoring < 2). This would reduce prompt size by ~400–800 tokens per dropped chunk. Rejected because the reranker is not infallible — a useful chunk could be underscored — and the generator handles the redundancy gracefully at negligible extra cost.

**Trade-off:** Low-scoring chunks add a small amount of noise to the generation prompt. Accepted: the generator filters them implicitly, and keeping them avoids the risk of silently losing a chunk that contains a partial but useful answer.

---

## 14. Vision Description Is Used Twice: Search Enrichment + Generator Context

**Decision:** When a user uploads an image, the Vision description is used in two places:
1. Appended to the text query before retrieval: `combined_query = f"{prompt}. {vision_description}"`
2. Passed separately to `generate_answer()` as a `vision_description` argument, where it appears in the generation prompt as a labelled "VISUAL CONTEXT" block.

**Why:** These are two different jobs. For retrieval, the description enriches the query so that semantically relevant manual pages surface — a photo of a warning light should retrieve chunks about warning indicators, even if the user typed nothing. For generation, the description gives GPT-4o the visual context it needs to frame the answer correctly (e.g. "based on the oil pressure light you're seeing, here's what to check..."). Using only the enriched query for generation would bury the visual signal inside a long string; using a dedicated block makes it unambiguous.

**Trade-off:** The description is sent to the API twice per query (once embedded for retrieval, once as text for generation). The token cost is negligible (~150 tokens). The alternative — using only the text query for retrieval and relying on the user to describe the image in words — degrades retrieval quality for users who upload an image without typing much.

---

## 15. Reranker Receives 400-Character Excerpts, Not Full Chunks

**Decision:** Before scoring, each chunk is truncated to its first 400 characters. The reranker sees `chunk["text"][:400]`, not the full page text.

**Why:** The reranker's job is relevance classification, not information extraction. The first 400 characters of a page-sized chunk (typically a section heading, subheading, and first few lines) contain enough signal to judge whether the page is on-topic. Sending the full chunk would roughly triple the input token count for the reranker call with no meaningful improvement in scoring quality.

**Trade-off:** A page where the relevant content appears late (e.g. a caution notice buried in the middle of a procedure page) could be underscored. Accepted: section headings and opening lines are reliable topical signals in a structured manual, and the generator receives the full chunk text regardless of how the reranker scored it.

---

## 16. SEMANTIC_FETCH = 20 and CANDIDATES = 20

**Decision:** The semantic search fetches 20 results from ChromaDB, and the reranker receives all 20 as candidates.

**Why:** The reranker is the quality gate, so the retriever's job is to cast a wide enough net that the right chunk is always present. With 130 chunks in the index, fetching 20 (15% of the corpus) gives the reranker strong coverage without including chunks so distant that they add noise. Fetching 5 (the original TOP_K) was insufficient — it missed the Technical Specifications page for the fuel query entirely.

**Why not fetch more (e.g. 40 or 50):** The reranker prompt grows linearly with candidates. At 400 chars per excerpt, 20 candidates ≈ 8,000 characters of input. 50 candidates would push toward 20,000 characters, materially increasing cost and latency with diminishing returns on a 130-chunk index.

**Trade-off:** If the correct chunk falls outside the top-20 by RRF score, the reranker cannot rescue it. With 130 total chunks, the probability of this for a well-formed query is very low. Accepted for this corpus size.

---

## 17. Query Rewriting for False-Premise Queries

**Decision:** Add a `rewrite_query()` step before retrieval. A lightweight GPT-4o call (temperature=0, max_tokens=80) neutralises false assertions in user queries and returns a neutral lookup question. The rewritten query goes to retrieval and reranking; the original query goes to the generator.

**Why:** False-premise queries ("The manual says to use 98 octane fuel right?", "Since the Interceptor uses diesel...") fail retrieval because their embedding matches the assertion rather than the underlying information need. Rewriting "The manual says to use 98 octane fuel right?" to "What fuel type does the Interceptor 650 require?" produces an embedding that retrieves the fuel spec page at 10/10 reranker score.

**Why the original query must still reach the generator:** The generator needs to see the false premise to correct it explicitly. If the generator only sees the rewritten query, it will give a neutral answer ("The Interceptor uses unleaded gasoline") without addressing the user's misconception. Rule 6 was added to the generator system prompt: "If the user's question contains a false assumption that contradicts information in the excerpts, explicitly correct the false assumption first." The dual-path design — rewritten to retrieval, original to generation — allows both correct retrieval and explicit premise correction in the answer.

**Alternatives considered:**
- Lower the rerank threshold from 6 to 5: Would not fix retrieval. The fuel spec page wouldn't surface for confirmation-style queries even with a lower threshold, because the query embedding is misaligned regardless of threshold.
- Accept refusal: A bot that says "I can't find that" to "The manual says use 98 octane, right?" would be actively unhelpful — the user has a wrong belief the manual could correct. Refusing is worse than answering.

**Cost/latency trade-off:** One extra GPT-4o call per query (~$0.001, ~200ms). The rewriter call is a small text completion (80 tokens max output), much cheaper than the reranker call. For the large majority of queries that have no false premise, the function returns the query unchanged (but still makes the API call). Accepted for a prototype; production would add a fast local classifier to detect false-premise queries before calling GPT-4o.

**Interview note:** This is the RAG "pre-retrieval" pattern — query transformation before embedding. In production systems this often includes query expansion, HyDE (hypothetical document embeddings), and decomposition. Our implementation is the simplest version: one-shot reformulation.

---

## 18. Long Query Guard at 75 Tokens (UX-Level, Not Architectural)

**Decision:** Queries longer than 75 tokens are rejected before retrieval with the message: "Your question covers multiple issues at once. Please ask about one symptom or topic at a time so I can give you a precise answer from the manual."

**Why 75 tokens:** The failing test case (4-symptom query covering engine knock, cam chain, rough idle, and burning smell) was 80 tokens. A detailed but single-topic question tops out around 50–70 tokens for this domain. 75 is the boundary between "detailed single-topic" and "multi-symptom essay." Anything above this reliably embeds as a diffuse centroid that scores below reranker threshold on all individual topics.

**Why a UX-level guard rather than architectural query decomposition:**
- Query decomposition (splitting the query, running parallel retrievals, merging results) would handle the underlying technical problem.
- For a demo prototype, decomposition adds: at least one extra LLM call for splitting, parallel retrieval calls, result merging logic, and a new failure mode (poorly split queries).
- The UX-level guard achieves the same outcome (user gets a useful answer) at zero added complexity: the user asks about one symptom, gets a precise answer from the manual.
- User education is more durable than query decomposition — a user who learns to ask focused questions will have better experiences across all query types.

**Decision NOT to build query decomposition:**
Production system would handle this architecturally. For this prototype: the guard + message is the right call. Documented deliberately so the architectural path is clear in an interview discussion.

**Alternatives considered:**
- Architectural decomposition: Correct production approach. Rejected for this demo scope.
- LLM-based symptom extraction: Extract the primary symptom from a long query before embedding. Adds cost and complexity similar to decomposition, with the same scope objection.
- Accept the failure: A silent "I couldn't find that" for a 4-symptom query is worse UX than telling the user why and how to improve their question.

**Trade-off:** A user asking a multi-symptom question receives guidance instead of an answer. Acceptable — the guidance is actionable and accurate.

**Interview note:** The 75-token threshold would be tuned based on production query distribution data. In a live system, you'd sample real queries, measure embedding dilution vs. query length, and set the threshold at the inflection point.

---

## 19. Reranker Prompt Update for Symptom/Diagnostic Queries

**Decision:** Updated `RERANKER_SYSTEM` scoring guide to explicitly score diagnostic/troubleshooting pages at 7–9 when the query is a symptom description and the page lists causes for that symptom.

**Why:** The original scoring guide defined 4–6 as "topically related but does not actually answer the question." The reranker applied this literally to troubleshooting pages: a diagnostic table that lists "possible causes for white smoke" was scored 4/10 because it doesn't give a single direct answer. The reranker's strict interpretation was correct by the guide's wording but wrong for diagnostic queries, where a list of possible causes IS the correct answer.

**The insight:** For specification queries ("what fuel type?"), a page that mentions fuel without specifying the type should score low — the user needs a number or a name. For symptom queries ("why is there white smoke?"), a page that lists causes for that symptom is the answer — the reranker should score it 7–9. These are different question types that require different scoring logic. The updated guide adds this distinction explicitly.

**Connection to the assignment example:** "White smoke from exhaust" is the literal example in the interviewer's question. This fix directly targets that case. The score improved from 4/10 to 7/10 after the prompt update. The residual failure (GPT-4o's grounding check) is a content gap in the ingest layer, not a retrieval issue.

**Why a prompt change beats lowering the threshold globally:**
- Lowering `RERANK_PASS_SCORE` from 6 to 5 would pass all topically-related pages, including ones that are genuinely off-topic. The reranker would lose its discriminative power.
- The prompt change teaches the reranker to distinguish query types, which improves precision where it matters (diagnostic queries) without degrading it elsewhere.

**Trade-off:** The reranker prompt is now longer and more complex. The distinction between "specification" and "symptom" queries requires the model to classify the query type while also scoring relevance. At temperature=0, GPT-4o is reliable at this dual task.

---

## 20. Vision Image Cache Keyed by (Filename, Filesize)

**Decision:** Vision descriptions are cached in `st.session_state.vision_cache` keyed by `(uploaded_file.name, uploaded_file.size)`. If the same file is uploaded again, the cached description is reused without calling the Vision API.

**Why not file hash:** Computing a cryptographic hash of image bytes (e.g. SHA-256) would be more collision-resistant but requires reading all bytes on every rerun. Streamlit reruns on every user interaction — hashing a large image on every rerun adds latency and CPU cost for zero benefit in practice. Two different images with the same name and size are extremely unlikely in normal use; the name+size key is sufficient for this use case.

**UX consequence:** The image description persists and is used for all follow-up queries in a session until the user changes the file or clicks Clear. This is intentional — if a user uploads a photo of a warning light and asks three follow-up questions about it, the visual context should inform all three answers. The tradeoff is that a user who leaves an old image uploaded without noticing will have their text queries augmented with stale visual context. Clear conversation resets the cache.

**Cost savings:** Without caching, every query in a session with an uploaded image triggers a Vision API call (~$0.002–0.005 each). Caching ensures Vision is called exactly once per unique image per session.

---

## 21. Multilingual Input: Documented Feature, Not Emergent Surprise

**Finding from audit (EC06):** A query in Hindi ("इंजन ऑयल कैसे चेक करें" — how to check engine oil) was correctly answered in Hindi, grounded in the manual. This was initially expected to be a refusal.

**Why it works:** `text-embedding-3-small` is a multilingual model — it encodes semantic meaning across languages into the same embedding space. A Hindi query about engine oil is close to the English engine-oil chunks in embedding space. GPT-4o generates answers in the query language by default. The pipeline is therefore language-agnostic on both input and output without any explicit routing.

**Decision:** Designate this as a supported feature, not a limitation. The system handles non-English queries without additional engineering.

**Architectural implication for Level 3:** This finding simplifies the planned multilingual architecture. A language-detection and routing step (e.g., Sarvam-M for Indian regional language queries) is not required for the core QA pipeline. Language routing would only be needed if: (a) we want language-specific answer formatting, or (b) we need to handle scripts where `text-embedding-3-small` performs poorly. For the current scope, the pipeline handles Hindi, and likely other major languages, natively.

**Interview note:** This is a benefit of using foundation models that were trained on multilingual corpora. The semantic space learned during pre-training aligns concepts across languages, so cross-lingual retrieval works without explicit translation steps. This is why RAG on multilingual corpora doesn't require a translate-then-retrieve pipeline in many cases.

---

## 22. Inline Multimodal Input Bar (Bottom of Chat, Not Sidebar)

**Decision:** Voice and image input live in a row at the bottom of the main chat area — `[📎 attach] [text area] [↑ send]` with the audio widget directly above — not in the sidebar.

**Why:** Modern AI chat interfaces (Claude.ai, ChatGPT, WhatsApp) place all input modes in a single bottom bar. The sidebar pattern we started with works for feature-limited demos but breaks the mental model users already have: "the bottom of the screen is where I compose my message." Sidebar inputs feel like settings, not primary inputs. Moving voice and image inline makes multimodal feel like a natural extension of the text input, not a separate mode.

**Why the sidebar is now minimal:** With both optional inputs moved inline, the sidebar would contain only organisational chrome. A near-empty sidebar wastes screen real estate and draws attention away from the conversation. Collapsing it by default focuses the user on the chat.

**Trade-off:** Losing the persistent sidebar image preview — the image is now shown as a small thumbnail above the input row. The thumbnail is less prominent than a full sidebar preview but is sufficient for a demo. A user who forgets an image is attached will see the thumbnail on their next query.

---

## 23. Transcription Pre-fills the Text Area — No Auto-Submit

**Decision:** When voice transcription completes, the transcript is placed into the text area (`st.session_state.text_input`) for the user to review and optionally edit. The pipeline does not run automatically. The user clicks ↑ Send to submit.

**Why:** Auto-submit on voice is the right default for a hands-free interface (e.g. a voice assistant). It is wrong for a multimodal chat bot where:
1. The user may want to add clarifying text to the transcription ("...and what page is this on?").
2. Sarvam Saaras V3 is accurate but not infallible — a transcription error mid-sentence could send a garbled query without the user noticing.
3. The user may want to combine a voice-transcribed question with an uploaded image — seeing the transcript first lets them confirm the question before attaching visual context.

Showing the transcript in the same input box also makes the voice path feel identical to the text path, not like a separate "voice mode."

**Streamlit implementation note:** `st.text_area(key="text_input")` is pre-filled by setting `st.session_state.text_input = transcript` before `st.rerun()`. Streamlit reads widget state from session state on render, so this correctly populates the visible text area.

---

## 24. Audio Widget Key Reset Fixes Post-Transcription Error

**Decision:** After a voice recording is transcribed, `st.session_state.audio_key` is incremented and `st.rerun()` is called. The audio widget's `key` parameter is `f"audio_{st.session_state.audio_key}"`, so incrementing the key causes Streamlit to mount a completely fresh widget instance on the next render.

**Why:** `st.audio_input` returns an `UploadedFile`-like object. After `audio_recording.read()` consumes the byte stream, the next Streamlit rerun (triggered by the pipeline completing) finds the same `audio_recording` object with the file pointer at the end. `read()` returns `b""`, whose MD5 hash differs from `last_audio_hash` (the hash of the actual audio bytes), so the dedup guard fails and we attempt transcription of an empty stream. Concurrently, Streamlit's internal widget state machine detects the inconsistency between the stored widget value and the consumed stream, showing "An error has occurred, please try again." in the browser.

Incrementing the key solves both problems in one `st.rerun()`: the new widget is empty (no recording), so `audio_recording is None` and neither code path runs. The old error state is discarded. A `seek(0)` guard is also added before `read()` as defence-in-depth against the consumed-pointer issue on any rerun that slips through before the key increment.

**Alternative considered:** Store a hash in `last_audio_hash` and rely on it to skip reprocessing. This prevents the double-transcription but does not fix the widget's internal error state — the "An error has occurred" UI still appears. Key reset is the correct fix because it eliminates the problematic widget instance entirely.

---

## 25. Sarvam-105b for Indic Language Generation

**Decision:** Route queries detected as Indic to Sarvam-105b (via `https://api.sarvam.ai/v1/chat/completions`) rather than GPT-4o.

**Why Sarvam-105b and not GPT-4o:** This is a Sarvam AI take-home assignment. Using a Sarvam model for Indic generation is the architecturally correct and interview-appropriate choice — it demonstrates that the system is built with purpose, not just assembled from the first available tool. Sarvam-105b is purpose-built for Indian languages with training on Indic corpora; GPT-4o handles Indic text but is not optimised for it.

**Why Sarvam-105b and not Sarvam-M (sarvam-m):** The Sarvam docs list `sarvam-m` as a legacy 24B model and recommend `sarvam-105b` for new workloads. Using the current recommended model is a stronger signal than using a legacy one. Both accept the same OpenAI-compatible API format, so the swap is a one-line change if needed.

**Integration approach:** The Sarvam chat completions API is OpenAI-compatible — it accepts the same `messages`, `model`, `temperature`, `max_tokens` parameters. We reuse the `openai` Python SDK with `base_url="https://api.sarvam.ai/v1"` and `SARVAM_API_KEY`. No additional SDK is needed. The Sarvam client is lazy-initialised so `SARVAM_API_KEY` is only required when an Indic query is actually processed.

**Same system prompt for both models:** GPT-4o and Sarvam-105b share a single `SYSTEM_PROMPT`. Rule 7 ("Respond in the same language the user's question is written in") was added at Level 3. Both models follow it, so one prompt covers both paths without duplication.

---

## 26. No Explicit Translation Step — Cross-Lingual Embeddings Are Sufficient

**Decision:** Retrieval always runs in English regardless of the query language. No translate-then-retrieve step is added.

**Why:** EC06 (Level 1 audit) confirmed that `text-embedding-3-small` encodes semantic meaning across languages into a shared embedding space. A Hindi query about engine oil lands close to the English engine-oil chunks in that space. This is emergent behaviour from the model's multilingual pre-training — the same semantic concept aligns across languages even without explicit translation. Adding an explicit translation step (translate query to English → embed → retrieve) would add one LLM call per query, ~200ms, and ~$0.001 in cost, for no retrieval quality gain.

**Interview note:** This is the standard pattern for multilingual RAG on multilingual embedding models. Translation pipelines are necessary only when the embedding model was trained on a single language. With foundation-model embeddings, translate-then-retrieve is usually unnecessary overhead.

---

## 27. Script Detection + langdetect for Language Routing; GPT-4o Fallback for Hinglish and Tanglish

**Decision:** Language detection uses two stages: (1) Unicode script range check for native-script queries; (2) `langdetect` ISO-639-1 code check for Latin-script queries. Queries that are not positively identified as Indic fall through to `"english"` and are routed to GPT-4o.

**Why two stages:**
- Native-script queries (Devanagari for Hindi, Tamil script, Telugu script, etc.) are unambiguous: any Indic Unicode character guarantees the query is Indic. Script detection is instant, requires no model, and is 100% reliable for this case.
- Latin-script queries (Hinglish like "Meri bike ka...", Tanglish like "Tyres pressure enna irukkanam?") cannot be identified by script alone. `langdetect` is used as a second pass.

**Known limitation — Hinglish and Tanglish misclassification (confirmed in testing):**
`langdetect` misclassifies code-mixed Latin-script Indic text as unrelated languages. In testing, "Meri bike ka engine oil kab change karna chahiye?" was classified as `"sw"` (Swahili), not `"hi"`. Similarly, Latin-script Tanglish falls through to `"english"`. These queries are routed to GPT-4o instead of Sarvam-105b.

**Why this is acceptable:** EC06 (Level 1 audit) explicitly confirmed that GPT-4o answers Indic queries correctly, grounded in the manual, and responds in the query's language. The fallback to GPT-4o is not a degraded experience — it is the same quality pipeline that was already working before Level 3. Sarvam-105b is preferred for Indic generation (interview signal, purpose-built model), but GPT-4o is a correct and validated fallback when detection fails.

**Why `langdetect` over a more sophisticated approach:**
- A word-list approach (checking for common Hindi/Urdu tokens in Latin script) would need to cover thousands of words, handle spelling variation, and still fail on Tanglish.
- A separate language-identification model (e.g. fastText lid.176) would be more accurate but adds a model download and inference dependency.
- For this prototype, the combination of script detection (handles the most common real-world case — users typing in their native script) + langdetect + GPT-4o fallback is the right scope. Production would use a dedicated LID model.

**`DetectorFactory.seed = 0`:** Set at import time to make `langdetect` deterministic. Without it, `langdetect` uses internal randomness and can return different codes for the same input across reruns.

---

## 28. Dynamic Refusal and Guard Messages for Indic Queries

**Decision:** When a query is classified as Indic, the canned refusal message ("I couldn't find that in the manual…") and the multi-topic guard message ("Your question covers multiple issues…") are generated dynamically via a short Sarvam-105b call (max 120 tokens), rather than returning a hardcoded English string.

**Why:** A hardcoded English refusal delivered to a user who asked in Hindi or Tamil is a UX failure — the user has to parse a response in a language they may not have queried in. Generating the refusal in the user's language takes one short API call and produces a response that is natural and consistent with the rest of the interaction.

**Why not hardcode translations for common languages:** Static strings require a maintained translation map. Adding a new language would require a new entry. The dynamic approach scales to any language Sarvam-105b supports without code changes.

**Cost/latency trade-off:** A refusal call is capped at 120 tokens and costs <$0.001 and ~300–500ms. This only fires when the retriever found nothing relevant (already a failed path) or the query is too long — so it does not add latency to the happy path.

---

## 29. Switch from sarvam-105b to sarvam-m: Reasoning Mode Discovery and Crash Resolution

**Decision:** Replace `sarvam-105b` with `sarvam-m` as the Indic generation model. Increase `max_tokens` for generation from 600 to 2048. Add `_strip_think()` to remove embedded reasoning blocks. Centralise all Sarvam API calls through a single `_call_sarvam()` helper.

**Root cause discovered:** Both `sarvam-105b` and `sarvam-m` run in reasoning/thinking mode by default — the model internally reasons through the problem before generating its answer. The two models expose this reasoning differently:
- `sarvam-105b`: reasoning trace goes to a separate `reasoning_content` field; `content` holds only the final answer. At `max_tokens=600`, the reasoning trace consumed all tokens, leaving `content=None`. Calling `.strip()` on `None` crashed every Indic code path (13 of 44 audit cases).
- `sarvam-m`: reasoning trace is embedded inline inside the `content` field as `<think>...</think>` blocks. Content is never `None`. This makes the crash impossible to reproduce on sarvam-m.

**Why switch to sarvam-m instead of fixing sarvam-105b:** The sarvam-105b `content=None` crash is triggered by token exhaustion in the reasoning trace, not by a bug in our code. Fixing it would require either (a) dramatically increasing `max_tokens` beyond what sarvam-105b allows for its reasoning budget, or (b) switching off reasoning mode (unsupported). sarvam-m exposes the same reasoning capability with a safer content structure. The switch is a one-field change (`model="sarvam-m"`); the rest of the API call is identical.

*Note: Decision #25 originally chose sarvam-105b as the production-recommended model. This decision supersedes that choice based on empirical behaviour discovered in the Level 2+3 audit.*

**`max_tokens` rationale (600 → 2048):** The 600-token limit was sized for a final answer, not a reasoning trace + answer. With sarvam-m, the `<think>` block can be 500–1000 tokens before the actual answer begins. At max_tokens=600, the model was truncating mid-think (producing dangling `<think>` tags) before ever reaching the answer. 2048 gives the model room for a full reasoning trace plus a complete answer. For refusal and guard messages (much shorter), 1024 is sufficient.

**`_strip_think()` implementation:** Uses `re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)` to remove complete think blocks. A second pass checks for a dangling `<think>` (no closing tag — meaning max_tokens still cut off mid-think) and truncates everything from it: `cleaned[:cleaned.index("<think>")]`. The None guard `content or ""` in `_call_sarvam()` is defence-in-depth against any future content=None scenario.

**`_call_sarvam()` centralisation:** Before this fix, each Sarvam call was written independently with its own None guard and think-stripping. Centralising removes the risk of forgetting either guard on a new call path. All five Sarvam call sites in `generator.py` (main generation, refusal, multi-topic, guard, translation) now go through one function.

**Trade-off:** sarvam-m is 24B parameters vs sarvam-105b at 105B. In practice, for the constrained task of answering bike manual questions (heavily grounded by the system prompt) or translating short canned messages, 24B is sufficient. No answer quality regression was observed in the 13 post-fix test cases.

---

## 30. Section-Diversity Heuristic for Distinguishing Multi-Topic Dilution from Genuine Content Gaps

**Decision:** After the reranker returns an empty list (no chunk scored ≥ 6), determine whether the failure is (a) a multi-topic query that diluted retrieval or (b) a genuinely off-topic query. Use cosine similarity scores and section diversity from the already-computed candidates — no extra API call.

**Why the distinction matters:** An empty reranker result was previously always treated as "not in manual — return canned refusal." But two very different query types produce empty reranker results:
1. *Genuinely off-topic:* "What is the capital of France?" — the corpus has near-zero overlap with the query. No useful answer exists; the canned refusal is correct.
2. *Multi-topic dilution:* "Engine noise + brakes + battery" — the corpus contains excellent pages on each topic individually. The query's embedding is the *centroid* of three distinct topics, so no single page is highly relevant to the whole query. The reranker correctly rejects a diluted retrieval, but a refusal is wrong — the answer exists; the question was just too broad.

The corrective action is different: (1) "I can't help with that" vs. (2) "Please ask about one symptom at a time." Returning the wrong message is actively misleading.

**Why cosine similarity scores already contain the signal:**
- Multi-topic queries produce high similarity scores (0.46–0.49 observed) because the embedding model finds individually relevant pages for each topic. The "dilution" shows up as score *spread across sections*, not in low scores.
- Genuinely OOS queries produce low scores across the board (0.11–0.12 observed) because the entire corpus is motorcycle-domain with near-zero semantic overlap with unrelated topics.

This asymmetry means the classification can be done purely on the candidate similarity scores already computed during retrieval — no second LLM call is needed.

**Thresholds (calibrated empirically on this corpus):**
- `_DILUTION_OOS_THRESHOLD = 0.20` — if the top candidate scores below 0.20, the query is genuinely outside the corpus domain. Return the standard refusal.
- `_DILUTION_SPREAD_MIN = 0.40` — candidates scoring ≥ 0.40 represent real content hits (the manual has relevant material for that sub-topic).
- `_DILUTION_MIN_SECS = 5` — dilution is confirmed if ≥ 3 of the top-5 candidates score ≥ 0.40 *and* all 5 top candidates come from distinct sections. Raised from 3 → 5 after Malayalam diagnostic (see below).

**Verification on real data:**
- OOS "capital of France": top-5 similarity 0.11–0.12 → correctly `"out_of_scope"` → standard refusal.
- Single-topic "engine oil level": reranker returns sources=5 — dilution classifier is never called (it only runs when chunks is empty).

**Recalibration — _DILUTION_MIN_SECS raised from 3 → 5 (post Malayalam diagnostic):**
Malayalam pipeline diagnostic ("How do I check engine oil?" via voice) revealed that a single-topic engine oil query retrieves from 4 sections (MINOR MAINTENANCE TIPS × 2 pages + WARNING INDICATIONS + RECOMMENDED LUBRICANTS + PERIODICAL MAINTENANCE) all at sim ≥ 0.40, causing the classifier to return `"dilution"` even though the query is entirely single-topic. These sections are thematically co-located — all relate to engine oil maintenance — and the diversity is an artefact of how the manual organises related content across sections, not evidence of a multi-topic query.

Raising `_DILUTION_MIN_SECS` to 5 requires all five top-5 candidates to come from distinct sections — a much tighter criterion that only fires when retrieval has genuinely spread across unrelated parts of the manual. Keyword multi-topic queries ("engine noise brakes battery") do not trigger this because their embeddings collapse into one dominant section. Natural-language multi-topic queries produce lower overall similarity (0.37–0.38 observed), which falls below `_DILUTION_SPREAD_MIN` and routes to `"out_of_scope"` — an acceptable fallback (the generic refusal is still correct, just less specific).

For Indic queries, the classifier is additionally guarded by the Decision #34 short-circuit in `generate_answer()`: Indic queries with empty chunks never reach `classify_retrieval_failure` regardless of thresholds.

**Difference from the 75-token guard (Decision #18):** The 75-token guard fires *before* retrieval on the raw token count of the query. The dilution classifier fires *after* retrieval. The two guards cover different failure modes: the token guard catches verbose multi-symptom queries; the dilution classifier catches short but broad queries retrieval couldn't focus.

**Trade-off:** The thresholds are calibrated on 130 chunks of this specific manual with `text-embedding-3-small`. The `_DILUTION_MIN_SECS = 5` threshold makes the multi-topic path harder to trigger; in practice the multi-topic message is now rarely shown and the fallback is the standard "I couldn't find that" refusal. Acceptable for a prototype where false positives are more damaging than false negatives.

---

## 31. Image Auto-Clears After Every Query; Inline Uploader Replaces Popover

**Decision:** The image attachment now auto-clears after each query is submitted. The popover-based upload widget is removed; `st.file_uploader` renders inline in the accessory row above the main input. One thumbnail location only (above the input), with a single ✕ Remove button.

**Why:** Manual testing (L2-10) revealed two UX failures: (1) the image persisted across queries, bleeding into Q2 even though the user did not intend to include it; (2) the image appeared in two separate places (thumbnail above input AND inside the popover), requiring the user to remove it from both locations to fully clear it. This is not how any modern chat interface works — Claude.ai, ChatGPT, and WhatsApp all clear the attachment after send.

**Implementation:** A `_clear_image` session state flag is set in the submit block before `_run_assistant_turn` is called. On the following rerun, the flag is consumed at the top of the script (before any widget renders), the image keys are deleted, and `uploader_key` is incremented to reset the `st.file_uploader` widget. The pipeline reads the image from session state inside `_run_assistant_turn`, which runs in the same rerun as the submit block — the image is still present when the pipeline reads it and clears on the next rerun.

**Why `uploader_key` increment:** Streamlit's `st.file_uploader` retains its selected file in widget state. Deleting the image bytes from session state does not reset the uploader's UI (it still shows the filename). Incrementing the key causes Streamlit to instantiate a fresh widget, visually clearing the uploader.

---

## 32. Voice Auto-Submits After Transcription; st.chat_input Replaces Text Area

**Decision:** Voice recordings now auto-submit after transcription. `st.chat_input` replaces the `st.text_area` + send button combination. Decision #23 (no auto-submit) is superseded.

**Why:** Manual testing (L2-11) confirmed two failures with the text_area approach: (1) the transcribed text remained in the box after sending — the `_clear_text_input` flag mechanism required an explicit `st.rerun()` that was missing from the submit path; (2) Enter key created a new line instead of submitting, which diverges from every modern chat interface. Both issues disappear with `st.chat_input`: Enter submits, Shift+Enter inserts a newline, and the input clears automatically after submission (managed by Streamlit internally).

**Why Decision #23 is no longer valid:** Decision #23 argued that users should review voice transcripts before submitting. In practice, manual testing showed the editing step created confusion — the box didn't clear, which made users unsure whether their query was sent. Auto-submit is cleaner: the transcript appears immediately in the chat history as a user message, so the user can see it and re-ask if the transcription was wrong. This matches the UX of all voice-enabled chat apps.

**Implementation:** On new recording, `transcribe_audio` is called in a thread with a 30-second timeout via `concurrent.futures.ThreadPoolExecutor`. On success, the transcript is stored as `_voice_prompt` in session state and `st.rerun()` is called. On the next rerun, `_voice_prompt` is consumed at the top of the script and treated exactly like a typed query from `st.chat_input`.

---

## 33. 75-Token Guard Applies to User Prompt Only, Not Combined Query

**Decision:** The 75-token length check is applied to `prompt` (the user's typed or spoken text) rather than `combined_query` (prompt + vision description).

**Why:** Manual testing (L2-16) showed that image queries with short prompts were incorrectly firing the token guard. A Hindi question "What is this and what should I do?" (~8 tokens) combined with a 120-token GPT-4o Vision description produces a 128-token `combined_query` — well above the 75-token threshold — even though the user's question is trivially short. The guard fired, showed a "please ask one topic at a time" message, and the vision box never appeared.

**The token guard's purpose** is to catch multi-topic text queries that dilute retrieval. The vision description is always single-topic context (it describes one image) and never contributes to topic dilution. It should not count toward the query length limit.

---

## 34. Dilution Classifier Skipped for Indic Queries

**Decision:** `classify_retrieval_failure()` is not called when `detected_language == "indic"`. Indic queries with zero surviving chunks go directly to `_generate_indic_refusal()`.

**Why:** Manual testing (L2-15 Tamil, L2-16 Hindi) showed single-topic Indic queries being incorrectly classified as "dilution" and returning "Your question covers multiple topics." The dilution classifier's thresholds (`_DILUTION_SPREAD_MIN = 0.40`, `_DILUTION_MIN_SECS = 3`) were calibrated on English queries against English manual chunks using `text-embedding-3-small`. Cross-lingual embedding similarity scores are systematically lower for Indic queries — the same relevant page that scores 0.58 for an English query might score 0.42 for an equivalent Hindi query. This places a single-topic Indic query in the "high enough score + multiple sections" zone that the classifier defines as dilution.

**Why skip rather than recalibrate:** Recalibrating would require a corpus of known-good Indic queries with ground-truth "dilution" vs "out_of_scope" labels — which we don't have. Skipping is provably safe: Indic queries with zero chunks are either genuine content misses (correct to refuse) or cross-lingual retrieval misses where the content exists but wasn't retrieved (also correct to refuse, and to route to sarvam-m for a graceful refusal in the user's language). The dilution classifier adds no value on the Indic path.

---

## 35. Indic Generation: GPT-4o Generates, sarvam-m Translates

**Decision:** The Indic answer generation path in `generator.py` was changed from "sarvam-m generates answer directly from manual chunks" to "GPT-4o generates English answer → sarvam-m translates to user's language."

**Root cause discovered:** sarvam-m has a 7192-token context window. The system prompt + 5 full manual chunks = ~9570 prompt tokens, which exceeds the limit before any answer budget is added. Every Tamil (and potentially all Indic) query that successfully passed retrieval and reranking was crashing with a 422 `unprocessable_entity_error`. The cross-lingual retrieval gap documented in L9 was masking this crash: most Indic queries were failing at retrieval (returning []), so the generation path was never reached. Once the rewriter fix caused retrieval to succeed (top scores 0.37–0.49 for Tamil), the context window crash became visible.

**L9 (Tamil retrieval gap) is now resolved.** The L9 diagnosis was based on measuring similarity of the raw Tamil query against the English corpus (0.09–0.11). This measurement was taken before the rewriter was updated to translate non-English queries to English. After the rewriter update, the English translation is what gets embedded — and its similarity is 0.37–0.49, fully sufficient for retrieval and reranking. Tamil, Bengali, and Gujarati retrieval now works correctly. L9 is struck as a known limitation.

**New Indic generation flow:**
1. `rewrite_query()` translates Indic query to English
2. English query → BM25 + semantic retrieval → reranker (unchanged)
3. GPT-4o generates answer from English chunks (128K context, no limit issue). Rule 7 in `SYSTEM_PROMPT` ("Respond in the same language the user's question is written in") causes GPT-4o to generate directly in the user's language.
4. sarvam-m translates the GPT-4o answer into the user's language as a second pass (ensures Indic-model quality on the final text; translation prompt is ~500 tokens, well within the 7192-token window).

**Why keep sarvam-m:** sarvam-m is purpose-built for Indic text and is the architecturally correct choice for a Sarvam AI take-home assignment. Its role shifts from generation (where its context window is a constraint) to translation (where it excels and the prompt is short).

**Trade-off:** Two API calls per Indic query (GPT-4o + sarvam-m) instead of one. Acceptable for a demo; production could cache or batch.

---

## 36. Token Guard Skipped for Indic Queries

**Decision:** The 75-token guard in `app.py` is skipped when `detected_language == "indic"`. The condition changed from `if len(_tokenizer.encode(prompt)) > MAX_QUERY_TOKENS` to `if detected_language != "indic" and len(_tokenizer.encode(prompt)) > MAX_QUERY_TOKENS`.

**Root cause:** `cl100k_base` (tiktoken) tokenizes Indic scripts 5–7× more densely than English because it has no vocabulary entries for non-Latin scripts and encodes them byte-by-byte. A short 11-word Tamil query tokenizes to 76 tokens; Kannada 74; Telugu 60; Gujarati 65 — all above or dangerously close to the 75-token limit. Every short Tamil query was hitting the guard and receiving the multi-topic response before the pipeline ran.

**Token counts for equivalent queries across scripts (11-word "how often to change engine oil"):**

| Language | Tokens |
|---|---|
| English | 11 |
| Hindi | 39 |
| Marathi | 39 |
| Bengali | 46 |
| Malayalam | 56 |
| Telugu | 60 |
| Gujarati | 65 |
| Kannada | 74 |
| Tamil | 76 |

**Why skip rather than apply guard post-translation:** Applying the guard to the rewritten English query (after `rewrite_query()`) would require making a GPT-4o API call before the guard — defeating its purpose as a cheap pre-filter. Additionally, the rewriter always compresses multi-topic queries to short English, so the guard would rarely fire on the rewritten output regardless.

**Multi-topic handling for Indic:** Indic multi-topic queries are caught downstream — the reranker returns [] when no single chunk scores ≥ 6, and `generate_answer()` returns `_generate_indic_refusal()` (Decision #34). The user gets a graceful refusal in their language rather than the specific "ask one topic at a time" message, but the outcome (re-ask) is the same.

---

## 37. Rewriter Always Outputs "Interceptor 650", Never "My Bike"

**Decision:** Added Rule 4 to `_REWRITE_SYSTEM` in `retriever.py`: "Always refer to the bike as 'the Interceptor 650', never 'my bike' or 'the bike'." Added Tamil and Hindi examples that demonstrate the substitution.

**Root cause discovered:** The rewriter was correctly translating Indic queries to English, but producing "my bike" instead of "Interceptor 650" — e.g. Tamil "how often should the oil be changed in my bike?" → `"How often should the engine oil be changed in my bike?"`. The English query rewriter was already outputting "Interceptor 650" (e.g. `"How often should the oil be changed in the Interceptor 650?"`). This small difference caused meaningfully different retrieval:

| Query variant | Top candidate | Sim | Reranker |
|---|---|---|---|
| "…in my bike" | MINOR MAINTENANCE TIPS p59 (procedure) | 0.509 | 3–4 → returns [] |
| "…in the Interceptor 650" | PERIODICAL MAINTENANCE p104 (schedule) | 0.560 | 7–9 → returns chunks |

`text-embedding-3-small` places "Interceptor 650" close to chunks that mention the model name — which are predominantly the spec and maintenance pages. "My bike" is generic and drifts toward procedural tips pages that don't answer interval questions.

**Verified across all Indic scripts:** Tamil, Hindi, Malayalam, Kannada, Telugu, Gujarati, Bengali, Marathi — all produce identical output `"How often should the engine oil be changed in the Interceptor 650?"` at temperature=0.

**End-to-end verified for Tamil:** Top sim 0.560, reranker 5 chunks, top score 7.0, correct Tamil-language answer with oil change intervals and page citations (PERIODICAL MAINTENANCE pp. 99, 104).

---

## 38. sarvam-m Removed from Generation Path; Sarvam Stack Is Saaras V3 Only

**Decision:** sarvam-m is removed from `generator.py` entirely. GPT-4o handles all generation — English and Indic — via Rule 7 ("Respond in the same language the user's question is written in"). The Sarvam stack in this project is now Saaras V3 (ASR only, `transcriber.py`). Decisions #25, #28, and #29 are superseded.

**Why GPT-4o is sufficient for Indic output:** Decision #35 confirmed GPT-4o with Rule 7 generates correct, grounded, natural-sounding answers directly in the user's language. The sarvam-m "translation" step that followed was translating GPT-4o Tamil output back into Tamil — a no-op. All multilingual verification (Tamil, Hindi, Malayalam, Kannada, Telugu, Gujarati — 5/6 correct answers, Telugu refusal correct) was completed with GPT-4o generating the final answer. sarvam-m added latency and a failure point without improving quality.

**What was removed from `generator.py`:** `_sarvam_client`, `_get_sarvam_client()`, `_strip_think()`, `_call_sarvam()`, `_indic_message()`, `_generate_indic_refusal()`, `_generate_indic_multi_topic()`, `_INDIC_SYSTEM`. The `import re` is also removed (only used by `_strip_think`).

**Simplified generation flow (all languages):** One GPT-4o call with `SYSTEM_PROMPT` + `_build_user_message()`. Rule 7 handles language. No branching between English and Indic paths.

**Indic refusal path (no chunks):** GPT-4o is called with an empty excerpts message. Rule 2 fires ("If the excerpts do not contain enough information to answer, respond with exactly: 'I couldn't find that…'") and Rule 7 delivers it in the user's language. No dedicated refusal function needed.

**Guard message (multi-topic, token-guard path):** `generate_guard_message()` uses GPT-4o translation for non-English languages instead of `_call_sarvam()`.

**`sarvamai` SDK stays in `requirements.txt`:** Still required by `transcriber.py` (Saaras V3 ASR). Only sarvam-m is removed.

**Trade-off:** The project no longer uses a Sarvam generation model. For the Sarvam AI take-home context, Saaras V3 for ASR is the remaining Sarvam integration — a deliberate choice grounded in where each model excels rather than forced integration.

---

## Known Limitations (as of Level 1 Hardening)

The following limitations remain after all edge case fixes. These are honest assessments for interview discussion.

### L1. White Smoke from Exhaust — Content Absent from This Manual
Pages 96–97 were re-ingested via GPT-4o Vision to rule out the extraction hypothesis. The Vision-produced text is rich and complete (1888 and 2107 chars respectively, covering all table rows). The troubleshooting section covers four symptoms: engine starts then shuts off, engine misfires, poor pickup, and ABS lamp continuously on. "White smoke from exhaust" is not present in any form. This is a genuine content gap in the source document, not an extraction failure.

The Vision re-ingest improved the troubleshooting pages significantly: semantic similarity for related queries jumped from ~0.28 to ~0.48. The pages now surface correctly for the symptoms they do cover.

The system behaves correctly for the white smoke query: it refuses rather than hallucinating an answer for something not in the manual, and directs the user to an authorised service centre. This is the desired behaviour.

Interview note: The initial hypothesis (garbled table extraction) was tested and disproved by re-ingesting with Vision. The investigation itself demonstrates the diagnostic approach — when something doesn't work, determine whether the failure is in the pipeline or in the source data before deciding on a fix. In this case, the source data is the limiting factor, not the pipeline.

### L2. 75-Token Query Limit
Queries above 75 tokens are refused with a UX message. A very detailed but single-topic question that happens to be verbose could hit this limit. The threshold was set to catch the specific 80-token multi-symptom failure case; it is not derived from a large distribution of real queries. Production would calibrate this against real usage data.

### L3. ~~Image Cache Is User-Visible State~~ — RESOLVED (post-manual-testing)
Image now auto-clears after each query is submitted. See Decision #31.

### L4. One GPT-4o Rewrite Call Per Query
`rewrite_query()` makes a GPT-4o API call for every user query, even queries with no false premise (the function returns the query unchanged, but the API call is still made). This adds ~200ms and ~$0.001 per query. For a prototype this is fine; production would add a fast heuristic pre-filter (e.g. query starts with assertion-like tokens) before calling GPT-4o.

### L5. Ingest Is Fixed — No Live Updates
The ChromaDB index is built once from the specific PDF. There is no mechanism to update the index if the manual is revised, or to add supplementary documents. This is a deliberate simplification for the prototype scope.

### L7. ~~Voice Input is English Only (Level 2)~~ — RESOLVED (Level 3)
`language_code` changed to `"unknown"` in `transcriber.py`; Saaras V3 auto-detects language. Manual testing confirmed Saaras transcribes spoken Hinglish into Devanagari, correctly detected as Indic, routed through the pipeline and answered by GPT-4o in the user's language via Rule 7 (Decision #38).

### L8. Microphone Access Requires Browser Permission
`st.audio_input` relies on the browser's `getUserMedia` API. If the user denies microphone permission, the widget renders but cannot capture audio. No in-app error message is shown — the browser handles the permission denial. Not fixable at the application layer.

### L6. Session State Is In-Memory
Streamlit session state holds the full chat history in memory. Long sessions with many image uploads could consume significant memory. No pagination or history summarisation is implemented. Acceptable for a demo; production would persist history to a database.

### ~~L9. Tamil (and Bengali/Gujarati) Cross-Lingual Retrieval Gap~~ — RESOLVED (Decision #35)

The L9 diagnosis was incorrect. The 0.09–0.11 similarity scores were measured on the raw Tamil text before the rewriter ran. The rewriter (`rewrite_query()`) translates all Indic queries to English before embedding; the English translation scores 0.37–0.49 against the English corpus — fully sufficient for retrieval and reranking. A live end-to-end diagnostic on 2026-05-25 confirmed Tamil retrieval and reranking both succeed (5 chunks returned, top rerank score 7.0). The actual failure was in the generation step: sarvam-m's 7192-token context window was exceeded by the 5-chunk prompt. Fixed in Decision #35.

### L10. Post-Transcription Error Flash in Audio Widget
After a successful voice transcription, the Streamlit audio widget briefly shows a red error state for ~2 seconds before clearing. Root cause: `st.session_state.audio_key` is incremented and `st.rerun()` is called to reset the widget after transcription; Streamlit's internal audio widget state briefly renders an error indicator during the transition between the old keyed instance and the new one. The transcription, text population, and pipeline all complete correctly — this is a cosmetic rendering artifact only. Not fixable without replacing `st.audio_input` with a custom component, which was rejected as out of scope for a demo (Decision #32 notes).

### L10b. Enter Key Does Not Submit — Ctrl+Enter Does
Plain Enter in the text area creates a newline instead of submitting. Ctrl+Enter submits correctly; Shift+Enter inserts a newline as expected.

Root cause: React intercepts the native `keydown` Enter event on `<textarea>` elements before the JS listener injected via `components.html` can call `e.preventDefault()`. Plain Enter is consumed by React's synthetic event system at the root container before it bubbles to our handler. Ctrl+Enter is not a React-handled shortcut so it passes through to our listener unmodified.

Three iterations attempted: (1) `btn.click()` — Ctrl+Enter submitted, Enter suppressed newline but did not submit (value not synced); (2) `data-testid` selector + React value-setter force-sync — button not found, Enter fell through to newline; (3) multi-selector fallback + force-sync — Ctrl+Enter works, plain Enter still does not.

Fix would require either: replacing `st.text_area` with a custom HTML input (rejected, over-engineering for demo), or intercepting at the capture phase with `useCapture=true` in a Streamlit-aware component. Accepted as known limitation for demo. Users can use Ctrl+Enter or the ↑ button.

### L11. Text Box Lingers Briefly After Send
Any question — typed or voice-transcribed — remains visible in the text box for 1–2 seconds after the send button is clicked, before clearing on the next rerun. Root cause: Streamlit reruns are not instantaneous; the `_clear_text_input` flag is processed at the top of the next rerun cycle, which completes only after the pipeline finishes. The end state is always correct (box clears, message appears in chat history). Cosmetic artifact; acceptable for a demo.

### L12. File Uploader Shows "200MB per file • JPG, PNG, WEBP" Label
The native `st.file_uploader` widget renders a file-size and format hint below the upload button. This label is not suppressable via the Streamlit API — `label_visibility="collapsed"` hides the widget label but not the built-in format/size hint. A custom HTML component would remove it, but custom components were rejected as over-engineering for a demo. Functionally sufficient; cosmetic only.
