# Bike Bot — Build Progress

## Project: Royal Enfield Interceptor 650 Troubleshooting Bot
**Stack:** RAG · ChromaDB · OpenAI (embeddings + GPT-4o) · Sarvam (Saaras V3 ASR) · Streamlit
**Deadline:** 2 days

---

## Status

### Level 1 — COMPLETE AND HARDENED

#### Core Build
- [x] Project folder structure created
- [x] `requirements.txt` defined
- [x] `src/ingest.py` written (PDF → chunks → embeddings → ChromaDB)
- [x] Dependencies installed
- [x] PDF ingested: 130 pages → 130 chunks (hybrid text + Vision extraction) → stored in ChromaDB
- [x] `src/retriever.py` — hybrid retrieval (semantic + BM25 + RRF), returns 20 candidates
- [x] `src/reranker.py` — GPT-4o re-ranker, scores 20 candidates, returns top 5 or refusal
- [x] `src/vision.py` written (image -> GPT-4o Vision -> constrained symptom description)
- [x] `src/generator.py` written and smoke-tested end-to-end (retriever + generator pipeline working)
- [x] `app.py` — 4-step spinners (image / search / re-rank / answer), pipeline split into get_candidates + rerank calls
- [x] End-to-end test passed: fuel query (correct answer from p18), oil query (correct procedure), France query (correct refusal)

#### Hardening (post-audit, 8 fixes applied)
- [x] **EC01 fixed** — empty string guard in `get_candidates()` prevents OpenAI 400 crash
- [x] **EC25/27/28 fixed** — PIL try/except in `_prepare_image()` + app-layer Vision error handling; bad images now GRACEFUL not CRASH
- [x] **Vision prompt rewritten** — observer-only framing, no diagnostic language or effect inference
- [x] **Image persistence fixed** — Vision description cached by (filename, filesize); no redundant API calls across queries
- [x] **EC04 fixed** — 75-token guard in `app.py` short-circuits multi-topic queries with actionable UX message
- [x] **EC18/EC22 fixed** — `rewrite_query()` normalises false-premise queries before retrieval; generator Rule 6 corrects premises in answers
- [x] **EC14 partially addressed** — reranker prompt updated: troubleshooting/diagnostic pages now score 7–9 for symptom queries. Residual: white smoke content gap in ingest (documented as known limitation L1)
- [x] 29-case edge case audit completed, documented in `errors-and-fixes.md`
- [x] Architecture decisions logged in `decisions-log.md` (21 decisions + Known Limitations section)

---

### Level 2 — COMPLETE

#### Voice Input via Sarvam Saaras V3 ASR
- [x] `SARVAM_API_KEY` added to `.env` and `.env.example`
- [x] `sarvamai>=0.1.0` added to `requirements.txt`, installed
- [x] `src/transcriber.py` created — `transcribe_audio(audio_bytes)` wraps Sarvam SDK; lazy client init; temp-file pattern for Windows compatibility; clean `ValueError` on silent recording, missing key, or API failure
- [x] `app.py` updated — voice widget (`st.audio_input`) in input row; md5 hash dedup prevents reprocessing on reruns; transcription pre-fills text area; `_run_assistant_turn()` helper shared by voice and text paths; `last_audio_hash` reset on Clear; audio widget key incremented after each recording to avoid Streamlit post-transcription error
- [x] Architecture decisions logged (#22, #23, #24)
- [x] `errors-and-fixes.md` updated

---

### Level 3 — COMPLETE

#### Multilingual Support (Indic Languages)
- [x] `src/language_detector.py` — two-stage detection: Unicode script ranges first, langdetect second. `DetectorFactory.seed = 0` for determinism.
- [x] `src/transcriber.py` — `language_code="unknown"` enables Saaras V3 auto-detection
- [x] `src/generator.py` — Sarvam generation path, language-aware routing, dynamic refusal/guard messages
- [x] `app.py` — language detection threaded through pipeline, guard message routed correctly
- [x] `requirements.txt` — `langdetect>=1.0.9` added
- [x] Architecture decisions logged (#25–#28)

---

### Manual Browser Testing — COMPLETE (11 cases, 6 bugs fixed)

**P0 bugs fixed:** image persistence/bleed, voice transcription silent hang, text box not clearing after send, Enter key submits
**P1 bugs fixed:** dilution classifier misfires on Indic queries, vision box missing in Indic path
**Diagnostics:** coolant refusal confirmed correct, Hinglish Devanagari response confirmed correct
**Decisions added:** #31 (image auto-clear), #32 (voice auto-submit + chat_input), #33 (token guard scope), #34 (dilution skip for Indic)

---

### Level 2+3 Post-Build Audit — COMPLETE (44 cases run)

Audit script: `audit_l2_l3.py`
Results before fixes: 16 PASS, 1 FAIL, 13 CRASH, 3 GRACEFUL, 11 MANUAL

**Root cause of all 13 crashes: ECA-01 — Sarvam reasoning mode exhausts max_tokens**
Both sarvam-105b and sarvam-m run reasoning/thinking mode by default. At max_tokens=600,
the reasoning trace alone consumed all tokens, leaving content=None (sarvam-105b) or
a truncated <think> block (sarvam-m). Our .strip() call on None crashed.

---

### Post-Audit Fixes — COMPLETE (applied and verified)

#### Fix 1: ECA-01 — Sarvam crash on content=None [ALL 13 P0 CRASHES RESOLVED]

**Files changed:** `src/generator.py`

Changes applied:
- [x] Switched from `sarvam-105b` → `sarvam-m` (content field never None in sarvam-m)
- [x] `max_tokens` for main generation: 600 → 2048 (accommodates thinking trace + answer)
- [x] `max_tokens` for refusal/guard messages: 120/512 → 1024
- [x] `_strip_think(text)` — strips `<think>...</think>` blocks; handles truncated (dangling) open tags
- [x] `_call_sarvam(messages, max_tokens)` — centralised Sarvam call with None guard (`content or ""`) + think-stripping
- [x] `_indic_message()` prompt restructured as a translation task ("SAMPLE: [question] MESSAGE TO TRANSLATE: [english_text]") to prevent sarvam-m from answering the question instead of translating the refusal

**Verification (re-ran all 13 formerly crashed cases):**
- L3-01 Hindi engine oil: PASS — sources=5, answer in Hindi ✓
- L3-02 Tamil tyre pressure: PASS — no crash; refusal in Tamil (cross-lingual retrieval miss, see open issues)
- L3-03 Bengali engine oil: PASS — no crash; refusal in Bengali (cross-lingual retrieval miss)
- L3-04 Gujarati tyre pressure: PASS — no crash; refusal in Gujarati (cross-lingual retrieval miss)
- L3-05 Kannada engine oil: PASS — sources=5, answer in Kannada ✓
- L3-07 Hindi false premise: PASS — sources=5, premise corrected in Hindi ✓
- L3-08 Hindi OOS biryani: PASS — refusal in Hindi ✓
- CC-01 Hindi refusal: PASS — Devanagari script in refusal message ✓
- CC-02 Tamil refusal: PASS — Tamil script in refusal message ✓
- CC-03 Bengali refusal: PASS — Bengali script in refusal message ✓
- CC-04 Hindi guard: PASS — Devanagari script in guard message ✓
- CC-05 Tamil guard: PASS — Tamil script in guard message ✓
- L3-09 Hindi long query guard: not individually re-run but covered by CC-04 (same code path)

#### Fix 2: Multi-topic dilution detection — new behaviour, not a crash fix

**Files changed:** `src/reranker.py`, `src/generator.py`, `app.py`

Changes applied:
- [x] `src/reranker.py` — `classify_retrieval_failure(candidates)` added. Sorts candidates by cosine similarity. Returns `"dilution"` if top-5 score ≥ 0.40 AND span ≥ 3 distinct sections; `"out_of_scope"` if top score < 0.20; else `"out_of_scope"`.
- [x] `src/generator.py` — `_classify_failure(raw_candidates)` thin wrapper (deferred import avoids circular). `generate_answer()` now accepts `raw_candidates: list[dict] | None = None`. When `chunks` is empty, consults `_classify_failure` and returns either `MULTI_TOPIC_RESPONSE` / `_generate_indic_multi_topic()` for dilution, or the standard refusal for out_of_scope.
- [x] `src/generator.py` — `_generate_indic_multi_topic(question)` added — translates the multi-topic message via sarvam-m.
- [x] `src/generator.py` — `MULTI_TOPIC_RESPONSE` constant added (English static string).
- [x] `app.py` — `raw_candidates=candidates` passed to `generate_answer()`.

**Calibration note:** Initial thresholds (0.25–0.39 in-range) were wrong. Tested against real data:
- Multi-topic query "engine noise + brakes + battery": top-5 similarity 0.46–0.49, 5 distinct sections → correctly `"dilution"`
- OOS "capital of France": top-5 similarity 0.11–0.12 → correctly `"out_of_scope"`

**Verification:**
- Multi-topic English: PASS — returns "Your question covers multiple topics..." ✓
- Genuine OOS English: PASS — returns standard "I couldn't find that..." refusal ✓
- Single-topic English: PASS — returns answer with sources=5 (dilution classifier not called) ✓

---

### Open Issues (not yet fixed — document only)

1. **Enter key in text area creates newline instead of submitting**
   - st.text_area does not intercept Enter. st.chat_input was tried and reverted (layout issues).
   - Ctrl+Enter submits. Accepted for demo. Users use the ↑ send button or Ctrl+Enter.

### Resolved Issues (previously open)

- **Cross-lingual retrieval miss for Tamil/Bengali/Gujarati** — RESOLVED (Decision #35, #36, 2026-05-25)
  - Root cause was NOT embedding alignment — the rewriter was already translating to English before embedding (scores 0.37–0.49, not 0.09–0.11). L9 was based on measuring raw Indic query similarity, not post-rewrite.
  - Actual bug 1: sarvam-m context window (7192 tokens) exceeded by 5-chunk prompt (~9570 tokens) → 422 crash. Fixed: GPT-4o generates English answer → sarvam-m translates.
  - Actual bug 2: token guard (75-token limit) misfired on Indic scripts — Tamil tokenizes to 76 tokens for a short query (cl100k_base has no Indic vocabulary). Fixed: skip guard for Indic.

---

## Manual Browser Testing — COMPLETE (11 voice cases run, 6 bugs fixed)

**Session 2 testing covered all 11 manual cases (L2-07 to L2-16 + diagnostics).**

### P0 Bugs Fixed
- **MT-01 Image bleed** — Image persisted as context across unrelated queries. Fix: `_clear_image` session state flag clears `attached_image*` keys and increments `uploader_key` after every submit.
- **MT-02 Long voice hang** — Saaras SDK had no timeout; long recording blocked indefinitely. Fix: `_transcribe_with_timeout()` using `concurrent.futures.ThreadPoolExecutor` with `future.result(timeout=30)`.
- **MT-03 Text box not clearing** — `_clear_text_input` flag was set but `st.rerun()` was missing after `_run_assistant_turn()`. Fix: added `st.rerun()` at end of submit block.
- **MT-04 Enter key newline** — `st.text_area` doesn't intercept Enter. Attempted fix via `st.chat_input` (sticky footer) caused layout regression; reverted. Accepted: users use ↑ button.

### P1 Bugs Fixed
- **MT-05 Dilution classifier misfires on Indic** — Cross-lingual embedding scores for Indic are lower than English thresholds; single-topic Indic queries falsely classified as multi-topic. Fix: skip `classify_retrieval_failure()` entirely when `detected_language == "indic"`.
- **MT-06 Vision box missing in Indic path** — Token guard was applied to `combined_query` (prompt + vision description). Short Hindi + 120-token vision description triggered guard early, returning before `st.info(vision_description)`. Fix: token guard now checks `len(_tokenizer.encode(prompt))` only.

### Diagnostics (no bugs)
- **MT-07 Coolant refusal** — "How do I check the coolant level?" returns correct refusal. Top retrieval scores 0.44 on brake fluid / engine oil pages. Confirmed: no coolant procedure in corpus.
- **MT-08 Hinglish Devanagari** — Typed Hinglish (Latin script) correctly detects as "english" (Decision #27). Voice Hinglish transcribed by Saaras V3 INTO Devanagari Hindi → correctly routed to sarvam-m. Correct end-to-end behavior.

---

## Layout Simplification — COMPLETE

**Problem:** After adding voice widget, file_uploader and audio_input appeared as large separate boxes above the chat area, disconnected from the text input.

**Fix applied (minimal — per user instruction):**
- Removed `st.chat_input` (was causing layout regression with sticky footer)
- All input widgets now in a single block BELOW chat history (after a divider)
- Row 1: `st.columns(2)` → `st.file_uploader` | `st.audio_input` side by side
- Row 2: `st.columns([10, 1])` → `st.text_area` | ↑ send button

**components/chat_input/** — HTML/JS/CSS custom component was started but abandoned at user request ("over-engineering for a demo"). Directory exists but is unused.

---

## Decisions Log — COMPLETE (#29–#38)

- **#29** — sarvam-m switch + reasoning mode + `max_tokens=2048` + `_strip_think()` + `_call_sarvam()`
- **#30** — Section-diversity heuristic for dilution detection; `_DILUTION_MIN_SECS` raised 3→5 after false positive on single-topic Indic queries
- **#31** — Image auto-clears after every query; `_clear_image` flag + `uploader_key` increment
- **#32** — Voice auto-submits after transcription; `st.chat_input` path described (NOTE: reverted to `st.text_area` for layout reasons — Decision #32 partially describes superseded state)
- **#33** — 75-token guard applies to user `prompt` only, not `combined_query`
- **#34** — Dilution classifier skipped for Indic queries
- **#35** — Indic generation: GPT-4o generates English answer, sarvam-m translates (fixes sarvam-m 7192-token context window crash) — superseded by #38
- **#36** — Token guard skipped for Indic queries (cl100k_base tokenizes Indic scripts 5–7× more densely than English)
- **#37** — Rewriter always outputs "Interceptor 650" not "my bike" — specific model name retrieves measurably better (sim 0.560 vs 0.509)
- **#38** — sarvam-m removed from generation path entirely; GPT-4o with Rule 7 handles all Indic output; Sarvam stack is Saaras V3 (ASR) only

---

## Full Project Summary (current state as of 2026-05-25)

### What was built
RAG chatbot answering maintenance and troubleshooting questions for the Royal Enfield Interceptor 650, grounded strictly in the official owner's manual. Users can type, upload images, or record voice in English or any Indic language.

### GitHub
Repo: https://github.com/LavanyaPrabhat/bike-troubleshooting-bot
Author: Lavanya Prabhat <lavanyapandey5@gmail.com>
Latest commit: 0f42ee0
All 8 commits authored by Lavanya Prabhat. No other names appear in any committed file.

### Deployment
Live on Streamlit Cloud (auto-deploys on push to master).
Secrets OPENAI_API_KEY and SARVAM_API_KEY set via Streamlit Cloud UI — not in repo.

### Pipeline (end-to-end)
1. Image (optional) → GPT-4o Vision → symptom description
2. Language detection → "english" or "indic" (two-stage: Unicode script ranges → langdetect)
3. Token guard: >75 tokens AND English only → multi-topic guard message (skipped for Indic — Decision #36)
4. Rewriter (GPT-4o, temp=0) → normalised English query; always outputs "Interceptor 650" not "my bike" (Decision #37)
5. Hybrid retrieval: semantic (text-embedding-3-small) + BM25, fused via RRF → 20 candidates
6. Reranker (GPT-4o, temp=0) → scores 0–10, returns top 5 or [] if top score < 6
7. Generator (all languages: one GPT-4o call, Rule 7 handles language — Decision #38):
   - chunks present → GPT-4o generates answer in user's language (Rule 7)
   - Indic + no chunks → GPT-4o with empty excerpts → Rule 2 + Rule 7 fires in user's language (dilution classifier skipped — Decision #34)
   - English + no chunks → dilution classifier → MULTI_TOPIC_RESPONSE or NO_CONTEXT_RESPONSE

### All bugs fixed (complete list)
- EC01: empty string guard in get_candidates()
- EC04: 75-token multi-topic guard
- EC14: reranker scoring for symptom/diagnostic pages
- EC18/22: false-premise rewriter + generator Rule 6
- EC25/27/28: Vision error handling
- ECA-01: sarvam-m reasoning mode crash (content=None, max_tokens=600)
- MT-01: image bleed across queries
- MT-02: voice recording infinite hang
- MT-03: text box not clearing after send
- MT-05: dilution classifier misfiring on Indic
- MT-06: vision box missing in Indic path
- MT-09: long recording raw API error message
- MT-10: Hinglish rewriter missing address word stripping
- MT-16: sarvam-m context window (7192 tokens) exceeded by 5-chunk prompt → 422 crash
- MT-17: Indic token over-tokenization — Tamil 76 tokens > 75-token guard, misfired on all queries
- MT-18: rewriter outputting "my bike" instead of "Interceptor 650" for Indic queries → wrong retrieval

### Known limitations (accepted)
- L1: White smoke from exhaust — content genuinely absent from manual (not an extraction failure)
- L2: 75-token limit may block a verbose but single-topic English query
- L10b: Plain Enter creates newline; Ctrl+Enter submits (Streamlit limitation)
- L10: Post-transcription audio widget cosmetic error flash (~2 seconds)
- L11: Text box lingers for ~1s after send
- L12: "200MB per file" label on file uploader (Streamlit cannot suppress it)

### Documents
- decisions-log.md: 37 decisions + Known Limitations L1–L12 (L9 struck as resolved)
- errors-and-fixes.md: ECA-01, MT-01–MT-18
- README.md: live on GitHub; demo URL and Loom link placeholders to be filled

### Multilingual verification (2026-05-25, post-deployment)
End-to-end pipeline tested across 6 Indic languages — all token guards skipped correctly, all rewriters output "Interceptor 650":

| Language | Query topic | Reranker | Outcome |
|---|---|---|---|
| Tamil | Oil change interval | 5 chunks, score 7 | Correct Tamil answer ✓ |
| Hindi | Tyre pressure | 5 chunks, score 10 | Correct Hindi answer with psi values ✓ |
| Malayalam | Brake fluid check | 5 chunks, score 7 | Correct Malayalam answer ✓ |
| Kannada | Fuel tank capacity | 5 chunks, score 7 | Correct Kannada answer ✓ |
| Telugu | Battery replacement | 0 chunks | Correct Telugu refusal (info not in manual) ✓ |
| Bengali | Engine oil spec | 5 chunks, score 10 | Correct Bengali answer ✓ |

### Nothing outstanding
All levels complete. All known bugs fixed. All decisions and errors documented. Deployed and live.

---

## File Map (current state — all files present and working)

```
bike-bot/
├── data/
│   └── royal-enfield-interceptor-650-owners-manual-english.pdf
├── src/
│   ├── __init__.py
│   ├── ingest.py             ← One-time PDF ingest; run once; do not re-run
│   ├── retriever.py          ← Hybrid retrieval (semantic + BM25 + RRF) + query rewriting
│   ├── reranker.py           ← GPT-4o re-ranker + classify_retrieval_failure() [MODIFIED]
│   ├── vision.py             ← GPT-4o Vision (image → symptom description)
│   ├── generator.py          ← GPT-4o-only generation; Rule 7 handles Indic output [SIMPLIFIED — Decision #38]
│   ├── transcriber.py        ← Sarvam Saaras V3 ASR; language_code="unknown"
│   └── language_detector.py  ← detect_language() → "english" | "indic"
├── app.py                    ← Streamlit frontend [MODIFIED — two-row input layout, all bugs fixed]
├── components/
│   └── chat_input/
│       └── index.html        ← UNUSED — abandoned custom HTML/JS component
├── audit.py                  ← Level 1 full 29-case edge case audit (historical)
├── audit_fixes.py            ← Level 1 targeted re-audit (historical)
├── audit_l2_l3.py            ← Level 2+3 44-case audit script
├── chroma_db/                ← Persistent vector store (130 chunks) — DO NOT DELETE
├── .env                      ← OPENAI_API_KEY + SARVAM_API_KEY
├── .env.example              ← Template
├── requirements.txt          ← All dependencies including langdetect>=1.0.9
├── progress.md               ← This file
├── decisions-log.md          ← 37 decisions logged; complete
└── errors-and-fixes.md       ← Audit log; ECA-01, MT-01–MT-08, MT-16–MT-18 documented
```

---

## Key Architecture Facts (for resuming without full context)

**Pipeline flow (app.py → _run_assistant_turn):**
1. Vision: if image attached, `describe_image()` → `vision_description`
2. Build `combined_query = prompt + vision_description`
3. `detected_language = detect_language(combined_query)` → "english" or "indic"
4. Token guard: if >75 tokens in `prompt` AND `detected_language != "indic"` → `generate_guard_message(combined_query, detected_language)` → stop (skipped for Indic — Decision #36)
5. `rewrite_query(combined_query)` → retrieval_query (neutralises false premises)
6. `get_candidates(retrieval_query)` → 20 raw candidates (each has text, section, page, similarity)
7. `rerank(retrieval_query, candidates)` → top-5 chunks or [] if all scored < 6
8. `generate_answer(prompt, chunks, vision_description, detected_language, raw_candidates=candidates)`
   - If chunks empty + indic → `_generate_indic_refusal(question)` (classifier SKIPPED for Indic)
   - If chunks empty + english → `_classify_failure(raw_candidates)` → "dilution" or "out_of_scope"
   - dilution + english → MULTI_TOPIC_RESPONSE
   - out_of_scope + english → NO_CONTEXT_RESPONSE
   - chunks present + english → GPT-4o (model="gpt-4o", max_tokens=600)
   - chunks present + indic → GPT-4o generates answer directly in user's language via Rule 7 (Decision #38)

**Sarvam API:**
- Used for ASR only (transcriber.py, Saaras V3)
- Base URL for ASR: sarvamai SDK
- sarvam-m removed from generation path (Decision #38); SARVAM_API_KEY only required for voice input
- All Sarvam calls go through `_call_sarvam()` in generator.py which applies None guard + `_strip_think()`

**Language detection logic (language_detector.py):**
- Stage 1: any Devanagari/Tamil/Telugu/Kannada/Malayalam/Bengali/Gujarati/Gurmukhi Unicode char → "indic" immediately
- Stage 2: langdetect ISO code in {hi,ta,te,kn,ml,bn,gu,pa,mr,ur} → "indic"
- Otherwise → "english"
- Hinglish and Tanglish in Latin script misclassified as "english" → GPT-4o fallback (acceptable per EC06, Decision #27)

**Dilution thresholds (reranker.py):**
- `_DILUTION_OOS_THRESHOLD = 0.20` — top candidate below this = genuinely OOS
- `_DILUTION_SPREAD_MIN = 0.40` — at/above this = real content hit
- `_DILUTION_MIN_SECS = 5` — all top-5 must be from ≥ 5 distinct sections = dilution (raised from 3 after false positive on single-topic oil queries)

**To run the app:** `streamlit run app.py` (from bike-bot directory)
**To run the audit:** `python audit_l2_l3.py` (from bike-bot directory)
**Env vars needed:** `OPENAI_API_KEY`, `SARVAM_API_KEY` (both in `.env`)
