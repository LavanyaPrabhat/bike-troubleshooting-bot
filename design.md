**A MORE DETAILED WALKTHROUGH OF 30+ ARCHITECTURAL DECISIONS (ALONG WITH ALTERNATIVES, TRADE OFFS AND KNOWN LIMITATIONS) IS PRESENT IN THE DECISIONS-LOG FILE**



**What is this app:** This is a RAG-based troubleshooting assistant for the Royal Enfield Interceptor 650 owner's manual. It accepts text, image, and voice input in English and Indic languages, returning answers grounded strictly in the manual. Questions outside the scope of the manual are refused.



**Architecture:**



The owner's manual was ingested once before the bot went live:

pdfplumber extracted text from text-heavy pages
GPT-4o Vision described image-heavy pages — diagrams, spec tables, warning symbols
All 130 pages chunked and embedded into a ChromaDB vector store using text-embedding-3-small
One-time offline step, never repeated at runtime



**English text query:**



1. User types a question
2. Token guard checks length → if over 75 tokens, asks user to focus on one topic and stops
3. Language detected as English
4. Query rewriter cleans up the query → strips false assumptions, conversational filler, informal address words → standardises "my bike" to "Interceptor 650" for better retrieval
5. Cleaned query is embedded and matched against the 130 manual chunks using two methods in parallel: Semantic similarity (meaning-based) and BM25 keyword matching, both results fused together using RRF → top 20 candidate chunks returned
6. Reranker scores each of the 20 candidates from 0 to 10 → only chunks scoring 6 or above pass through
7. If nothing passes → refusal returned, LLM never called (Layer 1 grounding)
8. If chunks pass → GPT-4o generates an answer strictly from those chunks, with an explicit instruction to refuse if the context doesn't cover the question (Layer 2 grounding)
9. Answer returned to user



**English voice query:**



1. User records audio
2. Sarvam Saaras V3 transcribes it to English text with automatic language detection
3. Transcript enters the English text flow above



**Indic text query:**



1. User types in Hindi, Tamil, Malayalam, Kannada, Telugu, Gujarati or another Indic language
2. Language detected as Indic via Unicode script check
3. Token guard is skipped → the tokeniser used (cl100k\_base) has no Indic vocabulary and counts Indic characters at 5-7x the rate of English, so an 11-word Tamil question already reads as 76 tokens. Applying the guard here would refuse almost every Indic query incorrectly.
4. Query rewriter translates the query to English and applies the same cleanup as the English path
5. Retrieval and reranking happen identically to the English path, on the English translation
6. If nothing passes → GPT-4o generates a refusal directly in the user's language. The dilution classifier used in the English path is skipped here because its thresholds were calibrated on English embedding scores and would consistently misread Indic-translated queries as multi-topic when they aren't.
7. If chunks pass → GPT-4o generates the answer in the user's original language



**Indic voice query:**



1. User records audio in any Indic language
2. Sarvam Saaras V3 auto-detects the language and transcribes to native script → Hindi becomes Devanagari, Tamil becomes Tamil script
3. Transcript enters the Indic text flow above



**Image input:**



1. User uploads a photo alongside their question
2. GPT-4o Vision describes what is visually observable → what the part is, its condition, any visible damage. It never diagnoses or interprets causes.
3. The same image is cached so multiple follow-up questions don't trigger repeated Vision calls
4. The description is appended to the user's text as additional context
5. The token guard runs on the user's text only, not the combined string
6. Combined query enters the English or Indic flow depending on language
7. Image clears automatically after the query is sent



**Models and their roles:**



1. GPT-4o → query rewriting, reranking, answer generation, image description
2. text-embedding-3-small → converting text to vectors for semantic search
3. rank-bm25 → keyword matching alongside semantic search
4. Sarvam Saaras V3 → speech to text across all languages



**Key Decisions:**



1. **RAG over Long Context Stuffing or Fine Tuning:** Fine tuning would require retraining and would make the grounding condition of "Only from the manual" difficult to implement. Long Context Stuffing would be expensive and slow per query.
2. **Models, DB, hosting service used:
a. GPT 4o:** I used it for query rewriting, reranking, image description and answer generation. This is because it is natively multimodal. It can also handle common indic languages natively, which is why I replaced Sarvam M from the final translation of the answer from English to Indic, because it was an additional API call (latency and failure risk). I also use GPT 4o to make the initial translation of the query from indic to English before chunk matching starts, because in the query rewriting the translation is bundled with 3/4 other steps (such as removing fillers, making the language formal, removing user's false assumptions, adding in the word "Interceptor 650").
**b. text-embedding-3-small:** I used it for semantic matching. It is cost effective and has sufficient quality for technical english such as in a bike manual.
**c. Sarvam Saaras V3:** I used it for speech to text. Chose it over Whisper because I wanted to include indic language support in voice format as well. Whisper is English dominant, and might degrade with heavy Indian accents, code-switching, etc.
**d. Sarvam M:** Chosen initially to convert English results into indic to show to the user. Removed because GPT seemed to be doing a decent job already, no need to add another API call for this use case.
**e. ChromaDB:** Chosen for its simplicity, it stores the vector index as local files with no separate server process, making it easy to deploy alongside the Streamlit app.
**f. Streamlit:** Native chat, file upload and audio components, along with one-click GitHub deployment to Streamlit Community Cloud reduced UI implementation time.
3. **Chunking Strategy:** I used pdfplumber to chunk the manual. I started off with section based chunking. That resulted in coarse chunks of 4+ pages each, where context got lost, and images getting missed out. I readjusted to 1 chunk per page with fallback on vision based chunking for image heavy pages. Pages with mixed topic will still have some matching issues, but better than before.
4. **Retrieval:
a.** I started with semantic search and a similarity threshold. Most queries returned nothing, which led me to reduce chunk size to one-per-page.
**b.** But spec-sheet chunks covering multiple topics had diluted embeddings, so precise queries still scored poorly. I added BM25 keyword search to catch exact matches semantic search missed, with results fused via Reciprocal Rank Fusion.
**c.** BM25 introduced frequency bias - pages mentioning a word many times outranked pages with one precise answer. I added a GPT-4o reranker to fix this: unlike semantic search which compares vectors independently, the reranker reads query and chunk together and scores 0-10 for direct relevance.
5. **Grounding:** Two layers ensure the bot never answers from outside the manual.
**a. Layer 1 -** reranker threshold: chunks scoring below 6 are dropped, and if nothing passes, the generator is never called. So weak matches don't have a chance of triggering GPT 4o to use outside knowledge and create an answer.
**b. Layer 2 -** system prompt: GPT-4o is explicitly instructed to answer only from provided chunks and refuse otherwise. The threshold handles missing context; the system prompt handles present-but-bounded context. Neither alone is sufficient.
6. **Query rewriting should fix false premises:** If the user asks a question with a baked in false premise, the query rewriting fixes this and removes the false premise. Such queries were otherwise not returning any results because their embeddings were matching the false assertion which might be absent from the manual. Alternative was to lower the match threshold which can compromise other query results, hence rejected. Another option was to send a refusal to such queries which would have been unhelpful.
7. **Token Guard:** For any query above 75 tokens, the response given to the user is that they should ask one question at a time for best results. This is because long queries were giving bad matches. This guard avoids a wasted API call. This does not work for indic queries because the cl100k tokeniser does not work economically on indic scripts. small queries become very long token wise. I could not use Sarvam's tokeniser for this because it was not available as a local Python library like cl100k, which works on the laptop and takes microseconds. Sarvam would have to be a different API call and add latency.
8. **Dilution Classifier:** When the reranker returns no results, instead of a generic refusal, I check whether the failure was because the query was too broad or because the content is genuinely not in the manual. If the top candidates score reasonably well but spread across 5+ different manual sections, it suggests the query was multi-topic - the user gets a "please ask one question at a time" message. If scores are low across the board, the content isn't in the manual - standard refusal. This classifier is skipped for Indic queries because even after translation to English, they produce scores in the 0.37-0.49 range for single-topic questions - the same range the classifier uses to detect multi-topic dilution in English. A focused Indic question looks statistically identical to a broad English one, so the classifier would wrongly tell a focused Indic user to rephrase. Skipping it is safe because when the reranker returns nothing for an Indic query, the right response is always a refusal. The reason is not correct, this is a noted limitation.



**Some Known Limitations:**

1. 75 token limit: This will block single topic queries which are simply too long
2. Fixed manual ingest: This means updates to the manual are not automatic. Neither does this bot allow the user to choose the bike model
3. Enter key does not submit the query, Ctrl+Enter does, tried and failed to fix
4. The image upload button, the text box and the mic don't all line up neatly at the bottom of the screen. Custom component approach was evaluated and rejected - adds significant complexity for a cosmetic improvement that doesn't affect functionality
5. The query rewriter makes a GPT-4o API call for every query even when no rewriting is needed. In production, I would add a rule-based pre-filter to skip the call for straightforward queries with no false assumptions



**What I would do with more time:**

1. Multi bike support
2. Conversation memory, user login
3. Followup questions
4. Live manual updates
5. Better indic tokenizer
6. Separately calibrated dilution threshold for indic
7. Mobile adjusted UI
8. Better aligned image attachment-text box-recording mic

