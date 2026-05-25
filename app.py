import sys
import hashlib
import concurrent.futures
import tiktoken
import streamlit as st
import streamlit.components.v1 as components

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.transcriber import transcribe_audio
from src.vision import describe_image
from src.retriever import get_candidates, rewrite_query
from src.reranker import rerank
from src.generator import generate_answer, generate_guard_message
from src.language_detector import detect_language

MAX_QUERY_TOKENS      = 75
TRANSCRIPTION_TIMEOUT = 30
_tokenizer = tiktoken.get_encoding("cl100k_base")

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Interceptor 650 Assistant",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── SESSION STATE ──────────────────────────────────────────────────────────────

for key, default in [
    ("messages",         []),
    ("vision_cache",     {}),
    ("audio_key",        0),
    ("uploader_key",     0),
    ("last_audio_hash",  None),
    ("text_input",       ""),
    ("_audio_error",     None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Clear flags: consumed at the top of each rerun, before widgets render.
if st.session_state.pop("_clear_text_input", False):
    st.session_state.text_input = ""

if st.session_state.pop("_clear_image", False):
    for k in ("attached_image", "attached_image_name", "attached_image_size"):
        st.session_state.pop(k, None)
    st.session_state.uploader_key += 1

# ── SIDEBAR ────────────────────────────────────────────────────────────────────

with st.sidebar:
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages        = []
        st.session_state.vision_cache    = {}
        st.session_state.last_audio_hash = None
        st.session_state.audio_key      += 1
        st.session_state.uploader_key   += 1
        st.session_state.text_input      = ""
        st.session_state._audio_error    = None
        for k in ("attached_image", "attached_image_name", "attached_image_size"):
            st.session_state.pop(k, None)
        st.rerun()
    st.caption(
        "Powered by GPT-4o · Sarvam-M · Sarvam Saaras V3\n"
        "Royal Enfield Interceptor 650 Owner's Manual\n"
        "English · Hindi · Tamil · Telugu · Kannada + more"
    )

# ── HEADER ─────────────────────────────────────────────────────────────────────

st.title("Royal Enfield Interceptor 650")
st.caption(
    "Ask about maintenance, warning lights, fluids, or procedures. "
    "Type or record your question, or attach a photo."
)
st.divider()

# ── CHAT HISTORY ───────────────────────────────────────────────────────────────

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("vision_note"):
            st.info(f"From your image: {message['vision_note']}")
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources from the manual"):
                for source in message["sources"]:
                    st.caption(f"**{source['section']}** — Page {source['page']}")

# ── PIPELINE HELPER ────────────────────────────────────────────────────────────

def _transcribe_with_timeout(audio_bytes: bytes) -> str:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(transcribe_audio, audio_bytes)
        try:
            return future.result(timeout=TRANSCRIPTION_TIMEOUT)
        except concurrent.futures.TimeoutError:
            raise RuntimeError(
                "Recording was too long (30 second limit) — "
                "please try a shorter question."
            )


def _run_assistant_turn(prompt: str) -> None:
    image_bytes = st.session_state.get("attached_image")
    image_name  = st.session_state.get("attached_image_name")
    image_size  = st.session_state.get("attached_image_size")

    with st.chat_message("assistant"):
        vision_description = None

        if image_bytes:
            file_key = (image_name, image_size)
            if file_key in st.session_state.vision_cache:
                vision_description = st.session_state.vision_cache[file_key]
            else:
                with st.spinner("Analysing your image..."):
                    try:
                        vision_description = describe_image(image_bytes)
                        st.session_state.vision_cache[file_key] = vision_description
                    except Exception:
                        st.warning(
                            "Could not analyse the image — continuing with text query only"
                        )

        combined_query = prompt
        if vision_description:
            combined_query = f"{prompt}. {vision_description}"

        detected_language = detect_language(combined_query)

        # Token guard applies to the user's text only, not the vision description.
        # Skipped for Indic: cl100k_base tokenizes Indic scripts ~5-7x more densely
        # than English, so even a short Tamil or Kannada query exceeds 75 tokens.
        # Indic multi-topic handling is done via Decision #34's dilution bypass.
        if detected_language != "indic" and len(_tokenizer.encode(prompt)) > MAX_QUERY_TOKENS:
            with st.spinner("Writing answer...") if detected_language == "indic" else st.empty():
                answer_text = generate_guard_message(combined_query, detected_language)
            st.markdown(answer_text)
            st.session_state.messages.append({
                "role": "assistant", "content": answer_text,
                "sources": [], "vision_note": None,
            })
            return

        with st.spinner("Searching the manual..."):
            retrieval_query = rewrite_query(combined_query)
            candidates      = get_candidates(retrieval_query)

        with st.spinner("Re-ranking results..."):
            chunks = rerank(retrieval_query, candidates)

        with st.spinner("Writing answer..."):
            result = generate_answer(
                prompt, chunks, vision_description,
                detected_language=detected_language,
                raw_candidates=candidates,
            )

        if vision_description:
            st.info(f"From your image: {vision_description}")
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources from the manual"):
                for source in result["sources"]:
                    st.caption(f"**{source['section']}** — Page {source['page']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content":     result["answer"],
        "sources":     result["sources"],
        "vision_note": vision_description,
    })

# ── INPUT AREA ─────────────────────────────────────────────────────────────────
# Everything in one block below the chat history.

st.divider()

# Persistent audio error — survives reruns until next successful recording or submit.
if st.session_state.get("_audio_error"):
    st.warning(st.session_state._audio_error)

# Thumbnail when an image is attached — single location only.
if st.session_state.get("attached_image"):
    col_thumb, col_remove = st.columns([1, 9])
    with col_thumb:
        st.image(st.session_state.attached_image, width=56)
    with col_remove:
        if st.button("✕ Remove image", key="remove_image"):
            for k in ("attached_image", "attached_image_name", "attached_image_size"):
                st.session_state.pop(k, None)
            st.session_state.uploader_key += 1
            st.rerun()

# Row 1: file upload | voice recording — side by side.
col_file, col_mic = st.columns(2)

with col_file:
    uploaded = st.file_uploader(
        "📎 Attach photo",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploaded is not None:
        uploaded.seek(0)
        img_bytes = uploaded.read()
        if img_bytes != st.session_state.get("attached_image"):
            st.session_state.attached_image      = img_bytes
            st.session_state.attached_image_name = uploaded.name
            st.session_state.attached_image_size = uploaded.size

with col_mic:
    audio_recording = st.audio_input(
        "🎤 Record question",
        key=f"audio_{st.session_state.audio_key}",
        label_visibility="collapsed",
    )

# Process new voice recording.
if audio_recording is not None:
    audio_recording.seek(0)
    audio_bytes = audio_recording.read()
    audio_hash  = hashlib.md5(audio_bytes).hexdigest()

    if audio_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = audio_hash
        with st.spinner("Transcribing audio..."):
            try:
                transcript = _transcribe_with_timeout(audio_bytes)
                st.session_state.text_input  = transcript
                st.session_state._audio_error = None
            except ValueError as exc:
                st.session_state._audio_error = str(exc)
            except Exception as exc:
                st.session_state._audio_error = str(exc)
        st.session_state.audio_key += 1
        st.rerun()

# Row 2: text area | send button.
col_text, col_send = st.columns([10, 1])

with col_text:
    st.text_area(
        "Message",
        placeholder="Ask about your Interceptor 650… (voice transcript will appear here)",
        height=68,
        label_visibility="collapsed",
        key="text_input",
    )

with col_send:
    st.markdown("<br>", unsafe_allow_html=True)
    send_clicked = st.button("↑", use_container_width=True, type="primary")

# Intercept Enter key in the textarea: submit on Enter, newline on Shift+Enter.
# Runs in a 0-height iframe; accesses parent document via window.parent on same origin.
components.html("""
<script>
(function () {
  if (window._enterSubmitActive) return;
  window._enterSubmitActive = true;

  var lastTa = null;

  // Try multiple selectors — Streamlit's rendered attribute varies by version.
  function findBtn() {
    var doc = window.parent.document;
    return doc.querySelector('[data-testid="baseButton-primary"]') ||
           doc.querySelector('button[kind="primary"]') ||
           Array.from(doc.querySelectorAll('button')).find(function (b) {
             return b.textContent.trim() === '↑';
           });
  }

  function attach() {
    var doc = window.parent.document;
    var ta  = doc.querySelector('textarea');
    // Attach to textarea regardless of whether the button is found yet;
    // we search for the button fresh at keydown time.
    if (!ta || ta === lastTa) return;
    lastTa = ta;

    ta.addEventListener('keydown', function (e) {
      if (e.key !== 'Enter' || e.shiftKey) return;   // Shift+Enter → newline
      e.preventDefault();

      var el = this;
      // Force React to sync the current textarea value before the click fires.
      // Without this Streamlit's debounced onChange may not have reached the
      // backend yet, so the button click sees an empty text_input and no-ops.
      var setter = Object.getOwnPropertyDescriptor(
        window.parent.HTMLTextAreaElement.prototype, 'value'
      ).set;
      setter.call(el, el.value);
      el.dispatchEvent(new Event('input', { bubbles: true }));

      setTimeout(function () {
        var btn = findBtn();
        if (btn) btn.dispatchEvent(new MouseEvent('click', {
          bubbles: true, cancelable: true, view: window.parent
        }));
      }, 100);
    });
  }

  attach();
  setInterval(attach, 300);
}());
</script>
""", height=0)

# Submit.
if send_clicked and st.session_state.text_input.strip():
    prompt = st.session_state.text_input.strip()
    st.session_state["_clear_text_input"] = True
    st.session_state["_clear_image"]      = True
    st.session_state._audio_error         = None

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({
        "role": "user", "content": prompt, "sources": [], "vision_note": None,
    })

    _run_assistant_turn(prompt)
    st.rerun()
