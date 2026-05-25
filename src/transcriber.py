import os
import tempfile
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

_client: SarvamAI | None = None

# Maps browser MIME types (from MediaRecorder) to temp-file extensions.
_MIME_TO_SUFFIX: dict[str, str] = {
    "audio/wav":              ".wav",
    "audio/wave":             ".wav",
    "audio/webm":             ".webm",
    "audio/webm;codecs=opus": ".webm",
    "audio/ogg":              ".ogg",
    "audio/ogg;codecs=opus":  ".ogg",
    "audio/mp4":              ".m4a",
}


def _get_client() -> SarvamAI:
    global _client
    if _client is None:
        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key:
            raise ValueError("SARVAM_API_KEY not set in environment")
        _client = SarvamAI(api_subscription_key=api_key)
    return _client


def transcribe_audio(audio_bytes: bytes, mime: str = "audio/wav") -> str:
    """Send audio bytes to Sarvam Saaras V3 ASR and return the transcript.

    Writes to a temp file because the SDK expects a file path (not bytes).
    mime controls the file extension so the SDK (and Saaras) identify the format.
    Temp file is always deleted, even on failure.
    """
    if not audio_bytes:
        raise ValueError("Recording was silent — please try again")

    client = _get_client()
    suffix = _MIME_TO_SUFFIX.get(mime.split(";")[0].strip(), ".wav")

    # Write to a named temp file; delete=False required on Windows (can't open
    # a file that's still held open by another handle)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        response = client.speech_to_text.transcribe(
            file=open(tmp_path, "rb"),
            language_code="unknown",
            model="saaras:v3",
            mode="transcribe",
        )
        transcript = (response.transcript or "").strip()
        if not transcript:
            raise ValueError("Recording was silent — please try again")
        return transcript
    except ValueError:
        raise
    except Exception as exc:
        msg = str(exc).lower()
        if "audio duration" in msg or "maximum limit" in msg or "too long" in msg:
            raise ValueError(
                "Recording was too long (30 second limit) — "
                "please try a shorter question."
            ) from exc
        raise ValueError("Could not transcribe audio — please try again.") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
