import os
import sys
import base64
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VISION_SYSTEM_PROMPT = """You are a visual observer describing images for a motorcycle manual Q&A system.

Describe only what is directly visible in the image — the part, its condition, and any measurable state (colour, fluid level, crack location, indicator state).
Do not diagnose, infer causes, or describe effects or consequences.
Use precise technical terms a mechanic would use.
Respond with ONLY the observation (1–3 sentences). No greeting, no commentary."""

VISION_USER_PROMPT = (
    "Describe what is visually observable in this image: the part, its condition, "
    "and any measurable or visible state. Do not infer causes or effects."
)

MAX_IMAGE_PIXELS = 1024  # resize long edge to this before sending


def _prepare_image(image_bytes: bytes) -> str:
    """
    Resize the image so the long edge is at most MAX_IMAGE_PIXELS,
    then return a base64-encoded PNG string.

    Resizing reduces token cost while keeping enough detail for diagnosis.
    """
    from io import BytesIO

    try:
        img = Image.open(BytesIO(image_bytes))
        img = img.convert("RGB")
    except Exception as exc:
        raise ValueError("Could not read image — try a clearer photo") from exc

    w, h = img.size
    if max(w, h) > MAX_IMAGE_PIXELS:
        scale = MAX_IMAGE_PIXELS / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def describe_image(image_bytes: bytes) -> str:
    """
    Send an image to GPT-4o Vision and get back a constrained symptom description.

    The description is designed to be:
    1. Appended to the user's text query for retrieval (improves search relevance)
    2. Passed to the generation prompt (gives GPT-4o visual context alongside manual chunks)

    Returns a plain string, e.g.:
      "Oil pressure warning light illuminated on instrument cluster while engine is running."
    """
    img_b64 = _prepare_image(image_bytes)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                            "detail": "low",  # low detail is sufficient for symptom ID
                        },
                    },
                    {"type": "text", "text": VISION_USER_PROMPT},
                ],
            },
        ],
        max_tokens=150,
        temperature=0.2,  # low temperature = consistent, factual output
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Smoke test: pass an image path as a command-line argument
    # Usage: python src/vision.py path/to/image.jpg
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/vision.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "rb") as f:
        image_bytes = f.read()

    print("Vision description:")
    print(describe_image(image_bytes))
