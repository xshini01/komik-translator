from PIL import Image
from google import genai
from google.genai import types
from typing import Optional

token_set: bool = False
genai_token: Optional[str] = None

# Default OCR and translation models
GEMINI_OCR_MODEL = "gemini-2.5-flash-lite"
GEMINI_TRANSLATE_MODEL = "gemini-2.5-flash-lite"

safety_set = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]


def save_token(token: str) -> str:
    """Save the Gemini API token and return a masked confirmation string."""
    global token_set, genai_token
    genai_token = token.strip() if token else None
    token_set = True
    if genai_token:
        masked = genai_token[:4] + "*" * max(0, len(genai_token) - 4)
        return f"Token diterima: {masked}"
    return "Melanjutkan tanpa token (menggunakan model lokal)"


def gemini_ai_ocr(img_path: str, source_lang: str = "auto") -> str:
    """
    Perform OCR on an image using Gemini AI.

    Args:
        img_path (str): Path to the combined bubble image.
        source_lang (str): Language hint for OCR (e.g. 'Japanese', 'English').
                           Use 'auto' for automatic detection.

    Returns:
        str: Extracted text with each bubble ending in ';' on its own line.
    """
    if not genai_token:
        raise RuntimeError("Gemini token not set.")

    client = genai.Client(api_key=genai_token)
    image = Image.open(img_path)

    lang_hint = "" if source_lang == "auto" else f" The text is in {source_lang}."
    prompt = (
        f"Extract each speech bubble's text.{lang_hint} "
        "End each bubble's text with ';' on its own line. "
        "Preserve original capitalization exactly. "
        "Output only the extracted text, nothing else."
    )

    response = client.models.generate_content(
        model=GEMINI_OCR_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are an OCR engine for comics. Extract text exactly as-is, "
                "preserving case. Output only the extracted text."
            ),
            safety_settings=safety_set,
        ),
        contents=[prompt, image],
    )
    return response.text.strip()


def gemini_ai_translator(text: str, source_lang: str = "English", target_lang: str = "Indonesian") -> str:
    """
    Translate text using Gemini AI.

    Args:
        text (str): Text to translate.
        source_lang (str): Source language name (e.g. 'English', 'Japanese').
        target_lang (str): Target language name (e.g. 'Indonesian').

    Returns:
        str: Translated text.
    """
    if not genai_token:
        raise RuntimeError("Gemini token not set.")

    client = genai.Client(api_key=genai_token)

    response = client.models.generate_content(
        model=GEMINI_TRANSLATE_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=(
                f"Translate {source_lang} to {target_lang}. "
                "Keep original capitalization (UPPERCASE stays UPPERCASE). "
                "Output must be natural, casual, and easy to understand. "
                "Output only the translation, nothing else."
            ),
            safety_settings=safety_set,
            temperature=0.5,
        ),
        contents=[f"Translate to {target_lang}: {text}"],
    )
    return response.text.strip()
