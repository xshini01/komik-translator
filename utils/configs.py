import os
from utils import gemini_ai

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")


class Translator:
    def __init__(self):
        self.models = {
            "Model-1": os.path.join(ROOT_DIR, "model.pt"),
            "Model-2": os.path.join(ROOT_DIR, "best.pt"),
        }

        self.full_methods = {
            "Google": "google",
            "Helsinki-NLP (opus-mt)": "hf",
            "Gemini AI": "gemini",
            "DeepL": "deepl",
        }

        self.fonts = {
            "animeace_i": os.path.join(ROOT_DIR, "fonts", "fonts_animeace_i.ttf"),
            "mangati":    os.path.join(ROOT_DIR, "fonts", "fonts_mangati.ttf"),
            "ariali":     os.path.join(ROOT_DIR, "fonts", "fonts_ariali.ttf"),
        }

        # Source languages for OCR hint & translator source
        self.source_languages = {
            "Auto-detect":          "auto",
            "English":              "en",
            "Japanese":             "ja",
            "Korean":               "ko",
            "Simplified Chinese":   "zh",
            "French":               "fr",
            "Spanish":              "es",
        }

        # Human-readable names for Gemini OCR hint
        self.source_lang_names = {
            "auto": "auto",
            "en":   "English",
            "ja":   "Japanese",
            "ko":   "Korean",
            "zh":   "Chinese",
            "fr":   "French",
            "es":   "Spanish",
        }

    def get_available_methods(self) -> list[str]:
        """Return translation methods available based on current token state."""
        methods = self.full_methods.copy()
        if not gemini_ai.genai_token:
            methods.pop("Gemini AI", None)
        return list(methods.keys())
