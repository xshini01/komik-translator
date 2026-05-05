from deep_translator import GoogleTranslator, DeeplTranslator
from transformers import pipeline
from typing import Optional

# Cache for HuggingFace translation pipelines (keyed by model name)
_hf_pipeline_cache: dict[str, any] = {}

# Language configs for HF models
HF_MODELS = {
    ("en", "id"): "Helsinki-NLP/opus-mt-en-id",
    ("id", "en"): "Helsinki-NLP/opus-mt-id-en",
    ("ja", "en"): "Helsinki-NLP/opus-mt-ja-en",
    ("ko", "en"): "Helsinki-NLP/opus-mt-ko-en",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
}


def _get_hf_pipeline(model_name: str):
    """Load and cache a HuggingFace translation pipeline."""
    if model_name not in _hf_pipeline_cache:
        print(f"[translator] Loading HF model: {model_name}")
        _hf_pipeline_cache[model_name] = pipeline("translation", model=model_name)
    return _hf_pipeline_cache[model_name]


class MangaTranslator:
    def __init__(self, source: str = "en", target: str = "id"):
        """
        Args:
            source (str): Source language code (e.g. 'en', 'ja', 'ko').
            target (str): Target language code (e.g. 'id', 'en').
        """
        self.source = source
        self.target = target
        self.translators = {
            "google": self._translate_with_google,
            "hf":     self._translate_with_hf,
            "deepl":  self._translate_with_deepl,
        }

    def translate(self, text: str, method: str = "google", api: Optional[str] = None) -> str:
        """
        Translate text using the specified method.

        Args:
            text (str): Text to translate.
            method (str): One of 'google', 'hf', 'deepl'.
            api (str, optional): API key for DeepL.

        Returns:
            str: Translated text, or original text on failure.
        """
        text = self._preprocess_text(text)
        if not text.strip():
            return text

        translator_func = self.translators.get(method)
        if not translator_func:
            raise ValueError(f"Unknown translation method: '{method}'. "
                             f"Choose from: {list(self.translators.keys())}")
        try:
            result = translator_func(text, api)
            return result if result else text
        except Exception as e:
            print(f"[translator] Warning: translation failed ({e}), returning original text.")
            return text

    def _translate_with_google(self, text: str, api=None) -> str:
        translator = GoogleTranslator(source=self.source, target=self.target)
        return translator.translate(text) or text

    def _translate_with_hf(self, text: str, api=None) -> str:
        model_name = HF_MODELS.get((self.source, self.target))
        if not model_name:
            print(f"[translator] No HF model for {self.source}→{self.target}, "
                  "falling back to Google.")
            return self._translate_with_google(text)
        pipe = _get_hf_pipeline(model_name)
        result = pipe(text)
        return result[0]["translation_text"] if result else text

    def _translate_with_deepl(self, text: str, api: Optional[str] = None) -> str:
        if not api:
            raise ValueError("DeepL API key is required.")
        translator = DeeplTranslator(
            api_key=api,
            source=self.source,
            target=self.target,
            use_free_api=True,
        )
        return translator.translate(text) or text

    @staticmethod
    def _preprocess_text(text: str) -> str:
        return text.replace("．", ".").strip()
