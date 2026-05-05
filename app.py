from utils.add_text import add_text
from utils.detect_bubbles import detect_bubbles
from utils.process_bubble import process_bubble
from utils.qwen2_vl_ocr import qwen2_vl_ocr
from utils.extract_file import extract_file
from utils.compress_toPDF import compress_toPDF
from utils import gemini_ai
from utils.mangadex_downloader import mangadex_download
from utils.translator import MangaTranslator
from utils import configs

from IPython.display import clear_output
from PIL import Image
import gradio as gr
import cv2
import time
import os
import shutil
from tqdm import tqdm
from natsort import natsorted
from google.genai.errors import ClientError
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
config = configs.Translator()
model_ocr = None
processor_ocr = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def split_semicolon(ocr_text: str) -> list[str]:
    """Split OCR output (each bubble ends with ';') into a list of strings."""
    lines = [line.strip() for line in ocr_text.strip().split("\n") if line.strip()]
    segments: list[str] = []
    current: list[str] = []

    for line in lines:
        if line.endswith(";"):
            current.append(line.rstrip(";"))
            segments.append(" ".join(current).strip())
            current = []
        else:
            current.append(line)

    if current:
        segments.append(" ".join(current).strip())

    return segments


def combine_bubbles_vertically(cropped_images: list[Image.Image]) -> Image.Image:
    """Stack PIL images vertically with a small gap."""
    widths, heights = zip(*(img.size for img in cropped_images))
    max_width = max(widths)
    total_height = sum(heights) + 10 * (len(cropped_images) - 1)

    combined = Image.new("RGB", (max_width, total_height), (255, 255, 255))
    y_offset = 0
    for img in cropped_images:
        combined.paste(img, (0, y_offset))
        y_offset += img.size[1] + 10
    return combined


def get_images(folder: str) -> list[str]:
    """Return sorted list of image paths in a folder."""
    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    return [os.path.join(folder, f) for f in natsorted(files)]


def retry_on_429(func, *args, max_retries: int = 10, base_wait: int = 5, **kwargs):
    """Retry a function on Gemini 429/503 errors with exponential back-off."""
    retries = 0
    while retries < max_retries:
        try:
            return func(*args, **kwargs)
        except ClientError as e:
            msg = str(e)
            is_429 = "RESOURCE_EXHAUSTED" in msg or "429" in msg
            is_503 = "UNAVAILABLE" in msg or "503" in msg
            if is_429 or is_503:
                retries += 1
                wait = base_wait * (2 ** (retries - 1))
                label = "Token habis" if is_429 else "Model unavailable"
                print(f"[retry] {label}. Coba lagi dalam {wait}s ({retries}/{max_retries})")
                time.sleep(wait)
            else:
                raise
        except Exception:
            raise
    raise RuntimeError(f"Gagal setelah {max_retries} percobaan.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def predict(
    files_input,
    model_name: str,
    translation_method: str,
    source_language: str,
    font_name: str,
    api_key: str | None = None,
    jpeg_quality: int = 85,
    progress=gr.Progress(track_tqdm=True),
) -> tuple:
    source_dir = "folder_ekstrak"
    save_dir = "save_images"

    MODEL = config.models.get(model_name, "model.pt")
    font = config.fonts.get(font_name, "fonts/fonts_animeace_i.ttf")
    tl_method = config.full_methods.get(translation_method, "google")
    src_code = config.source_languages.get(source_language, "en")
    src_name = config.source_lang_names.get(src_code, "English")

    # Clean up previous run
    for d in (save_dir, source_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(save_dir, exist_ok=True)

    # Input: URL or file(s)
    if isinstance(files_input, str) and files_input.startswith(("http://", "https://")):
        progress(0, desc="Mendownload manga dari MangaDex...")
        mangadex_download(files_input)
    else:
        progress(0, desc="Mengekstrak file...")
        extract_file(files_input)

    # Build translator (non-Gemini path)
    manga_translator = MangaTranslator(
        source=src_code if src_code != "auto" else "en",
        target="id",
    )

    image_files = [
        os.path.join(source_dir, f)
        for root, _, files in os.walk(source_dir)
        for f in natsorted(files)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
        for _ in [None]  # flatten trick
    ]
    # Flatten properly:
    image_files = []
    for root, _, files in os.walk(source_dir):
        for f in natsorted(files):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                image_files.append(os.path.join(root, f))

    errors: list[str] = []

    for idx, file_path in enumerate(tqdm(image_files, desc="Memproses Gambar")):
        try:
            results = detect_bubbles(MODEL, file_path)
            image = cv2.imread(file_path)
            if image is None:
                print(f"[app] Cannot read image: {file_path}")
                continue

            bubbles_data: list[dict] = []
            for result in results:
                x1, y1, x2, y2, score, class_id = result
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                original_crop = image[y1:y2, x1:x2]
                if original_crop.size == 0:
                    continue
                pil_crop = Image.fromarray(original_crop)
                processed_crop, bubble_cont = process_bubble(original_crop.copy())

                bubbles_data.append({
                    "coords": (x1, y1, x2, y2),
                    "original_crop": pil_crop,
                    "processed_crop": processed_crop,
                    "bubble_cont": bubble_cont,
                })

            if not bubbles_data:
                # No bubbles — copy image as-is
                output_idx = sum(1 for e in os.scandir(save_dir) if e.is_file()) + 1
                cv2.imwrite(os.path.join(save_dir, f"output_image_{output_idx:04d}.png"), image)
                continue

            # --- OCR ---
            combined_image = combine_bubbles_vertically([b["original_crop"] for b in bubbles_data])
            combined_path = "combined_bubbles.png"
            combined_image.save(combined_path)

            if gemini_ai.genai_token:
                ocr_result = retry_on_429(gemini_ai.gemini_ai_ocr, combined_path, src_name)
            else:
                ocr_result = qwen2_vl_ocr(combined_image, model_ocr, processor_ocr)

            ocr_lines = split_semicolon(ocr_result)

            # Pad if OCR returned fewer lines than bubbles
            while len(ocr_lines) < len(bubbles_data):
                ocr_lines.append("")

            # --- Translation ---
            translated_lines: list[str] = []
            for line in ocr_lines:
                if not line.strip():
                    translated_lines.append("")
                    continue
                if gemini_ai.genai_token and tl_method == "gemini":
                    translated = retry_on_429(
                        gemini_ai.gemini_ai_translator, line, src_name, "Indonesian"
                    )
                else:
                    translated = manga_translator.translate(line, method=tl_method, api=api_key)
                translated_lines.append(translated)

            # --- Render text ---
            for bubble, translated in zip(bubbles_data, translated_lines):
                if not translated.strip():
                    continue
                x1, y1, x2, y2 = bubble["coords"]
                final_crop = add_text(
                    bubble["processed_crop"], translated, font, bubble["bubble_cont"]
                )
                image[y1:y2, x1:x2] = final_crop

            output_idx = sum(1 for e in os.scandir(save_dir) if e.is_file()) + 1
            output_path = os.path.join(save_dir, f"output_image_{output_idx:04d}.png")
            cv2.imwrite(output_path, image)

        except Exception as e:
            errors.append(f"{os.path.basename(file_path)}: {e}")
            print(f"[app] Error processing {file_path}: {e}")

        time.sleep(0.05)

    if errors:
        print(f"[app] {len(errors)} file(s) had errors:\n" + "\n".join(errors))

    try:
        pdf_path = compress_toPDF(jpeg_quality=int(jpeg_quality))
    except RuntimeError as e:
        print(f"[app] PDF generation failed: {e}")
        pdf_path = None

    return (
        get_images(source_dir),
        get_images(save_dir),
        gr.update(value=pdf_path, visible=pdf_path is not None),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    # --- Token setup screen ---
    with gr.Blocks(title="Setup Token") as token_interface:
        gr.Markdown("## 🔑 Token / API Key Gemini AI (Opsional)")
        gr.Markdown(
            "Jika diisi, OCR dan terjemahan akan menggunakan **Gemini AI** (lebih akurat). "
            "Jika kosong, model lokal **Qwen2-VL** akan digunakan untuk OCR."
        )
        token_input = gr.Textbox(
            label="API Key Gemini AI",
            info="Dapatkan API Key gratis di https://aistudio.google.com/apikey",
            placeholder="Masukkan API Key di sini (atau kosongkan untuk melanjutkan)...",
            type="password",
        )
        save_button = gr.Button("Lanjutkan ▶", variant="primary")
        output_label = gr.Label(label="Status Token")
        save_button.click(fn=gemini_ai.save_token, inputs=token_input, outputs=output_label)

    clear_output()
    token_interface.launch(share=True)

    # Wait for token screen to be submitted
    while not gemini_ai.token_set:
        time.sleep(1)

    # Load local OCR model if no Gemini token
    if not gemini_ai.genai_token:
        global model_ocr, processor_ocr
        if model_ocr is None:
            print("[app] Loading Qwen2-VL OCR model (this may take a while)...")
            model_ocr = Qwen2VLForConditionalGeneration.from_pretrained(
                "prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
                torch_dtype="auto",
                device_map="auto",
            )
            processor_ocr = AutoProcessor.from_pretrained(
                "prithivMLmods/Qwen2-VL-OCR-2B-Instruct"
            )

    # --- Main app UI ---
    with gr.Blocks(theme="JohnSmith9982/small_and_pretty", title="Komik Translator") as ui:
        gr.Markdown("# 🈯 Komik Translator")
        gr.Markdown("Translate komik / manga dari berbagai bahasa ke **Bahasa Indonesia**.")

        with gr.Row():
            # --- Left panel: inputs & settings ---
            with gr.Column(variant="panel"):
                input_mode = gr.Radio(
                    ["Upload File / Gambar", "Link MangaDex"],
                    value="Upload File / Gambar",
                    label="Metode Input",
                    interactive=True,
                )

                with gr.Group(visible=True) as content_file:
                    input_files = gr.Files(
                        label="Upload File",
                        file_count="multiple",
                        file_types=["image", ".zip", ".rar", ".cbz", ".cbr", ".pdf"],
                    )
                    submit_button = gr.Button("🚀 Terjemahkan", variant="primary")

                with gr.Column(visible=False) as content_link:
                    input_link = gr.Textbox(
                        label="Link MangaDex",
                        placeholder="https://mangadex.org/chapter/...",
                    )
                    button_link = gr.Button("🚀 Download & Terjemahkan", variant="primary")

                gr.Markdown("### ⚙️ Pengaturan")

                input_source_lang = gr.Dropdown(
                    choices=list(config.source_languages.keys()),
                    label="Bahasa Sumber",
                    value="English",
                    info="Pilih bahasa teks asli dalam komik",
                    interactive=True,
                )
                input_model = gr.Dropdown(
                    choices=list(config.models.keys()),
                    label="Model Deteksi Bubble (YOLO)",
                    value="Model-1",
                    interactive=True,
                )
                input_tl_method = gr.Dropdown(
                    choices=config.get_available_methods(),
                    label="Metode Terjemahan",
                    value="Google",
                    interactive=True,
                )
                deepl_api = gr.Textbox(
                    label="API Key DeepL",
                    info="Diperlukan hanya jika menggunakan metode DeepL",
                    type="password",
                    placeholder="Masukkan API Key DeepL...",
                    interactive=True,
                    visible=False,
                )
                input_font = gr.Dropdown(
                    choices=list(config.fonts.keys()),
                    label="Font Teks",
                    value="animeace_i",
                    interactive=True,
                )
                jpeg_quality_slider = gr.Slider(
                    minimum=50,
                    maximum=95,
                    value=85,
                    step=5,
                    label="Kualitas PDF (JPEG)",
                    info="85 = seimbang. Turunkan untuk file lebih kecil, naikkan untuk kualitas lebih tinggi.",
                    interactive=True,
                )

            # --- Right panel: outputs ---
            with gr.Column(variant="panel"):
                gr.Markdown("### 📄 Hasil")
                with gr.Tab("Gambar Asli"):
                    ori_imgs = gr.Gallery(label="Gambar Asli", columns=2, height=500)
                with gr.Tab("Hasil Terjemahan"):
                    result_imgs = gr.Gallery(label="Hasil Terjemahan", columns=2, height=500)
                result_file = gr.File(label="⬇️ Download PDF", visible=False)

        # --- Event handlers ---
        def show_mode(mode):
            if mode == "Link MangaDex":
                return gr.update(visible=True), gr.update(visible=False)
            return gr.update(visible=False), gr.update(visible=True)

        def api_visibility(method):
            return gr.update(visible=method.lower() == "deepl")

        input_mode.change(show_mode, inputs=input_mode, outputs=[content_link, content_file])
        input_tl_method.change(api_visibility, inputs=input_tl_method, outputs=deepl_api)

        common_inputs = [input_model, input_tl_method, input_source_lang, input_font, deepl_api, jpeg_quality_slider]
        common_outputs = [ori_imgs, result_imgs, result_file]

        button_link.click(
            predict,
            inputs=[input_link] + common_inputs,
            outputs=common_outputs,
        )
        submit_button.click(
            predict,
            inputs=[input_files] + common_inputs,
            outputs=common_outputs,
        )

    clear_output()
    ui.launch(debug=True, share=True, inline=False)


if __name__ == "__main__":
    main()
