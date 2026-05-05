from qwen_vl_utils import process_vision_info
from PIL import Image


def qwen2_vl_ocr(image: Image.Image, model, processor) -> str:
    """
    Performs OCR on a combined bubble image using the Qwen2-VL model.

    Each speech bubble's text is separated by ';' on a new line, matching
    the format expected by split_semicolon() in app.py.

    Args:
        image (PIL.Image): Combined image of speech bubble crops.
        model: Loaded Qwen2VLForConditionalGeneration model.
        processor: Loaded AutoProcessor for the model.

    Returns:
        str: Extracted text with each bubble's text ending in ';' on its own line.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": (
                        "Extract the text from each speech bubble in this image. "
                        "Output each bubble's text on a separate line, ending with ';'. "
                        "Preserve the original capitalization exactly (UPPERCASE stays UPPERCASE). "
                        "Output only the extracted text, nothing else."
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    buffer = "".join(output_text).replace("<|im_end|>", "").strip()
    return buffer
