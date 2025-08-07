import torch
from PIL import Image
from pypdf import PdfReader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Configuration 
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct" # Uncomment this line for better model 
FLOWCHART_PATH = "flowchart.jpg"
SPEC_PDF_PATH = "specification.pdf"
MAX_TOKENS = 1024

# Load model & processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Helper function: extract PDF to text
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if not page_text:
            print(f"[Warning] Page {i + 1} has no extractable text.")
            continue
        text += f"Page {i + 1}:\n{page_text.strip()}\n\n"
    return text.strip()


# Helper function: prepare prompt
def build_prompt(spec_text):
    return (
        "You are a systems analyst. Compare the specification text and the attached flowchart image. "
        "Answer the following:\n"
        "- Are all steps and logic from the specification correctly represented in the flowchart?\n"
        "- Are there any contradictions or missing steps?\n\n"
        "Respond clearly, listing matches and mismatches.\n\n"
        "Specification text:\n\n"
        f"{spec_text}"
    )


# Load inputs
flowchart_image = Image.open(FLOWCHART_PATH).convert("RGB") 
specification_text = extract_pdf_text(SPEC_PDF_PATH)
prompt_text = build_prompt(specification_text)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": flowchart_image},
            {"type": "text", "text": prompt_text},
        ],
    }
]


# Prepare inputs for model
formatted_text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[formatted_text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(device)


# Generate result
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        do_sample=False  # supaya deterministik
    )

generated_ids_trimmed = [
    output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated_ids)
]

output_texts = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

result = output_texts[0]

# Output result
print("\n=== Model Response ===\n\n")
print(result)

with open('result.txt', 'w') as file:
    file.write(result)