import os
import asyncio
import time
import base64
import httpx
import ollama
import easyocr
from PIL import Image
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from llama_cloud import AsyncLlamaCloud
from chunker import chunk_markdown
import indexer

# Load environment variables (API Key)
load_dotenv()

WATCH_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ingest_folder'))
VISION_MODEL = "llava"   # Ollama vision model for image-to-text descriptions

# --- TOGGLE: Set to False to skip image/diagram detection and description ---
ENABLE_IMAGE_DESCRIPTION = False

# Initialize the async LlamaCloud client
client = AsyncLlamaCloud(api_key=os.getenv("LLAMA_CLOUD_API_KEY"))

# Initialize EasyOCR only if image description is enabled
ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False) if ENABLE_IMAGE_DESCRIPTION else None


def extract_text_labels_ocr(image_path: str) -> list[str]:
    """Pass 1: Use EasyOCR to extract raw, verified text strings from the image.
    Returns a deduplicated list of text labels found in the image.
    """
    results = ocr_reader.readtext(image_path, detail=0)  # detail=0 returns text only
    # Deduplicate while preserving order
    seen = set()
    labels = []
    for text in results:
        cleaned = text.strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            labels.append(cleaned)
    return labels


def describe_image_with_ollama(image_path: str, nearby_text: str = "") -> str:
    """Pass 2: OCR-Grounded Vision — EasyOCR extracts verified text labels first,
    then those labels are injected into the LLaVA prompt to prevent hallucination.
    nearby_text: Text found in the surrounding PDF markdown for additional grounding.
    """
    print(f"    [*] OCR pass on '{os.path.basename(image_path)}'...")
    ocr_labels = extract_text_labels_ocr(image_path)
    ocr_label_str = ", ".join(ocr_labels) if ocr_labels else "(no text detected by OCR)"
    print(f"    [OCR] Verified labels: {ocr_label_str}")

    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode("utf-8")

    context_hint = f"\nCONTEXT FROM PAPER: {nearby_text}\n" if nearby_text else ""

    prompt = (
        "You are a Senior AI Research Architect performing an OCR-Grounded Structural Deconstruction "
        "of an academic figure for a vector database index.\n\n"
        f"VERIFIED TEXT LABELS (extracted by OCR — treat these as ground truth):\n[{ocr_label_str}]\n"
        f"{context_hint}\n"
        "Using ONLY the verified labels above, follow these steps:\n"
        "1. TEXT LABELS: List every verified label. Do not add any that are not in the OCR list.\n"
        "2. LOGOS/ENTITIES: Identify specific AI frameworks or companies from the verified labels "
        "(e.g., Hugging Face, OpenAI, Gemini, Transformer, GAN, LSTM, Diffusion).\n"
        "3. LOGICAL FLOW: Describe arrow directions using the verified labels. Format: "
        "'[Source Node] -> [Action/Process] -> [Destination Node]'.\n"
        "4. CATEGORIZATION: Group verified labels into distinct functional zones.\n"
        "5. SEMANTIC SUMMARY: Write a 2-sentence keyword-rich summary grounded in the verified labels.\n\n"
        "STRICT RULE: Do NOT invent or assume any label, framework, or component not present in the "
        "OCR-verified list above."
    )

    print(f"    [*] Vision pass with {VISION_MODEL} (OCR-grounded)...")
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [image_data]}],
    )
    return response["message"]["content"]


async def parse_pdf_agentic(file_path: str):
    """Upload a PDF and parse it using the Agentic tier.
    Handles text, tables, and generates Ollama vision descriptions for images/diagrams.
    """
    filename = os.path.basename(file_path)
    base_name = os.path.splitext(filename)[0]

    # Create a dedicated assets folder for images from this document
    assets_dir = os.path.join(os.path.dirname(file_path), f"{base_name}_assets")
    os.makedirs(assets_dir, exist_ok=True)

    print(f"[*] Uploading '{filename}' to LlamaCloud...")

    # Step 1: Upload the file
    with open(file_path, "rb") as f:
        file_obj = await client.files.create(file=(filename, f, "application/pdf"), purpose="parse")

    print(f"[*] File uploaded (id={file_obj.id}). Starting Agentic parse...")

    # Step 2: Run the agentic parse job
    result = await client.parsing.parse(
        file_id=file_obj.id,
        tier="agentic",
        version="latest",
        input_options={},
        output_options={
            "markdown": {
                "tables": {
                    "output_tables_as_markdown": True,
                },
            },
            # Only request image extraction if the feature is enabled
            **(  {"images_to_save": ["embedded"]}  if ENABLE_IMAGE_DESCRIPTION else {}  ),
        },
        processing_options={
            "ignore": {
                "ignore_diagonal_text": True,
            },
        },
        expand=["markdown", "images_content_metadata"] if ENABLE_IMAGE_DESCRIPTION else ["markdown"],
    )

    # Step 3: Combine all pages markdown
    pages_markdown = [page.markdown for page in result.markdown.pages]
    full_markdown = "\n\n---\n\n".join(pages_markdown)

    # Step 4: (Optional) Download each figure and generate an Ollama vision description
    image_descriptions = []

    if ENABLE_IMAGE_DESCRIPTION and result.images_content_metadata and result.images_content_metadata.images:
        figures = [
            img for img in result.images_content_metadata.images
            if img.presigned_url
        ]
        print(f"[*] Found {len(figures)} figure(s). Downloading and describing each...")

        async with httpx.AsyncClient(timeout=60.0) as http_client:
            for idx, image in enumerate(figures, start=1):
                local_image_path = os.path.join(assets_dir, image.filename)

                # Download the image
                response = await http_client.get(image.presigned_url)
                response.raise_for_status()
                with open(local_image_path, "wb") as img_file:
                    img_file.write(response.content)

                # Filter out small artifacts (icons, bullets, decorative elements)
                # Only process images larger than 150x150 pixels
                MIN_DIMENSION = 150
                with Image.open(local_image_path) as img:
                    w, h = img.size
                if w < MIN_DIMENSION or h < MIN_DIMENSION:
                    print(f"    [~] Skipping '{image.filename}' — too small ({w}x{h}px), likely an artifact.")
                    os.remove(local_image_path)  # clean up the tiny file
                    continue

                # Use surrounding markdown as contextual grounding for the vision model
                nearby_text = full_markdown[:500] if full_markdown else ""

                # Generate structured technical description using Ollama vision
                description = describe_image_with_ollama(local_image_path, nearby_text=nearby_text)

                caption = os.path.splitext(image.filename)[0].replace("_", " ").title()
                image_descriptions.append(
                    f"### Figure {idx}: {caption}\n\n"
                    f"**Visual Description (AI-generated):** {description}\n"
                )

    # Step 5: Append image descriptions as a dedicated section to the markdown
    if image_descriptions:
        full_markdown += "\n\n## Extracted Figures & Visual Descriptions\n\n"
        full_markdown += "\n\n".join(image_descriptions)

    # Step 6: Save the complete enriched markdown
    md_path = os.path.splitext(file_path)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(full_markdown)

    print(f"[✓] Done! Enriched markdown saved -> {md_path}")
    print(f"    Pages: {len(pages_markdown)} | Figures described: {len(image_descriptions)}")

    # Step 7: Semantic Chunking
    chunks = chunk_markdown(full_markdown, source_file=filename)
    print(f"[chunker] Total chunks ready for indexing: {len(chunks)}")

    # Step 8: Qdrant Indexing (Hybrid)
    indexer.index_chunks(chunks)


class PDFIngestionHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.pdf'):
            file_path = event.src_path
            print(f"\n[+] New PDF Detected: {file_path}")
            # Give the OS a moment to finish flushing the file to disk
            time.sleep(1)
            asyncio.run(parse_pdf_agentic(file_path))


if __name__ == "__main__":
    if not os.path.exists(WATCH_DIRECTORY):
        print(f"Creating watch directory: {WATCH_DIRECTORY}")
        os.makedirs(WATCH_DIRECTORY)

    event_handler = PDFIngestionHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIRECTORY, recursive=False)
    observer.start()

    print(f"[*] Watcher started. Monitoring '{WATCH_DIRECTORY}' for PDF files.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n[*] Watcher gracefully stopped.")
    observer.join()
