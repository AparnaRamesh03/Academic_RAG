import os
import re
import asyncio
import time
import base64
import traceback
import httpx
import ollama
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from llama_cloud import AsyncLlamaCloud

from chunker import chunk_markdown
import indexer

# Load environment variables
load_dotenv()

WATCH_DIRECTORY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../ingest_folder")
)
VISION_MODEL = "llava"

# --- TOGGLE: Set to False to skip image/diagram detection and description ---
ENABLE_IMAGE_DESCRIPTION = False

PROCESSING_FILES = set()

# Lazy globals to avoid import-time failures when image description is disabled
_llama_client = None
_ocr_reader = None
_PILImage = None


def _get_llama_client() -> AsyncLlamaCloud:
    """
    Lazy-load the AsyncLlamaCloud client.
    """
    global _llama_client

    if _llama_client is None:
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise RuntimeError("LLAMA_CLOUD_API_KEY is not set in the environment.")
        _llama_client = AsyncLlamaCloud(api_key=api_key)

    return _llama_client


def _ensure_image_dependencies_loaded() -> None:
    """
    Lazy-load OCR/image dependencies only when image description is enabled.
    """
    global _ocr_reader, _PILImage

    if not ENABLE_IMAGE_DESCRIPTION:
        return

    if _ocr_reader is None or _PILImage is None:
        try:
            import easyocr
            from PIL import Image as PILImage

            _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            _PILImage = PILImage
        except Exception as e:
            raise RuntimeError(f"Failed to load image/OCR dependencies: {e}")


def wait_for_file_to_stabilize(file_path: str, checks: int = 3, interval: float = 1.0) -> bool:
    """
    Wait until file size stops changing for `checks` consecutive checks.
    Returns True if stable, False if file disappears.
    """
    stable_count = 0
    last_size = -1

    while stable_count < checks:
        if not os.path.exists(file_path):
            return False

        current_size = os.path.getsize(file_path)

        if current_size == last_size and current_size > 0:
            stable_count += 1
        else:
            stable_count = 0
            last_size = current_size

        time.sleep(interval)

    return True


def process_pdf_safely(file_path: str):
    """
    Full guarded ingestion entrypoint.
    Prevents duplicate concurrent processing and prints tracebacks on failure.
    """
    if file_path in PROCESSING_FILES:
        print(f"[~] Already processing: {file_path}")
        return

    PROCESSING_FILES.add(file_path)

    try:
        print(f"[+] Preparing to ingest: {file_path}")

        stable = wait_for_file_to_stabilize(file_path)
        if not stable:
            print(f"[!] File disappeared or never stabilized: {file_path}")
            return

        asyncio.run(parse_pdf_agentic(file_path))

    except Exception as e:
        print(f"[ERROR] Failed processing '{file_path}': {e}")
        traceback.print_exc()

    finally:
        PROCESSING_FILES.discard(file_path)


def extract_text_labels_ocr(image_path: str) -> list[str]:
    """
    Pass 1: Use EasyOCR to extract raw, verified text strings from the image.
    Returns a deduplicated list of text labels found in the image.
    """
    _ensure_image_dependencies_loaded()

    results = _ocr_reader.readtext(image_path, detail=0)

    seen = set()
    labels = []
    for text in results:
        cleaned = text.strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            labels.append(cleaned)
    return labels


def _normalize_whitespace(text: str) -> str:
    """Light cleanup for nearby page text."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _safe_getattr(obj, *names):
    """Return the first available non-None attribute from an object."""
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


def _extract_image_page_number(image) -> int | None:
    """
    Best-effort extraction of figure page number.

    Tries common metadata attribute names first.
    Falls back to parsing the filename if it includes page info,
    e.g. 'page_7_image_1_v2.jpg'
    """
    raw_page = _safe_getattr(
        image,
        "page_number",
        "page",
        "page_num",
        "page_index",
        "page_idx",
    )

    if raw_page is not None:
        try:
            page_num = int(raw_page)
            if page_num >= 1:
                return page_num
        except Exception:
            pass

    filename = _safe_getattr(image, "filename") or ""
    match = re.search(r"page[_\-\s]?(\d+)", filename, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def _get_nearby_page_text(pages_markdown: list[str], page_number: int | None, max_chars: int = 1200) -> str:
    """
    Get local text from the same page for figure grounding.
    """
    if page_number is None:
        return ""

    idx = page_number - 1
    if idx < 0 or idx >= len(pages_markdown):
        return ""

    page_text = _normalize_whitespace(pages_markdown[idx])
    return page_text[:max_chars]


def _extract_figure_caption_from_page(page_text: str, idx: int) -> tuple[int | None, str]:
    """
    Extract the real figure number + caption from the page text.

    Returns:
        (figure_number, caption_text)

    Example:
        'Fig. 1: A generic RAG architecture...'
        -> (1, 'A generic RAG architecture...')
    """
    patterns = [
        r"(?:^|\n)\s*Fig\.?\s*(\d+)\s*:\s*(.+?)(?=\n|$)",
        r"(?:^|\n)\s*Figure\s*(\d+)\s*:\s*(.+?)(?=\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, page_text, flags=re.IGNORECASE)
        if match:
            fig_num = int(match.group(1))
            caption = _normalize_whitespace(match.group(2))
            return fig_num, caption

    return None, f"Extracted Figure {idx}"


def _shorten_figure_caption(caption: str) -> str:
    """
    Keep the figure header short and retrieval-friendly.
    """
    caption = caption.strip()
    parts = re.split(r"(?<=[.!?])\s+", caption)
    return parts[0].strip() if parts else caption


def describe_image_with_ollama(image_path: str, nearby_text: str = "", figure_caption: str = "") -> str:
    """
    OCR-grounded, compact retrieval-oriented figure description.
    """
    print(f"    [*] OCR pass on '{os.path.basename(image_path)}'...")
    ocr_labels = extract_text_labels_ocr(image_path)
    ocr_label_str = ", ".join(ocr_labels) if ocr_labels else "(no text detected by OCR)"
    print(f"    [OCR] Verified labels: {ocr_label_str}")

    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode("utf-8")

    caption_hint = f"\nFIGURE CAPTION FROM PAGE:\n{figure_caption}\n" if figure_caption else ""
    context_hint = f"\nLOCAL PAGE CONTEXT:\n{nearby_text}\n" if nearby_text else ""

    prompt = (
        "You are summarizing one academic figure for retrieval in a vector database.\n\n"
        f"OCR-VERIFIED LABELS:\n[{ocr_label_str}]\n"
        f"{caption_hint}"
        f"{context_hint}\n"
        "Write a compact description using ONLY the OCR labels, the figure caption, and the local page context.\n\n"
        "Return exactly these 3 fields:\n"
        "Key labels: <comma-separated important labels only>\n"
        "Diagram flow: <one short sentence>\n"
        "Retrieval summary: <two short sentences, max 80 words total>\n\n"
        "Rules:\n"
        "- Do not invent labels or entities.\n"
        "- Do not create long lists.\n"
        "- Prefer the figure caption wording when available.\n"
        "- Be concise and retrieval-focused."
    )

    print(f"    [*] Vision pass with {VISION_MODEL} (compact OCR-grounded summary)...")
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_data],
            }
        ],
    )

    return response["message"]["content"].strip()


async def parse_pdf_agentic(file_path: str):
    """
    Upload a PDF and parse it using the Agentic tier.
    Handles text, tables, and optionally generates Ollama vision descriptions
    for images/diagrams.

    Stage-by-stage protection is added so failures are easier to debug.
    """
    filename = os.path.basename(file_path)
    base_name = os.path.splitext(filename)[0]

    # Create a dedicated assets folder for images from this document
    assets_dir = os.path.join(os.path.dirname(file_path), f"{base_name}_assets")
    os.makedirs(assets_dir, exist_ok=True)

    client = _get_llama_client()

    # Step 1: Upload the file
    print(f"[*] Uploading '{filename}' to LlamaCloud...")
    try:
        with open(file_path, "rb") as f:
            file_obj = await client.files.create(
                file=(filename, f, "application/pdf"),
                purpose="parse"
            )
    except Exception as e:
        raise RuntimeError(f"Upload failed for '{filename}': {e}")

    # Step 2: Run the parse job
    print(f"[*] File uploaded (id={file_obj.id}). Starting Agentic parse...")
    try:
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
                **({"images_to_save": ["embedded"]} if ENABLE_IMAGE_DESCRIPTION else {}),
            },
            processing_options={
                "ignore": {
                    "ignore_diagonal_text": True,
                },
            },
            expand=["markdown", "images_content_metadata"] if ENABLE_IMAGE_DESCRIPTION else ["markdown"],
        )
    except Exception as e:
        raise RuntimeError(f"Parse failed for '{filename}': {e}")

    # Step 3: Combine page markdown
    try:
        pages_markdown = [page.markdown for page in result.markdown.pages]
        full_markdown = "\n\n---\n\n".join(pages_markdown)
    except Exception as e:
        raise RuntimeError(f"Markdown assembly failed for '{filename}': {e}")

    # Step 4: Optional figure processing
    image_descriptions = []

    if ENABLE_IMAGE_DESCRIPTION:
        try:
            _ensure_image_dependencies_loaded()
        except Exception as e:
            raise RuntimeError(f"Image dependency initialization failed for '{filename}': {e}")

        try:
            if result.images_content_metadata and result.images_content_metadata.images:
                figures = [
                    img for img in result.images_content_metadata.images
                    if _safe_getattr(img, "presigned_url")
                ]
            else:
                figures = []
        except Exception as e:
            raise RuntimeError(f"Figure metadata extraction failed for '{filename}': {e}")

        if figures:
            print(f"[*] Found {len(figures)} figure(s). Downloading and describing each...")

            async with httpx.AsyncClient(timeout=60.0) as http_client:
                for idx, image in enumerate(figures, start=1):
                    try:
                        image_url = _safe_getattr(image, "presigned_url")
                        image_filename = _safe_getattr(image, "filename") or f"figure_{idx}.png"
                        local_image_path = os.path.join(assets_dir, image_filename)

                        # Download image
                        response = await http_client.get(image_url)
                        response.raise_for_status()
                        with open(local_image_path, "wb") as img_file:
                            img_file.write(response.content)

                        # Filter out likely artifacts
                        MIN_DIMENSION = 150
                        with _PILImage.open(local_image_path) as img:
                            w, h = img.size

                        if w < MIN_DIMENSION or h < MIN_DIMENSION:
                            print(
                                f"    [~] Skipping '{image_filename}' — too small "
                                f"({w}x{h}px), likely an artifact."
                            )
                            os.remove(local_image_path)
                            continue

                        page_number = _extract_image_page_number(image)
                        nearby_text = _get_nearby_page_text(pages_markdown, page_number)

                        page_text = (
                            pages_markdown[page_number - 1]
                            if page_number and 1 <= page_number <= len(pages_markdown)
                            else ""
                        )

                        figure_number, figure_caption = _extract_figure_caption_from_page(page_text, idx)
                        figure_caption = _shorten_figure_caption(figure_caption)

                        description = describe_image_with_ollama(
                            local_image_path,
                            nearby_text=nearby_text,
                            figure_caption=figure_caption,
                        )

                        if figure_number is None:
                            figure_number = idx

                        figure_block_parts = [
                            f"### Figure {figure_number}: {figure_caption}",
                        ]

                        if page_number is not None:
                            figure_block_parts.append(f"Page {page_number}")

                        figure_block_parts.append(
                            f"**Visual Description (AI-generated):** {description}"
                        )

                        image_descriptions.append("\n\n".join(figure_block_parts))

                    except Exception as e:
                        # Do not fail the whole PDF for one bad figure
                        print(f"[WARN] Figure processing failed for '{filename}', figure #{idx}: {e}")
                        traceback.print_exc()
                        continue

    # Step 5: Append image descriptions to markdown
    try:
        if image_descriptions:
            full_markdown += "\n\n## Extracted Figures & Visual Descriptions\n\n"
            full_markdown += "\n\n".join(image_descriptions)
    except Exception as e:
        raise RuntimeError(f"Figure description merge failed for '{filename}': {e}")

    # Step 6: Save markdown
    md_path = os.path.splitext(file_path)[0] + ".md"
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(full_markdown)
    except Exception as e:
        raise RuntimeError(f"Markdown save failed for '{filename}': {e}")

    print(f"[✓] Done! Enriched markdown saved -> {md_path}")
    print(f"    Pages: {len(pages_markdown)} | Figures described: {len(image_descriptions)}")

    # Step 7: Chunking
    try:
        chunks = chunk_markdown(full_markdown, source_file=filename)
    except Exception as e:
        raise RuntimeError(f"Chunking failed for '{filename}': {e}")

    print(f"[chunker] Total chunks ready for indexing: {len(chunks)}")

    # Step 8: Indexing
    try:
        indexer.index_chunks(chunks)
    except Exception as e:
        raise RuntimeError(f"Indexing failed for '{filename}': {e}")


class PDFIngestionHandler(FileSystemEventHandler):
    def _handle_pdf(self, event):
        if event.is_directory:
            return

        file_path = event.src_path
        if not file_path.lower().endswith(".pdf"):
            return

        process_pdf_safely(file_path)

    def on_created(self, event):
        self._handle_pdf(event)



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