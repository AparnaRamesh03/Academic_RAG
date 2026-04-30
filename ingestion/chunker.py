"""
chunker.py — Hierarchical Semantic Chunking for Academic Markdown Documents

Pipeline:
    1. Split markdown into pages first
    2. Clean markdown before chunking
    3. Extract special blocks first:
        - markdown tables
        - mermaid diagram blocks
        - display equations
    4. Header-aware splitting on remaining prose
    5. Recursive character chunking inside each page
    6. Metadata tagging per chunk
    7. Figure description chunks treated as standalone content_type="figure_description"

Design notes:
    - Payload `text` stays page-local for cleaner citations
    - `embedding_text` includes:
        * optional boundary tail from previous page
        * section header
        * current chunk text
    - Inline body image markdown is removed to avoid duplication because
      figure_description chunks already capture visual content
"""

import os
import re
import json
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# ── Chunking Parameters ─────────────────────────────────────────────────────
CHUNK_SIZE = 1300
CHUNK_OVERLAP = 150

# Retrieval-only boundary carry-over from previous page
BOUNDARY_CONTEXT_CHARS = 300

# Save chunk JSON for debugging
DEBUG_SAVE_CHUNKS = False

# Headers to split on
HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# ── Splitters ───────────────────────────────────────────────────────────────
_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT,
    strip_headers=True,   # headings stay in metadata; embedding_text reintroduces them cleanly
)

_char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _has_table(text: str) -> bool:
    """Detect if a chunk contains a Markdown table."""
    return bool(re.search(r"^\|.+\|", text, re.MULTILINE))


def _has_image_description(text: str) -> bool:
    """Detect if a chunk contains an AI-generated image description."""
    return "**Visual Description (AI-generated):**" in text


def _derive_section_header(chunk_metadata: dict) -> str:
    """Build a readable section path from header metadata."""
    parts = []
    for key in ["h1", "h2", "h3"]:
        val = chunk_metadata.get(key, "").strip()
        if val:
            parts.append(val)
    return " > ".join(parts) if parts else "Preamble"


def _normalize_whitespace(text: str) -> str:
    """Collapse noisy whitespace while keeping readable paragraph flow."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_inline_image_markdown(text: str) -> str:
    """
    Remove inline markdown image embeds from body text.

    Example removed:
        ![alt text](page_1_image_1_v2.jpg)
    """
    return re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)


def _strip_basic_html(text: str) -> str:
    """Convert simple HTML remnants into plain text."""
    text = re.sub(r"<sup>(.*?)</sup>", r"\1", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<sub>(.*?)</sub>", r"\1", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[^>]+>", "", text)
    return text


def _fix_unicode_artifacts(text: str) -> str:
    """Remove common PDF/OCR/parser artifacts."""
    replacements = {
        "\uFFFE": "",
        "\uFFFF": "",
        "\uFFFD": "",
        "￾": "",
        "\u00AD": "",  # soft hyphen
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def _fix_broken_hyphenation(text: str) -> str:
    """
    Join words broken across lines/pages:
        end-\nto-end -> end-to-end
    """
    return re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "-", text)


def _merge_wrapped_lines(text: str) -> str:
    """
    Merge PDF-style hard-wrapped lines into normal paragraphs,
    while preserving headings, lists, tables, code fences, equations, and page separators.
    """
    lines = text.split("\n")
    merged = []
    buffer = ""
    in_code_fence = False
    in_equation_block = False

    def flush_buffer():
        nonlocal buffer
        if buffer.strip():
            merged.append(buffer.strip())
        buffer = ""

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        # Fence toggles
        if stripped.startswith("```"):
            flush_buffer()
            merged.append(line)
            in_code_fence = not in_code_fence
            continue

        if stripped == "$$":
            flush_buffer()
            merged.append(line)
            in_equation_block = not in_equation_block
            continue

        if in_code_fence or in_equation_block:
            merged.append(line)
            continue

        # Blank line -> paragraph break
        if not stripped:
            flush_buffer()
            merged.append("")
            continue

        # Preserve structural lines
        is_structural = (
            stripped.startswith("#")
            or stripped.startswith("* ")
            or stripped.startswith("- ")
            or re.match(r"^\d+\)", stripped)
            or re.match(r"^\d+\.", stripped)
            or stripped.startswith("|")
            or stripped == "---"
        )

        if is_structural:
            flush_buffer()
            merged.append(stripped)
            continue

        # Normal prose line
        if not buffer:
            buffer = stripped
        else:
            buffer += " " + stripped

    flush_buffer()

    text = "\n".join(merged)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_markdown_for_chunking(body_md: str) -> str:
    """Full normalization pipeline before page splitting / chunking."""
    body_md = _remove_inline_image_markdown(body_md)
    body_md = _fix_unicode_artifacts(body_md)
    body_md = _strip_basic_html(body_md)
    body_md = _fix_broken_hyphenation(body_md)
    body_md = _merge_wrapped_lines(body_md)
    body_md = _normalize_whitespace(body_md)
    return body_md


def _build_embedding_text(section_header: str, clean_text: str, previous_page_tail: str = "") -> str:
    """
    Build the text used for embeddings.
    """
    parts = []

    if previous_page_tail:
        parts.append(f"Boundary context from previous page:\n{previous_page_tail}")

    if section_header and section_header != "Preamble":
        parts.append(f"Section: {section_header}")

    parts.append(clean_text)
    return "\n\n".join(parts).strip()


def _split_markdown_pages(body_md: str) -> list[tuple[int, str]]:
    """
    Split full markdown into exact pages using page separator:
        --- on its own line
    """
    raw_pages = re.split(r"^\s*---\s*$", body_md, flags=re.MULTILINE)
    pages = []

    for idx, page_text in enumerate(raw_pages, start=1):
        page_text = page_text.strip()
        if page_text:
            pages.append((idx, page_text))

    return pages


def _extract_page_tail_for_boundary(page_text: str, max_chars: int = BOUNDARY_CONTEXT_CHARS) -> str:
    """
    Extract a short tail from the previous page for boundary-aware retrieval.
    """
    cleaned = _normalize_whitespace(page_text)
    if not cleaned:
        return ""

    tail = cleaned[-max_chars:]
    tail = tail.lstrip(" ,.;:-")
    return tail.strip()


def _extract_figure_page_number(block: str):
    """
    Extract page number from stored figure block.
    Expected watcher format includes a line like:
        Page 7
    """
    patterns = [
        r"\bPage\s+(\d+)\b",
        r"\bpage\s+(\d+)\b",
        r"\bpage[_\-\s]?(\d+)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, block)
        if match:
            return int(match.group(1))

    return None


def _extract_mermaid_blocks(text: str):
    """
    Extract fenced mermaid blocks.
    Returns:
        cleaned_text, blocks
    """
    pattern = re.compile(r"```mermaid\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    blocks = pattern.findall(text)
    cleaned_text = pattern.sub("", text)
    return cleaned_text, [b.strip() for b in blocks if b.strip()]


def _extract_equation_blocks(text: str):
    """
    Extract display equations:
        $$ ... $$
    Returns:
        cleaned_text, blocks
    """
    pattern = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
    blocks = pattern.findall(text)
    cleaned_text = pattern.sub("", text)
    return cleaned_text, [b.strip() for b in blocks if b.strip()]


def _extract_markdown_tables(text: str):
    """
    Extract markdown tables as standalone blocks.
    Consecutive lines starting with '|' are treated as one table block.
    """
    lines = text.split("\n")
    kept = []
    tables = []
    current_table = []

    def flush_table():
        nonlocal current_table
        if current_table:
            table_block = "\n".join(current_table).strip()
            if table_block:
                tables.append(table_block)
            current_table = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            current_table.append(line)
        else:
            flush_table()
            kept.append(line)

    flush_table()

    cleaned_text = "\n".join(kept)
    return cleaned_text, tables


def _make_special_chunk(
    source_file: str,
    section_header: str,
    page_number: int | None,
    chunk_index: int,
    content_type: str,
    block_text: str,
):
    """
    Build one standalone special-content chunk.
    """
    clean_text = _normalize_whitespace(block_text)
    if not clean_text:
        return None

    embedding_text = _build_embedding_text(
        section_header=section_header,
        clean_text=clean_text,
        previous_page_tail="",
    )

    return {
        "text": clean_text,
        "embedding_text": embedding_text,
        "metadata": {
            "source_file": source_file,
            "section_header": section_header,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "content_type": content_type,
            "has_table": content_type == "table",
            "has_image_description": False,
            "continued_from_previous_page": False,
            "previous_page_number": None,
        },
    }

def _extract_figure_header_parts(fig_header: str) -> tuple[int | None, str]:
    """
    Parse:
        'Figure 3: Caption text'
    into:
        (3, 'Caption text')
    """
    match = re.match(r"Figure\s+(\d+)\s*:\s*(.+)", fig_header, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), match.group(2).strip()
    return None, fig_header.strip()

def chunk_markdown(markdown_text: str, source_file: str) -> list[dict]:
    """
    Convert a full markdown document into a list of richly-tagged chunk dicts
    ready for embedding and Qdrant upsert.

    Returns chunks shaped like:
        {
            "text": clean_text_for_payload,
            "embedding_text": text_used_for_embeddings,
            "metadata": {...}
        }
    """
    # ── Separate figure description section from body text ──────────────────
    figure_section_marker = "## Extracted Figures & Visual Descriptions"
    if figure_section_marker in markdown_text:
        body_md, figure_md = markdown_text.split(figure_section_marker, 1)
    else:
        body_md = markdown_text
        figure_md = ""

    # Clean body markdown before page splitting and chunking
    body_md = _clean_markdown_for_chunking(body_md)

    all_chunks = []
    chunk_index = 0

    # ── Step 1: Split document into exact pages first ───────────────────────
    pages = _split_markdown_pages(body_md)

    # Track latest meaningful section header across pages in case a section continues
    inherited_section_header = "Preamble"

    # ── Step 2: Chunk each page independently ───────────────────────────────
    for page_idx, (page_number, page_md) in enumerate(pages):
        previous_page_tail = ""
        previous_page_number = None

        if page_idx > 0:
            previous_page_number, previous_page_md = pages[page_idx - 1]
            previous_page_tail = _extract_page_tail_for_boundary(previous_page_md)

        header_chunks = _header_splitter.split_text(page_md)
        first_chunk_on_page = True

        for hc in header_chunks:
            section_header = _derive_section_header(hc.metadata)

            if section_header == "Preamble" and inherited_section_header != "Preamble":
                section_header = inherited_section_header
            elif section_header != "Preamble":
                inherited_section_header = section_header

            page_content = hc.page_content

            # Extract special blocks first
            page_content, mermaid_blocks = _extract_mermaid_blocks(page_content)
            page_content, equation_blocks = _extract_equation_blocks(page_content)
            page_content, table_blocks = _extract_markdown_tables(page_content)

            # Standalone table chunks
            for block in table_blocks:
                chunk = _make_special_chunk(
                    source_file=source_file,
                    section_header=section_header,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    content_type="table",
                    block_text=block,
                )
                if chunk:
                    all_chunks.append(chunk)
                    chunk_index += 1

            # Standalone diagram chunks
            for block in mermaid_blocks:
                chunk = _make_special_chunk(
                    source_file=source_file,
                    section_header=section_header,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    content_type="diagram_text",
                    block_text=block,
                )
                if chunk:
                    all_chunks.append(chunk)
                    chunk_index += 1

            # Standalone equation chunks
            for block in equation_blocks:
                chunk = _make_special_chunk(
                    source_file=source_file,
                    section_header=section_header,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    content_type="equation_block",
                    block_text=block,
                )
                if chunk:
                    all_chunks.append(chunk)
                    chunk_index += 1

            # Chunk remaining prose normally
            sub_chunks = _char_splitter.split_text(page_content)

            for sub_text in sub_chunks:
                sub_text = _normalize_whitespace(sub_text)
                if not sub_text:
                    continue

                # Skip chunks that are only headings with no real body
                if re.fullmatch(r"#{1,3}\s+.+", sub_text):
                    continue

                clean_text = sub_text

                # Skip tiny / non-alphabetic chunks
                if not clean_text or len(clean_text) < 5 or not any(c.isalpha() for c in clean_text):
                    continue

                apply_boundary_context = first_chunk_on_page and bool(previous_page_tail)

                embedding_text = _build_embedding_text(
                    section_header=section_header,
                    clean_text=clean_text,
                    previous_page_tail=previous_page_tail if apply_boundary_context else "",
                )

                metadata = {
                    "source_file": source_file,
                    "section_header": section_header,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "content_type": "text",
                    "has_table": _has_table(clean_text),
                    "has_image_description": False,
                    "continued_from_previous_page": apply_boundary_context,
                    "previous_page_number": previous_page_number if apply_boundary_context else None,
                }

                all_chunks.append({
                    "text": clean_text,
                    "embedding_text": embedding_text,
                    "metadata": metadata,
                })

                chunk_index += 1
                first_chunk_on_page = False

    # ── Step 3: Process figure descriptions as standalone chunks ────────────
    if figure_md.strip():
        figure_blocks = re.split(r"(?=### Figure \d+:)", figure_md.strip())

        for block in figure_blocks:
            block = block.strip()
            if not block:
                continue

            header_match = re.match(r"### (Figure \d+:[^\n]*)", block)
            fig_header = header_match.group(1).strip() if header_match else "Figure"

            figure_number, figure_caption = _extract_figure_header_parts(fig_header)

            clean_block = re.sub(r"^###\s+", "", block).strip()
            clean_block = _normalize_whitespace(clean_block)

            figure_page_number = _extract_figure_page_number(clean_block)

            embedding_text = _build_embedding_text(
                section_header=fig_header,
                clean_text=clean_block,
                previous_page_tail="",
            )

            all_chunks.append({
                "text": clean_block,
                "embedding_text": embedding_text,
                "metadata": {
                    "source_file": source_file,
                    "section_header": fig_header,
                    "page_number": figure_page_number,
                    "chunk_index": chunk_index,
                    "content_type": "figure_description",
                    "has_table": False,
                    "has_image_description": True,
                    "continued_from_previous_page": False,
                    "previous_page_number": None,
                    "figure_number": figure_number,
                    "figure_caption": figure_caption,
                },
            })
            chunk_index += 1

    print(
        f"[chunker] '{source_file}' -> {len(all_chunks)} chunks "
        f"({sum(1 for c in all_chunks if c['metadata']['content_type'] == 'figure_description')} figure description(s))"
    )

    # ── Debug: save chunks to JSON for inspection ───────────────────────────
    if DEBUG_SAVE_CHUNKS:
        debug_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "ingest_folder",
            os.path.splitext(source_file)[0] + "_chunks.json",
        )
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        print(f"[chunker] Debug chunks saved -> {os.path.abspath(debug_path)}")

    return all_chunks