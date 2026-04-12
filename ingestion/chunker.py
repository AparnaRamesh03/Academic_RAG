"""
chunker.py — Hierarchical Semantic Chunking for Academic Markdown Documents

Pipeline:
    1. MarkdownHeaderTextSplitter  → header-aware primary split
    2. RecursiveCharacterTextSplitter → max-token secondary split with overlap
    3. Metadata tagging per chunk (source_file, section_header, chunk_index, etc.)
    4. Figure description chunks are treated as standalone content_type="figure_description"
"""

import re
import json
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# ── Chunking Parameters ─────────────────────────────────────────────────────
CHUNK_SIZE    = 512   # max characters per chunk (approx 128–150 tokens for dense text)
CHUNK_OVERLAP = 64    # overlap between adjacent chunks to preserve continuity

# --- TOGGLE: Set to True to save chunks to a JSON file for inspection ---
DEBUG_SAVE_CHUNKS = True

# Headers to split on — ordered from broadest to narrowest
HEADERS_TO_SPLIT = [
    ("#",   "h1"),
    ("##",  "h2"),
    ("###", "h3"),
]

# ── Splitters ────────────────────────────────────────────────────────────────
_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT,
    strip_headers=True,   # headings are captured in section_header metadata; don't duplicate in chunk body
)

_char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],  # prefer paragraph → line → sentence breaks
)


def _has_table(text: str) -> bool:
    """Detect if a chunk contains a Markdown table."""
    return bool(re.search(r"^\|.+\|", text, re.MULTILINE))


def _has_image_description(text: str) -> bool:
    """Detect if a chunk contains an AI-generated image description."""
    return "**Visual Description (AI-generated):**" in text


def _derive_section_header(chunk_metadata: dict) -> str:
    """Build a readable section path from LangChain header metadata, e.g. 'Introduction > Background'."""
    parts = []
    for key in ["h1", "h2", "h3"]:
        val = chunk_metadata.get(key, "").strip()
        if val:
            parts.append(val)
    return " > ".join(parts) if parts else "Preamble"


# ── Public API ───────────────────────────────────────────────────────────────
def chunk_markdown(markdown_text: str, source_file: str) -> list[dict]:
    """
    Convert a full markdown document into a list of richly-tagged chunk dicts
    ready for embedding and Qdrant upsert.

    Args:
        markdown_text: Full markdown string from LlamaParse.
        source_file:   Original PDF filename (used for citation/metadata).

    Returns:
        List of dicts, each with 'text' and 'metadata' keys.
    """
    # ── Separate figure description section from body text ───────────────────
    figure_section_marker = "## Extracted Figures & Visual Descriptions"
    if figure_section_marker in markdown_text:
        body_md, figure_md = markdown_text.split(figure_section_marker, 1)
    else:
        body_md = markdown_text
        figure_md = ""

    # ── Step 1: Replace page separators with markers to preserve cross-page context ──
    # "---" = page break. We inject a parseable marker instead of hard-splitting,
    # so text that continues across a page boundary flows naturally through the splitters.
    page_marker_pattern = re.compile(r"^\s*---\s*$", re.MULTILINE)
    page_num = 1

    def _replace_page_break(m: re.Match) -> str:
        nonlocal page_num
        page_num += 1
        return f"\n<!-- PAGE {page_num} -->\n"

    annotated_body = page_marker_pattern.sub(_replace_page_break, body_md)

    # ── Step 2: Full-document header + char splitting (context flows across pages) ──
    all_chunks = []
    chunk_index = 0
    current_page = 1   # stateful page tracker — persists across all sub-chunks

    header_chunks = _header_splitter.split_text(annotated_body)

    for hc in header_chunks:
        section_header = _derive_section_header(hc.metadata)
        sub_chunks = _char_splitter.split_text(hc.page_content)

        for sub_text in sub_chunks:
            sub_text = sub_text.strip()
            if not sub_text:
                continue

            # Skip chunks that are only a heading with no body content
            if re.fullmatch(r"#{1,3}\s+.+", sub_text):
                continue

            # Update page tracker if this sub-chunk contains a page marker
            markers = re.findall(r"<!-- PAGE (\d+) -->", sub_text)
            if markers:
                current_page = int(markers[-1])   # use the last marker seen in this chunk

            # Strip all page markers from the stored text
            clean_text = re.sub(r"<!-- PAGE \d+ -->", "", sub_text).strip()
            
            # Skip empty chunks, chunks < 5 chars, or chunks lacking alphabetic characters (e.g. isolated punctuation)
            if not clean_text or len(clean_text) < 5 or not any(c.isalpha() for c in clean_text):
                continue

            all_chunks.append({
                "text": clean_text,
                "metadata": {
                    "source_file":    source_file,
                    "section_header": section_header,
                    "page_number":    current_page,
                    "chunk_index":    chunk_index,
                    "content_type":   "text",
                    "has_table":      _has_table(clean_text),
                    "has_image_description": False,
                },
            })
            chunk_index += 1

    # ── Step 3: Process figure descriptions as standalone chunks ──────────────
    if figure_md.strip():
        # Each figure block is separated by "\n\n### Figure N:"
        figure_blocks = re.split(r"(?=### Figure \d+:)", figure_md.strip())
        for block in figure_blocks:
            block = block.strip()
            if not block:
                continue

            # Extract figure header as section_header
            header_match = re.match(r"### (Figure \d+:[^\n]*)", block)
            fig_header = header_match.group(1).strip() if header_match else "Figure"

            all_chunks.append({
                "text": block,
                "metadata": {
                    "source_file":    source_file,
                    "section_header": fig_header,
                    "chunk_index":    chunk_index,
                    "content_type":   "figure_description",
                    "has_table":      False,
                    "has_image_description": True,
                },
            })
            chunk_index += 1

    print(f"[chunker] '{source_file}' → {len(all_chunks)} chunks "
          f"({sum(1 for c in all_chunks if c['metadata']['content_type'] == 'figure_description')} figure description(s))")

    # ── Debug: save chunks to JSON for inspection ────────────────────────────
    if DEBUG_SAVE_CHUNKS:
        debug_path = source_file.replace(".pdf", "_chunks.json").replace(".PDF", "_chunks.json")
        # Save to ingest_folder alongside the markdown
        import os
        debug_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "ingest_folder",
            os.path.splitext(source_file)[0] + "_chunks.json"
        )
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        print(f"[chunker] Debug chunks saved -> {os.path.abspath(debug_path)}")

    return all_chunks
