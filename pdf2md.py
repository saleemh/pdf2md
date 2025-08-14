#!/usr/bin/env python3
"""
PDF to Markdown Converter

A simple CLI tool that converts any PDF document to Markdown using IBM Docling.
For large PDFs, it automatically segments them into smaller files for better processing.

Usage:
    pdf2md input.pdf

Features:
- Universal PDF conversion using IBM Docling
- Automatic segmentation for large PDFs
- Preserves document structure and formatting
- Outputs to the same directory as input file
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import re
import statistics
import argparse
import os

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("Warning: docling is not installed. The 'docling' engine will be unavailable.")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("Warning: PyPDF2 is not installed. Page counting will be approximate.")

# Docling VLM pipeline (optional)
try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions, ApiVlmOptions, ResponseFormat
    from docling.pipeline.vlm_pipeline import VlmPipeline
    DOCLING_VLM_AVAILABLE = True
except Exception:
    DOCLING_VLM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from project .env (adjacent to this file) if present
try:
    from dotenv import load_dotenv
    _DOTENV_PATH = Path(__file__).parent / '.env'
    load_dotenv(dotenv_path=_DOTENV_PATH, override=True)
except Exception:
    pass


class PDF2MD:
    """PDF to Markdown converter with Docling and optional VLM auto-routing."""

    def __init__(self, openai_mode: str = "auto", openai_model: Optional[str] = None):
        """Initialize the converter.

        openai_mode: 'always' | 'auto' | 'never'
        openai_model: optional; overrides model from env; defaults to 'gpt-4o'
        """
        if openai_mode not in {"always", "auto", "never"}:
            raise ValueError("openai_mode must be one of: 'always', 'auto', 'never'")
        self.openai_mode = openai_mode
        self.openai_model = openai_model

        if not DOCLING_AVAILABLE:
            raise Exception("Docling is not installed. Install with: pip install docling")

        # Standard Docling converter is always available
        self.converter_docling = DocumentConverter()
        self.converter_vlm: Optional[DocumentConverter] = None

        if self.openai_mode in {"always", "auto"} and DOCLING_VLM_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set; VLM will be unavailable. Falling back to Docling.")
            else:
                model_name = self.openai_model or os.getenv("DOCLING_VLM_OPENAI_MODEL") or "gpt-4o"
                prompt_text = os.getenv("DOCLING_VLM_PROMPT") or (
                    "Convert this page to GitHub-Flavored Markdown. Preserve headings, lists, tables, and code blocks. "
                    "Use proper table syntax. Do not add commentary. Output only Markdown."
                )
                pipeline_options = VlmPipelineOptions(
                    enable_remote_services=True,
                    vlm_options=ApiVlmOptions(
                        url="https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        params={"model": model_name},
                        prompt=prompt_text,
                        response_format=ResponseFormat.MARKDOWN,
                        timeout=120,
                        scale=1.0,
                    ),
                )
                self.converter_vlm = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_cls=VlmPipeline,
                            pipeline_options=pipeline_options,
                        )
                    }
                )
                logger.info(f"VLM converter initialized (model={model_name})")
    
    def convert_pdf(self, input_path: Path) -> List[Path]:
        """
        Convert PDF to markdown, with automatic segmentation for large files.
        
        Args:
            input_path: Path to the input PDF file
            
        Returns:
            List of output markdown file paths
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        logger.info(f"Converting: {input_path.name}")

        # Prepare output directories
        page_count = self._get_page_count(input_path)
        logger.info(f"PDF has {page_count} pages")

        output_root = input_path.parent / f"{input_path.stem}-output"
        fragments_dir = output_root / "pdf-fragments"
        assets_dir = output_root / "assets" / input_path.stem
        output_root.mkdir(parents=True, exist_ok=True)
        fragments_dir.mkdir(parents=True, exist_ok=True)
        assets_dir.mkdir(parents=True, exist_ok=True)

        output_files: List[Path] = []

        if page_count <= 50:
            logger.info("Processing as single document")
            output_file = output_root / f"{input_path.stem}.md"
            # Decide converter based on mode and content
            converter_to_use = self._select_converter_for_document(input_path)
            # Single-file conversion
            self._convert_single_pdf(
                input_path,
                output_file,
                header_meta={
                    "title": input_path.stem,
                    "source": input_path.name,
                    "pages": f"1-{page_count}",
                    "section": None,
                },
                converter=converter_to_use,
            )
            output_files.append(output_file)
        else:
            logger.info(f"Large PDF detected ({page_count} pages). Segmenting by TOC...")
            output_files = self._convert_toc_segmented_pdf(
                input_path=input_path,
                output_root=output_root,
                fragments_dir=fragments_dir,
            )

        logger.info(
            f"Conversion complete. Generated {len(output_files)} markdown file(s) in {output_root}"
        )
        return output_files
    
    def _get_page_count(self, pdf_path: Path) -> int:
        """Get the number of pages in the PDF."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return len(reader.pages)
        except Exception as e:
            logger.warning(f"Could not get exact page count: {e}")
            # Estimate based on file size (rough approximation)
            file_size = pdf_path.stat().st_size
            return max(1, file_size // 50000)
    
    def _convert_single_pdf(
        self,
        input_path: Path,
        output_path: Path,
        header_meta: Optional[Dict[str, Optional[str]]] = None,
        converter: Optional[Any] = None,
    ):
        """Convert a single PDF to markdown.

        header_meta fields (optional):
          - title: str
          - source: str (original PDF filename)
          - pages: str (e.g., "51-72")
          - section: str (e.g., "H2 (TOC)")
        """
        try:
            # Convert with selected converter (default to standard Docling)
            conv = converter or self.converter_docling
            try:
                result = conv.convert(input_path)
            except Exception as primary_error:
                # If VLM conversion fails, fall back to standard Docling automatically
                if conv is self.converter_vlm:
                    logger.warning(
                        f"VLM conversion failed: {primary_error}. Falling back to standard Docling."
                    )
                    conv = self.converter_docling
                    result = conv.convert(input_path)
                else:
                    raise

            # Extract markdown content with fallback if VLM export fails
            try:
                if hasattr(result, 'document') and hasattr(result.document, 'export_to_markdown'):
                    markdown_content = result.document.export_to_markdown()
                else:
                    raise Exception("Could not extract markdown from conversion result")
            except Exception as export_error:
                if conv is self.converter_vlm:
                    logger.warning(
                        f"VLM export failed: {export_error}. Falling back to standard Docling."
                    )
                    conv = self.converter_docling
                    result = conv.convert(input_path)
                    if hasattr(result, 'document') and hasattr(result.document, 'export_to_markdown'):
                        markdown_content = result.document.export_to_markdown()
                    else:
                        raise Exception("Could not extract markdown from Docling conversion result")
                else:
                    raise
            
            # Add document header
            if header_meta is None:
                pages_text = f"1-{self._get_page_count(input_path)}"
                title_text = input_path.stem
                source_text = input_path.name
                section_text = None
            else:
                pages_text = header_meta.get("pages") or f"1-{self._get_page_count(input_path)}"
                title_text = header_meta.get("title") or input_path.stem
                source_text = header_meta.get("source") or input_path.name
                section_text = header_meta.get("section")

            header_lines = [
                f"# {title_text}",
                "",
                "## Document Information",
                f"- **Source**: {source_text}",
                f"- **Pages**: {pages_text}",
                "- **Converted**: PDF2MD",
                "- **Processing**: IBM Docling",
            ]
            if section_text:
                header_lines.insert(5, f"- **Section**: {section_text}")
            header_lines += ["", "---", "", ""]
            header = "\n".join(header_lines)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(header + markdown_content)
            
            logger.info(f"Created: {output_path.name}")
            
        except Exception as e:
            logger.error(f"Error converting PDF: {e}")
            raise
    
    def _convert_toc_segmented_pdf(
        self,
        input_path: Path,
        output_root: Path,
        fragments_dir: Path,
    ) -> List[Path]:
        """Convert a large PDF using TOC/bookmark segmentation with auto-selected level."""
        output_files: List[Path] = []
        page_count = self._get_page_count(input_path)

        # Read outline/TOC with compatibility across PyPDF2 versions
        outline = None
        try:
            with open(input_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                # Prefer singular 'outline' first (PyPDF2 >= 3)
                try:
                    outline = reader.outline  # may raise in older/newer versions
                except Exception:
                    outline = None
                # Fallback to 'outlines' (older versions)
                if not outline:
                    try:
                        outline = reader.outlines  # may raise in PyPDF2 3.x
                    except Exception:
                        outline = None
        except Exception as e:
            logger.warning(f"Could not read PDF outlines: {e}")
            outline = None

        if not outline:
            logger.info("No TOC found. Falling back to fixed 50-page segmentation.")
            return self._convert_fixed_segments(input_path, output_root, fragments_dir, segment_size=50)

        nodes = self._build_outline_nodes(outline, input_path)
        logger.info(f"Found {len(nodes)} TOC entries after normalization")
        if not nodes:
            logger.info("Empty TOC after normalization. Falling back to fixed 50-page segmentation.")
            return self._convert_fixed_segments(input_path, output_root, fragments_dir, segment_size=50)

        # Auto-select level
        level = self._auto_select_toc_level(nodes, page_count)
        logger.info(f"Selected TOC level: {level}")

        # Compute segments
        segments = self._compute_toc_segments(nodes, level, page_count)
        # Optional: merge tiny trailing segments
        merged_segments = self._merge_tiny_segments(segments)

        # Naming counters
        md_paths: List[Path] = []
        for seg in merged_segments:
            seg_id = seg.get('id')
            title = seg.get('title') or 'untitled'
            slug = self._slugify(title)
            start_page = seg['start_page']
            end_page = seg['end_page']
            level_str = f"H{seg['level']} (TOC)"

            md_name = f"{seg_id}-{slug}_{start_page:03d}-{end_page:03d}.md"
            pdf_name = f"{seg_id}-{slug}_{start_page:03d}-{end_page:03d}.pdf"

            pdf_segment_path = fragments_dir / pdf_name
            md_segment_path = output_root / md_name

            logger.info(f"Processing segment: {seg_id} {title} (pages {start_page}-{end_page})")

            try:
                # Clamp invalid page ranges defensively
                if end_page < start_page:
                    end_page = start_page
                self._create_pdf_segment(input_path, pdf_segment_path, start_page, end_page)
                # Decide converter for this segment
                converter_to_use = self._select_converter_for_segment(input_path, start_page, end_page)
                self._convert_single_pdf(
                    pdf_segment_path,
                    md_segment_path,
                    header_meta={
                        "title": title,
                        "source": input_path.name,
                        "pages": f"{start_page}-{end_page}",
                        "section": level_str,
                    },
                    converter=converter_to_use,
                )
                output_files.append(md_segment_path)
            except Exception as e:
                logger.error(f"Error processing segment {seg_id} ({start_page}-{end_page}): {e}")
                continue

        # Write index.md
        try:
            self._write_index(
                output_root=output_root,
                input_file=input_path.name,
                page_count=page_count,
                segments=merged_segments,
                chosen_level=level,
            )
        except Exception as e:
            logger.warning(f"Failed to write index.md: {e}")

        return output_files
    
    def _create_pdf_segment(self, source_path: Path, output_path: Path, start_page: int, end_page: int):
        """Create a PDF segment containing only the specified page range."""
        try:
            with open(source_path, 'rb') as input_file:
                reader = PyPDF2.PdfReader(input_file)
                writer = PyPDF2.PdfWriter()
                
                # Add pages (convert to 0-based indexing)
                for page_num in range(start_page - 1, min(end_page, len(reader.pages))):
                    if page_num < len(reader.pages):
                        writer.add_page(reader.pages[page_num])
                
                # Write segment file
                with open(output_path, 'wb') as output_file:
                    writer.write(output_file)
                    
        except Exception as e:
            # Clean up partial file on error
            if output_path.exists():
                output_path.unlink()
            raise e

    # ---------------------------
    # TOC helpers
    # ---------------------------
    def _build_outline_nodes(self, outline: Any, pdf_path: Path) -> List[Dict[str, Any]]:
        """Flatten PyPDF2 outline into a list of nodes with title, level, start_page (1-based).

        Handles PyPDF2 2.x nested lists and 3.x OutlineItem structures.
        """
        nodes: List[Dict[str, Any]] = []
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)

                def resolve_page_index0(it: Any) -> Optional[int]:
                    # 1) Explicit .page attribute
                    page_obj = getattr(it, 'page', None)
                    if page_obj is not None:
                        try:
                            return list(reader.pages).index(page_obj)
                        except Exception:
                            pass
                    # 2) Nested dest.page
                    dest = getattr(it, 'dest', None)
                    if dest is not None:
                        page_obj2 = getattr(dest, 'page', None)
                        if page_obj2 is not None:
                            try:
                                return list(reader.pages).index(page_obj2)
                            except Exception:
                                pass
                    # 3) Try legacy resolver
                    try:
                        return reader.get_destination_page_number(it)
                    except Exception:
                        return None

                def resolve_title_text(it: Any) -> str:
                    title = getattr(it, 'title', None)
                    if isinstance(title, str) and title.strip():
                        return title.strip()
                    # Some structures store title on dict-like object
                    try:
                        t = str(it)
                        return t.strip() if t else 'Untitled'
                    except Exception:
                        return 'Untitled'

                def iter_children(it: Any) -> List[Any]:
                    # OutlineItem may be iterable or have .children
                    ch = getattr(it, 'children', None)
                    if ch:
                        try:
                            return list(ch)
                        except Exception:
                            pass
                    try:
                        # Avoid treating strings/bytes as iterable of chars
                        if isinstance(it, (str, bytes)):
                            return []
                        lst = list(it)
                        return lst
                    except Exception:
                        return []

                def walk(items: Any, level: int):
                    # items can be a list or a single OutlineItem
                    if isinstance(items, list):
                        for sub in items:
                            walk(sub, level)
                        return
                    item = items
                    # If item is a list, it's a deeper level container
                    if isinstance(item, list):
                        walk(item, level + 1)
                        return
                    # Capture this item
                    page_idx0 = resolve_page_index0(item)
                    if page_idx0 is not None:
                        nodes.append({
                            'title': resolve_title_text(item),
                            'level': level,
                            'start_page': page_idx0 + 1,
                        })
                    # Recurse into children
                    children = iter_children(item)
                    if children:
                        for ch in children:
                            walk(ch, level + 1)

                walk(outline, 1)

            # Normalize ordering: by start_page then by level (shallow first)
            nodes.sort(key=lambda n: (n['start_page'], n['level']))

            # Dedupe identical page+level by keeping first (shallower already sorted first)
            seen = set()
            deduped: List[Dict[str, Any]] = []
            for n in nodes:
                key = (n['start_page'], n['level'], n['title'])
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(n)
            return deduped
        except Exception as e:
            logger.warning(f"Failed to parse outline nodes: {e}")
            return []

    def _auto_select_toc_level(self, nodes: List[Dict[str, Any]], page_count: int) -> int:
        """Choose a TOC level that yields a balanced number and size of segments."""
        if not nodes:
            return 1

        max_level = max(n['level'] for n in nodes)
        best_level = 1
        best_score = float('inf')

        def score_for_level(L: int) -> float:
            seg_ranges = self._segments_for_level(nodes, L, page_count)
            counts = len(seg_ranges)
            sizes = [end - start + 1 for start, end in seg_ranges]
            median_size = statistics.median(sizes) if sizes else 0
            variance = statistics.pvariance(sizes) if len(sizes) > 1 else 0.0
            # Penalties
            # target segments 8..40
            if counts < 8:
                count_penalty = (8 - counts) * 5
            elif counts > 40:
                count_penalty = (counts - 40) * 5
            else:
                count_penalty = 0
            # target size 5..30
            if median_size < 5:
                size_penalty = (5 - median_size) * 2
            elif median_size > 30:
                size_penalty = (median_size - 30) * 2
            else:
                size_penalty = 0
            variance_penalty = min(variance / 10.0, 1000)
            return count_penalty + size_penalty + variance_penalty

        for L in range(1, min(max_level, 6) + 1):
            s = score_for_level(L)
            if s < best_score or (abs(s - best_score) < 1e-6 and L > best_level):
                best_level, best_score = L, s
        return best_level

    def _segments_for_level(self, nodes: List[Dict[str, Any]], L: int, page_count: int) -> List[tuple]:
        starts = [n['start_page'] for n in nodes if n['level'] <= L]
        starts = sorted(set(starts))
        if not starts or starts[0] > 1:
            starts = [1] + starts
        ranges = []
        for i, start in enumerate(starts):
            end = (starts[i + 1] - 1) if i + 1 < len(starts) else page_count
            if start <= end:
                ranges.append((start, end))
        return ranges

    def _compute_toc_segments(
        self,
        nodes: List[Dict[str, Any]],
        level: int,
        page_count: int,
    ) -> List[Dict[str, Any]]:
        """Compute segments including ids and titles for chosen level."""
        # Prepare numbering counters per level
        counters: Dict[int, int] = {}
        segments: List[Dict[str, Any]] = []

        # Build a mapping from start_page to (title, level)
        key_nodes = [n for n in nodes if n['level'] <= level]
        # Ensure front matter if first node starts > 1
        starts = [n['start_page'] for n in key_nodes]
        if not starts or (starts and starts[0] > 1):
            # front matter segment
            segments.append({
                'id': '00',
                'title': 'Front matter',
                'level': 0,
                'start_page': 1,
                'end_page': max(1, (starts[0] - 1) if starts else page_count),
            })

        # Compute numbered segments
        sorted_nodes = sorted(key_nodes, key=lambda n: (n['start_page'], n['level']))
        for idx, n in enumerate(sorted_nodes):
            # Determine end page based on next same-or-higher level start
            next_start = None
            for m in sorted_nodes[idx + 1:]:
                if m['level'] <= n['level']:
                    next_start = m['start_page']
                    break
            start_page = n['start_page']
            end_page = (next_start - 1) if next_start is not None else page_count
            # Guard against zero/negative-length segments when successive nodes share the same start page
            if end_page < start_page:
                end_page = start_page

            # Update counters
            lvl = n['level']
            # Reset deeper levels
            for deeper in list(counters.keys()):
                if deeper >= lvl + 1:
                    counters[deeper] = 0
            counters[lvl] = counters.get(lvl, 0) + 1

            # Build id string (up to chosen level)
            id_parts: List[str] = []
            for l in range(1, lvl + 1):
                id_parts.append(f"{counters.get(l, 0):02d}")
            seg_id = ".".join(id_parts)

            segments.append({
                'id': seg_id,
                'title': n['title'],
                'level': lvl,
                'start_page': start_page,
                'end_page': end_page,
            })

        return segments

    def _merge_tiny_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not segments:
            return []
        merged: List[Dict[str, Any]] = []
        for seg in segments:
            if merged and seg['end_page'] - seg['start_page'] + 1 < 3:
                # try merge with previous if does not exceed 60 pages
                prev = merged[-1]
                if (seg['end_page'] - prev['start_page'] + 1) <= 60:
                    prev['end_page'] = seg['end_page']
                    prev['title'] = prev['title']  # keep previous title
                    continue
            merged.append(seg)
        return merged

    def _write_index(
        self,
        output_root: Path,
        input_file: str,
        page_count: int,
        segments: List[Dict[str, Any]],
        chosen_level: int,
    ) -> None:
        lines: List[str] = []
        lines.append(f"# {Path(input_file).stem} — TOC")
        lines.append("")
        lines.append("## Document Information")
        lines.append(f"- **Source**: {input_file}")
        lines.append(f"- **Pages**: 1–{page_count}")
        lines.append("- **Split mode**: TOC auto-level")
        lines.append("- **Converted**: PDF2MD (Docling)")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Render segments in order
        for seg in segments:
            title = seg['title']
            start_page = seg['start_page']
            end_page = seg['end_page']
            seg_id = seg['id']
            slug = self._slugify(title)
            if seg['level'] == 0:  # front matter
                filename = f"{seg_id}-{slug}_{start_page:03d}-{end_page:03d}.md"
                lines.append(f"- [{seg_id} {title}]({filename}) — pages {start_page}–{end_page}")
                continue
            filename = f"{seg_id}-{slug}_{start_page:03d}-{end_page:03d}.md"
            indent = "  " * max(0, seg['level'] - 1)
            if seg['level'] <= chosen_level:
                lines.append(f"{indent}- [{seg_id} {title}]({filename}) — pages {start_page}–{end_page}")
            else:
                lines.append(f"{indent}- {seg_id} {title} (in section)")

        index_path = output_root / "index.md"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")

    def _slugify(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\-\s]", "", text)
        text = re.sub(r"\s+", "-", text)
        text = re.sub(r"-+", "-", text)
        text = text.strip('-')
        return text[:60] if len(text) > 60 else text

    def _convert_fixed_segments(
        self,
        input_path: Path,
        output_root: Path,
        fragments_dir: Path,
        segment_size: int = 50,
    ) -> List[Path]:
        """Fallback: fixed-size page segmentation, storing outputs in new layout."""
        page_count = self._get_page_count(input_path)
        output_files: List[Path] = []
        for i in range(0, page_count, segment_size):
            start_page = i + 1
            end_page = min(i + segment_size, page_count)
            title = f"Pages {start_page}-{end_page}"
            seg_id = f"{(i // segment_size) + 1:02d}"
            slug = self._slugify(title)
            md_name = f"{seg_id}-{slug}_{start_page:03d}-{end_page:03d}.md"
            pdf_name = f"{seg_id}-{slug}_{start_page:03d}-{end_page:03d}.pdf"
            pdf_segment_path = fragments_dir / pdf_name
            md_segment_path = output_root / md_name
            try:
                self._create_pdf_segment(input_path, pdf_segment_path, start_page, end_page)
                # Decide converter for this segment
                converter_to_use = self._select_converter_for_segment(input_path, start_page, end_page)
                self._convert_single_pdf(
                    pdf_segment_path,
                    md_segment_path,
                    header_meta={
                        "title": title,
                        "source": input_path.name,
                        "pages": f"{start_page}-{end_page}",
                        "section": None,
                    },
                    converter=converter_to_use,
                )
                output_files.append(md_segment_path)
            except Exception as e:
                logger.error(f"Error processing segment {start_page}-{end_page}: {e}")
                continue
        # Write index
        self._write_index(
            output_root=output_root,
            input_file=input_path.name,
            page_count=page_count,
            segments=[
                {
                    'id': f"{(i // segment_size) + 1:02d}",
                    'title': f"Pages {i + 1}-{min(i + segment_size, page_count)}",
                    'level': 1,
                    'start_page': i + 1,
                    'end_page': min(i + segment_size, page_count),
                }
                for i in range(0, page_count, segment_size)
            ],
            chosen_level=1,
        )
        return output_files

    # ---------------------------
    # Auto-routing helpers
    # ---------------------------
    def _select_converter_for_document(self, pdf_path: Path) -> Any:
        if self.openai_mode == "never" or not self.converter_vlm:
            return self.converter_docling
        if self.openai_mode == "always":
            return self.converter_vlm
        # auto mode: inspect document
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
                pages_with_images_and_low_text = 0
                for i in range(num_pages):
                    page = reader.pages[i]
                    text_len = len((page.extract_text() or "").strip())
                    if self._page_has_images(reader, i) and text_len < 50:
                        pages_with_images_and_low_text += 1
                ratio = pages_with_images_and_low_text / max(1, num_pages)
                return self.converter_vlm if ratio >= 0.2 else self.converter_docling
        except Exception:
            return self.converter_docling

    def _select_converter_for_segment(self, pdf_path: Path, start_page: int, end_page: int) -> Any:
        if self.openai_mode == "never" or not self.converter_vlm:
            return self.converter_docling
        if self.openai_mode == "always":
            return self.converter_vlm
        # auto mode: inspect pages in range (1-based)
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                start_idx = max(0, start_page - 1)
                end_idx = min(len(reader.pages) - 1, end_page - 1)
                for i in range(start_idx, end_idx + 1):
                    page = reader.pages[i]
                    text_len = len((page.extract_text() or "").strip())
                    if self._page_has_images(reader, i) and text_len < 50:
                        return self.converter_vlm
                return self.converter_docling
        except Exception:
            return self.converter_docling

    def _page_has_images(self, reader: Any, page_index: int) -> bool:
        try:
            page = reader.pages[page_index]
            resources = page.get('/Resources')
            if not resources:
                return False
            xobj = resources.get('/XObject')
            if not xobj:
                return False
            for name in xobj.keys():
                try:
                    obj = xobj[name].get_object()
                    subtype = obj.get('/Subtype')
                    if subtype and str(subtype) == '/Image':
                        return True
                except Exception:
                    continue
            return False
        except Exception:
            return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Convert PDF documents to Markdown using IBM Docling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pdf2md document.pdf              Convert a single PDF
  pdf2md "my document.pdf"         Convert PDF with spaces in filename
  pdf2md /path/to/document.pdf     Convert PDF with full path
  pdf2md /path/to/folder           Convert all PDFs in a folder (non-recursive)

Notes:
  - Output files are created in the same directory as the input PDF
  - Large PDFs (>50 pages) are automatically split into segments
  - Both PDF segments and markdown files are kept for reference
  - OpenAI usage via Docling's VLM defaults to 'auto' routing; override with --openai-mode
        """
    )
    
    parser.add_argument(
        'pdf_file',
        help='Path to a PDF file or a directory containing PDFs (non-recursive)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--openai-mode',
        choices=['always', 'auto', 'never'],
        help="When to use OpenAI via Docling VLM: 'always', 'auto' (default), or 'never'"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert path argument
    input_path = Path(args.pdf_file).resolve()
    
    try:
        # Initialize converter
        # Determine whether to use OpenAI from CLI or environment (default: True)
        def _env_truthy(value: Optional[str]) -> Optional[bool]:
            if value is None:
                return None
            return value.strip().lower() in {"1", "true", "yes", "on"}

        env_val = _env_truthy(os.getenv('USE_OPENAI'))
        env_mode_raw = (os.getenv('OPENAI_MODE') or '').strip().lower()
        env_mode = env_mode_raw if env_mode_raw in {'always','auto','never'} else None
        # Default mode is 'auto' unless overridden by CLI or env
        if args.openai_mode:
            openai_mode = args.openai_mode
        elif env_mode is not None:
            openai_mode = env_mode
        else:
            if env_val is True:
                openai_mode = 'always'
            elif env_val is False:
                openai_mode = 'never'
            else:
                openai_mode = 'auto'

        converter = PDF2MD(openai_mode=openai_mode)
        
        if input_path.is_dir():
            pdf_files = sorted([
                p for p in input_path.iterdir()
                if p.is_file() and p.suffix.lower() == '.pdf'
            ])
            if not pdf_files:
                print(f"\n❌ No PDF files found in folder: {input_path}")
                sys.exit(1)
            total_outputs = 0
            processed = 0
            print(f"\n📂 Found {len(pdf_files)} PDF(s) in folder: {input_path}")
            for pdf_file in pdf_files:
                try:
                    outputs = converter.convert_pdf(pdf_file)
                    total_outputs += len(outputs)
                    processed += 1
                except Exception as e:
                    print(f"❌ Error converting {pdf_file.name}: {e}")
                    continue
            print(f"\n✅ Batch conversion completed!")
            print(f"📦 Folder: {input_path}")
            print(f"📄 PDFs processed: {processed}/{len(pdf_files)}")
            print(f"📝 Total markdown files generated: {total_outputs}")
        else:
            output_files = converter.convert_pdf(input_path)
            
            # Print results
            print(f"\n✅ Conversion completed successfully!")
            print(f"📄 Input: {input_path.name}")
            print(f"📁 Output directory: {input_path.parent / (input_path.stem + '-output')}")
            print(f"📝 Generated {len(output_files)} markdown file(s):")
            
            for output_file in output_files:
                print(f"   - {output_file.name}")
        
    except KeyboardInterrupt:
        print("\n❌ Conversion cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()