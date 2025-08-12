# Phase 2 Additions Plan

This plan updates the tool to always use TOC/bookmark-based segmentation for large PDFs (>50 pages), automatically choosing an appropriate TOC depth to produce a readable, well-sized set of sections. Outputs are organized in a dedicated folder per input file, retaining the PDF fragments alongside the Markdown exports. No new CLI flags will be introduced.

## Goals and Constraints
- Always-on TOC mode for large PDFs: if page_count > 50, use TOC/bookmarks to segment; otherwise, single-file conversion.
- Auto-select the TOC level for a “nice” breakdown (no manual flag).
- Keep PDF fragments; they are not temporary.
- All outputs live under an output directory named `<input-stem>-output` in the same folder as the input PDF.
  - Subfolder: `pdf-fragments/` contains the segment PDFs.
  - Markdown files are written at the root of the output folder.
  - Assets (images, etc.) remain under `assets/<input-stem>/` (created within the output folder) to keep relative links stable.
- Maintain simplicity: no new flags/options.

## High-Level Flow
1. Count pages using `PyPDF2`. If `page_count <= 50`: convert to a single Markdown file and write it to `<input-stem>-output/>`.
2. If `page_count > 50`: build a TOC from PDF bookmarks (outlines).
3. Auto-select a TOC split level that yields a balanced number of segments and reasonable pages per segment.
4. Compute page ranges for each chosen TOC node (start page inclusive, end page inclusive).
5. For each segment:
   - Create a PDF fragment for the page range and save to `<output>/pdf-fragments/`.
   - Convert the fragment to Markdown with Docling and save to `<output>/`.
6. Generate an `index.md` in `<output>/` mirroring the TOC with links to the segment files and page ranges.

## TOC Extraction and Normalization
- Use `PyPDF2.PdfReader(...).outlines` (or compatible API) to get the outline tree.
- Normalize to a flat list of nodes: `{ title, level, start_page_0_based }`, preserving hierarchical order.
- Sort nodes by `(start_page, level)` to ensure stable reading order.
- Remove duplicate nodes that resolve to the same `(start_page, level)`; prefer shallower level when duplicates exist.

## Auto-Selecting the TOC Level
Aim for a “nice” breakdown: not too many files, not too few, and segments that are neither tiny nor massive.

Heuristic:
- For each candidate level L in [1..max_level_found]:
  - Consider nodes whose level <= L as segment starts.
  - Compute page ranges by taking each node’s start page and the next node’s start page − 1 (or end at total pages).
  - Score(L) by combining:
    - `segment_count_penalty`: prefer 8–40 segments (penalty grows outside this range).
    - `segment_size_penalty`: prefer median pages/segment between 5–30.
    - `variance_penalty`: prefer lower variance in segment sizes.
  - Choose the level with minimal total penalty. If ties, prefer deeper level (finer granularity) within acceptable range.
- Edge refinements:
  - Merge trailing ultra-short segments (<3 pages) into previous unless it would create >60 pages.
  - If the first TOC entry starts after page 1, create a “Front matter” segment (pages 1..start-1).

## Segment Page Range Computation
- Given selected level L:
  - `start_page = node.start_page + 1` (convert to 1-based for filenames/headers).
  - `end_page = next_same_or_higher_level_node.start_page` (1-based) − 1, else `page_count`.
- Ensure `start_page <= end_page` after merges.

## File/Folder Layout
- Output root: `<input-dir>/<input-stem>-output/`
  - Markdown segments (files): directly in output root.
  - PDF fragments: `<output>/pdf-fragments/`
  - Assets: `<output>/assets/<input-stem>/` (Docling output is directed here or moved/collated post-conversion so relative links in MD are stable).

## Naming Conventions
- Hierarchical segment id based on the TOC numbering at the chosen level:
  - Level 1 files: `01-<slug>_<start>-<end>.md`
  - Level 2 files: `01.02-<slug>_<start>-<end>.md`
  - Similarly for PDFs (in `pdf-fragments/`): `01-<slug>_<start>-<end>.pdf` or `01.02-<slug>_<start>-<end>.pdf`.
- `slug` is a safe, lowercase, dash-separated version of the title (ASCII, deduped, max ~60 chars). If duplicate slugs occur under the same parent, append a short counter (`-2`, `-3`, …).
- Zero-padding width is derived from the maximum count at each level to keep lexicographic sort aligned with reading order.

## Markdown Header for Each Segment
Include a brief header at the top of each segment file:

```
# <Section Title>

## Document Information
- **Source**: <input-file>.pdf
- **Pages**: <start>–<end>
- **Section**: H<level> (TOC)
- **Converted**: PDF2MD (Docling)

---
```

## index.md Construction
- Location: `<output>/index.md`
- Content:
  - Title and metadata for the whole document (source filename, total pages, split strategy).
  - Hierarchical list mirroring the TOC up to the chosen level, with links and page ranges.
  - Deeper levels (below chosen level) are listed as nested bullets under their parent marked as “in section” without separate files, or linked with an in-page anchor if headings are emitted.
- Example snippet:

```
# <PDF Title> — TOC

## Document Information
- **Source**: <input-file>.pdf
- **Pages**: 1–<total-pages>
- **Split mode**: TOC auto-level
- **Converted**: PDF2MD (Docling)

---

- [01 Introduction](01-introduction_001-010.md) — pages 1–10
  - 1.1 Motivation (in section)
  - 1.2 Scope (in section)
- [02 Installation](02-installation_011-020.md) — pages 11–20
  - [02.01 Getting Started](02.01-getting-started_011-014.md) — pages 11–14
  - [02.02 Advanced Setup](02.02-advanced-setup_015-020.md) — pages 15–20
```

## Fallbacks and Edge Cases
- No TOC/empty outlines:
  - Fallback order (still without flags): Headings-based segmentation; if unavailable, fixed 50-page segments as today.
- Non-monotonic or duplicate destinations: sort by page, dedupe by keeping the shallower level.
- Micro-sections: merge forward/backward per the merge rule described above.
- Extremely long sections: if any segment exceeds ~80 pages, prefer a deeper effective level (if available) or perform an internal sub-split by headings while keeping a single PDF fragment (optional variant: also split the fragment by pages with consistent naming `01a`, `01b`).

## Implementation Notes (no new flags)
- Existing `convert_pdf` becomes the orchestrator:
  - If `page_count <= 50`: write single MD into `<output>/` and any assets into `<output>/assets/<stem>/`.
  - Else: execute TOC pipeline, store PDFs in `<output>/pdf-fragments/`, MDs in `<output>/`, and assets under `<output>/assets/<stem>/`.
- Reuse `_create_pdf_segment` for fragment creation.
- Introduce internal helpers (no public CLI changes):
  - `_build_outline_nodes(reader) -> List[Node]`
  - `_auto_select_toc_level(nodes, page_count) -> int`
  - `_compute_toc_segments(nodes, level, page_count) -> List[Segment]`
  - `_slugify(title) -> str`
  - `_write_index(output_dir, segments, meta)`

## Validation and Testing
- Test with PDFs featuring: rich TOCs, shallow TOCs, no TOC, and TOCs with many small entries.
- Validate that:
  - Segment counts look reasonable and ordering matches the original TOC.
  - Page ranges are continuous and cover the entire document without overlap (except intentional front matter).
  - Links in `index.md` work; assets render in MD.

## Risks and Mitigations
- Inconsistent bookmark structures: normalize, sort by page, and dedupe.
- Asset path correctness: standardize a single `assets/<stem>/` root inside the output directory and ensure Docling uses or is post-processed to this location.
- Very deep TOCs: cap the effective level using the heuristic to avoid hundreds of files.

---

Reference: `saleemh/pdf2md` on GitHub (repo link: https://github.com/saleemh/pdf2md)
