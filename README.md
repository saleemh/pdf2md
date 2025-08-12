# PDF2MD - PDF to Markdown Converter

A simple, powerful CLI tool that converts any PDF document to Markdown using IBM Docling. Perfect for converting documents, reports, meeting minutes, and other PDFs into readable, searchable Markdown format.

## Features

- **Universal PDF Conversion**: Handles any type of PDF document
- **Automatic Segmentation**: Large PDFs (>50 pages) are automatically split into manageable segments
- **High-Quality Output**: Uses IBM Docling for superior text extraction and formatting preservation
- **Simple CLI Interface**: Just `pdf2md filename.pdf` and you're done
- **Smart Output**: Files are created in the same directory as your input PDF
- **Robust Processing**: Handles complex layouts, tables, and multi-column documents

## Installation

### Option 1: Install from PyPI (once published)
```bash
pip install pdf2md
```

### Option 2: Install from Source
```bash
git clone https://github.com/yourusername/pdf2md.git
cd pdf2md
pip install -e .
```

### Option 3: Direct Installation
```bash
pip install docling PyPDF2
# Download pdf2md.py and make it executable
chmod +x pdf2md.py
```

## Requirements

- Python 3.8 or higher
- IBM Docling (`pip install docling`)
- PyPDF2 (`pip install PyPDF2`)

## Usage

### Basic Usage
```bash
# Convert a single PDF
pdf2md document.pdf

# Convert PDF with spaces in filename
pdf2md "my document.pdf"

# Convert with full path
pdf2md /path/to/document.pdf
```

### What Happens

1. **Small PDFs (≤50 pages)**: Converted to a single Markdown file
2. **Large PDFs (>50 pages)**: Automatically segmented into smaller PDFs (kept for reference) and corresponding Markdown files

### Examples

**Converting a small document:**
```bash
$ pdf2md report.pdf
PDF2MD converter initialized
Converting: report.pdf
PDF has 25 pages
Processing as single document
Created: report.md

✅ Conversion completed successfully!
📄 Input: report.pdf
📁 Output directory: /current/directory
📝 Generated 1 markdown file(s):
   - report.md
```

**Converting a large document:**
```bash
$ pdf2md large-manual.pdf
PDF2MD converter initialized
Converting: large-manual.pdf
PDF has 150 pages
Large PDF detected (150 pages). Segmenting...
Processing segment: pages 1-50
Created segment: large-manual_pages_001-050.pdf -> large-manual_pages_001-050.md
Processing segment: pages 51-100
Created segment: large-manual_pages_051-100.pdf -> large-manual_pages_051-100.md
Processing segment: pages 101-150
Created segment: large-manual_pages_101-150.pdf -> large-manual_pages_101-150.md

✅ Conversion completed successfully!
📄 Input: large-manual.pdf
📁 Output directory: /current/directory
📝 Generated 3 markdown file(s):
   - large-manual_pages_001-050.md
   - large-manual_pages_051-100.md
   - large-manual_pages_101-150.md
```

## Output Format

Each generated Markdown file includes:

- **Document header** with source information and metadata
- **Clean, structured content** extracted from the PDF
- **Preserved formatting** including headings, tables, and lists
- **Page range information** for segmented documents

Example output header:
```markdown
# Document Title

## Document Information
- **Source**: original-document.pdf
- **Pages**: 1-50
- **Converted**: PDF2MD
- **Processing**: IBM Docling

---

[Document content follows...]
```

## Advanced Usage

### Verbose Output
```bash
pdf2md document.pdf -v
```

### Help
```bash
pdf2md --help
```

## How It Works

1. **Analysis**: Determines PDF size and complexity
2. **Segmentation**: Large PDFs are split into 50-page segments
3. **Conversion**: IBM Docling processes each PDF into high-quality Markdown
4. **Output**: Clean, readable Markdown files are created in the source directory

## Why PDF2MD?

- **Simple**: One command, clean output
- **Reliable**: Built on IBM Docling's industrial-strength PDF processing
- **Smart**: Automatically handles large files without memory issues
- **Fast**: Efficient processing even for complex documents
- **Flexible**: Works with any PDF - reports, manuals, articles, forms

## Troubleshooting

### Installation Issues
```bash
# If docling installation fails, try:
pip install --upgrade pip
pip install docling

# For Apple Silicon Macs:
pip install --upgrade pip setuptools wheel
pip install docling
```

### Memory Issues with Large PDFs
The tool automatically segments large PDFs, but if you encounter memory issues:
- Ensure you have sufficient disk space for temporary files
- Close other applications to free up RAM

### Permission Issues
```bash
# Make the script executable
chmod +x pdf2md.py

# Or run with Python directly
python pdf2md.py document.pdf
```

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built with [IBM Docling](https://github.com/DS4SD/docling) for superior PDF processing
- Uses [PyPDF2](https://pypdf2.readthedocs.io/) for PDF manipulation