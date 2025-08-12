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
from typing import List
import argparse

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("Error: docling is not installed. Please install with: pip install docling")
    sys.exit(1)

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("Error: PyPDF2 is not installed. Please install with: pip install PyPDF2")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class PDF2MD:
    """PDF to Markdown converter with automatic segmentation for large files."""
    
    def __init__(self):
        """Initialize the converter with Docling."""
        if not DOCLING_AVAILABLE:
            raise Exception("Docling is required but not available")
        
        self.converter = DocumentConverter()
        logger.info("PDF2MD converter initialized")
    
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
        
        # Get page count and decide on segmentation
        page_count = self._get_page_count(input_path)
        logger.info(f"PDF has {page_count} pages")
        
        output_dir = input_path.parent
        output_files = []
        
        # If PDF is small enough, process as single file
        if page_count <= 50:
            logger.info("Processing as single document")
            output_file = output_dir / f"{input_path.stem}.md"
            self._convert_single_pdf(input_path, output_file)
            output_files.append(output_file)
        else:
            logger.info(f"Large PDF detected ({page_count} pages). Segmenting...")
            output_files = self._convert_segmented_pdf(input_path, output_dir)
        
        logger.info(f"Conversion complete. Generated {len(output_files)} markdown file(s)")
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
    
    def _convert_single_pdf(self, input_path: Path, output_path: Path):
        """Convert a single PDF to markdown."""
        try:
            # Convert with Docling
            result = self.converter.convert(input_path)
            
            # Extract markdown content
            if hasattr(result, 'document') and hasattr(result.document, 'export_to_markdown'):
                markdown_content = result.document.export_to_markdown()
            else:
                raise Exception("Could not extract markdown from conversion result")
            
            # Add document header
            header = f"""# {input_path.stem}

## Document Information
- **Source**: {input_path.name}
- **Pages**: 1-{self._get_page_count(input_path)}
- **Converted**: PDF2MD
- **Processing**: IBM Docling

---

"""
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(header + markdown_content)
            
            logger.info(f"Created: {output_path.name}")
            
        except Exception as e:
            logger.error(f"Error converting PDF: {e}")
            raise
    
    def _convert_segmented_pdf(self, input_path: Path, output_dir: Path) -> List[Path]:
        """Convert a large PDF by segmenting it into smaller files."""
        page_count = self._get_page_count(input_path)
        segment_size = 50  # pages per segment
        output_files = []
        
        # Create segments
        for i in range(0, page_count, segment_size):
            start_page = i + 1
            end_page = min(i + segment_size, page_count)
            
            # Create segment filename
            segment_name = f"{input_path.stem}_pages_{start_page:03d}-{end_page:03d}"
            pdf_segment_path = output_dir / f"{segment_name}.pdf"
            md_segment_path = output_dir / f"{segment_name}.md"
            
            logger.info(f"Processing segment: pages {start_page}-{end_page}")
            
            try:
                # Create PDF segment
                self._create_pdf_segment(input_path, pdf_segment_path, start_page, end_page)
                
                # Convert segment to markdown
                self._convert_single_pdf(pdf_segment_path, md_segment_path)
                output_files.append(md_segment_path)
                
                logger.info(f"Created segment: {pdf_segment_path.name} -> {md_segment_path.name}")
                
            except Exception as e:
                logger.error(f"Error processing segment {start_page}-{end_page}: {e}")
                # Continue with next segment
                continue
        
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

Notes:
  - Output files are created in the same directory as the input PDF
  - Large PDFs (>50 pages) are automatically split into segments
  - Both PDF segments and markdown files are kept for reference
        """
    )
    
    parser.add_argument(
        'pdf_file',
        help='Path to the PDF file to convert'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert path argument
    input_path = Path(args.pdf_file).resolve()
    
    try:
        # Initialize converter and process
        converter = PDF2MD()
        output_files = converter.convert_pdf(input_path)
        
        # Print results
        print(f"\n✅ Conversion completed successfully!")
        print(f"📄 Input: {input_path.name}")
        print(f"📁 Output directory: {input_path.parent}")
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