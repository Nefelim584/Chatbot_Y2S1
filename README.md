# PDF Parsing System

A comprehensive Python system for parsing PDF documents with multiple extraction methods including text, images, tables, and metadata.

## Features

- **Text Extraction**: Extract text content with formatting information (font, size, color, position)
- **Table Extraction**: Extract tables from PDFs using pdfplumber
- **Image Extraction**: Extract and save images from PDFs
- **Metadata Extraction**: Extract PDF metadata (title, author, creation date, etc.)
- **Text Search**: Search for specific text within PDFs
- **Page-by-Page Processing**: Extract content from specific pages
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **JSON Export**: Save parsing results in structured JSON format

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dependencies

- **PyMuPDF (fitz)**: Core PDF processing and text extraction
- **pdfplumber**: Advanced table extraction
- **Pillow**: Image processing
- **dataclasses-json**: Data serialization

## Quick Start

### Basic Usage

```python
from main import PDFParser

# Initialize the parser
parser = PDFParser()

# Parse a PDF file
results = parser.parse_pdf("your_document.pdf")

# Access extracted data
print(f"Pages: {results['metadata'].pages}")
print(f"Text blocks: {len(results['text_content'])}")
print(f"Tables found: {len(results['tables'])}")
print(f"Images extracted: {len(results['images'])}")
```

### Command Line Usage

```bash
# Parse a PDF and save results
python main.py document.pdf -o results.json

# Search for specific text
python main.py document.pdf -s "search term"

# Extract text from specific page
python main.py document.pdf -p 1

# Enable verbose logging
python main.py document.pdf -v
```

## API Reference

### PDFParser Class

#### Methods

- `parse_pdf(pdf_path)`: Parse entire PDF and return all extracted data
- `extract_metadata(pdf_path)`: Extract PDF metadata
- `extract_text(pdf_path)`: Extract text with formatting information
- `extract_tables(pdf_path)`: Extract tables from PDF
- `extract_images(pdf_path, output_dir)`: Extract and save images
- `search_text(pdf_path, search_term)`: Search for specific text
- `get_page_text(pdf_path, page_number)`: Get text from specific page
- `save_results(results, output_path)`: Save results to JSON file

#### Data Classes

- `PDFMetadata`: Contains PDF metadata information
- `ExtractedText`: Text content with formatting and position data
- `ExtractedTable`: Table data with page and position information
- `ExtractedImage`: Image data with metadata and file information

## Examples

### Example 1: Basic PDF Parsing

```python
from main import PDFParser

parser = PDFParser()
results = parser.parse_pdf("document.pdf")

# Print summary
metadata = results['metadata']
print(f"Document: {metadata.title}")
print(f"Author: {metadata.author}")
print(f"Pages: {metadata.pages}")

# Save results
parser.save_results(results, "output.json")
```

### Example 2: Text Search

```python
from main import PDFParser

parser = PDFParser()
results = parser.search_text("document.pdf", "important")

for result in results:
    print(f"Found on page {result['page_number']} at {result['bbox']}")
```

### Example 3: Table Extraction

```python
from main import PDFParser

parser = PDFParser()
results = parser.parse_pdf("document.pdf")

for table in results['tables']:
    print(f"Table on page {table.page_number}:")
    for row in table.table_data:
        print(row)
```

### Example 4: Image Extraction

```python
from main import PDFParser

parser = PDFParser()
images = parser.extract_images("document.pdf", "extracted_images/")

for image in images:
    print(f"Image on page {image.page_number}: {image.width}x{image.height}")
```

## Advanced Features

### Custom Text Processing

```python
# Find text with specific formatting
for text_block in results['text_content']:
    for font_info in text_block.font_info:
        if font_info['size'] > 12:  # Large text (potential headers)
            print(f"Header: {font_info['text']}")
```

### Batch Processing

```python
import os
from main import PDFParser

parser = PDFParser()
pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]

for pdf_file in pdf_files:
    print(f"Processing {pdf_file}...")
    results = parser.parse_pdf(pdf_file)
    parser.save_results(results, f"{pdf_file}_results.json")
```

## Error Handling

The system includes comprehensive error handling:

- File not found errors
- Invalid PDF format errors
- Page number out of range errors
- Image extraction errors
- Table extraction errors

All errors are logged and included in the results dictionary under `parsing_errors`.

## Logging

The system provides detailed logging:

- Info level: Basic operation information
- Debug level: Detailed debugging information
- Error level: Error messages and stack traces

Logs are written to both console and `pdf_parser.log` file.

## Output Format

Results are saved in JSON format with the following structure:

```json
{
  "file_path": "document.pdf",
  "metadata": {
    "title": "Document Title",
    "author": "Author Name",
    "pages": 10,
    "file_size": 1234567
  },
  "text_content": [
    {
      "page_number": 1,
      "text": "Extracted text content",
      "font_info": [...],
      "bbox": [0, 0, 612, 792]
    }
  ],
  "tables": [...],
  "images": [...],
  "parsing_errors": []
}
```

## Troubleshooting

### Common Issues

1. **ImportError for PyMuPDF**: Install with `pip install PyMuPDF`
2. **PDF not readable**: Ensure the PDF is not password-protected
3. **Memory issues with large PDFs**: Process page by page instead of loading entire document
4. **Table extraction not working**: Some PDFs have tables as images; consider OCR solutions

### Performance Tips

- For large PDFs, consider processing page by page
- Use specific extraction methods instead of full parsing when possible
- Monitor memory usage with very large documents

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
