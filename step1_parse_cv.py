"""
STEP 1: PARSE CV (Using Environment Variables)
"""
import os
import json
import PyPDF2
from dotenv import load_dotenv
from mistralai import Mistral

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION (from .env file)
# ============================================================================

API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY)

OUTPUT_FOLDER = "parsed_cvs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================================
# TEXT EXTRACTION
# ============================================================================

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyPDF2"""
    text_pages = []

    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            print(f"   PDF has {len(pdf_reader.pages)} pages")

            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text:
                    text_pages.append(text)
                else:
                    print(f"   ⚠️ Warning: Page {page_num} has no text")

            if not text_pages:
                raise ValueError("No text could be extracted from PDF")

            return "\n".join(text_pages)

    except Exception as e:
        raise Exception(f"PDF extraction failed: {e}")

# ============================================================================
# CV PARSING PROMPT
# ============================================================================

PROMPT = """
Extract information from this CV/Resume and return ONLY a JSON object.

Return these fields:
1. name - candidate's full name
2. email - email address
3. phone - phone number
4. skills - list of all technical skills mentioned
5. education - full education description (degree, institution, years)
6. experience - full work experience description (role, company, years)
7. location - current city/location

If any field is not found, use empty string "" or empty list [].
Format your response as pure JSON, no explanation, no markdown.
"""

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def parse_cv(pdf_path):
    """Parse a CV PDF and save to JSON"""

    filename = os.path.basename(pdf_path)
    print("="*70)
    print(f"PARSING CV: {filename}")
    print("="*70)

    # Extract Text
    print("\n[1/3] Extracting text from PDF...")
    try:
        cv_text = extract_text_from_pdf(pdf_path)
        print(f"   ✓ Extracted {len(cv_text)} characters")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

    # Parse with Mistral AI
    print("\n[2/3] Parsing CV with AI...")
    try:
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": cv_text}
            ],
            temperature=0
        )

        ai_response = response.choices[0].message.content
        start = ai_response.find("{")
        end = ai_response.rfind("}")
        json_text = ai_response[start:end + 1]
        cv_data = json.loads(json_text)
        tokens_used = response.usage.total_tokens

        print(f"   ✓ Parsed successfully (used {tokens_used} tokens)")
        print(f"   ✓ Name: {cv_data.get('name', 'Unknown')}")
        print(f"   ✓ Email: {cv_data.get('email', 'Not found')}")
        print(f"   ✓ Phone: {cv_data.get('phone', 'Not found')}")
        print(f"   ✓ Skills: {len(cv_data.get('skills', []))} found")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

    # Save to JSON
    print("\n[3/3] Saving to JSON...")

    base_name = os.path.splitext(filename)[0]
    output_filename = f"{base_name}_parsed.json"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    cv_data['_metadata'] = {
        'source_file': filename,
        'source_path': pdf_path,
        'extraction_method': 'PyPDF2',
        'tokens_used': tokens_used
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cv_data, f, indent=2, ensure_ascii=False)

    print(f"   ✓ Saved to: {output_path}")

    print("\n" + "="*70)
    print("✅ CV PARSING COMPLETE!")
    print("="*70)
    print(f"Input:  {filename}")
    print(f"Output: {output_filename}")
    print(f"Name:   {cv_data.get('name', 'Unknown')}")
    print(f"Email:  {cv_data.get('email', 'Not found')}")
    print(f"Phone:  {cv_data.get('phone', 'Not found')}")
    print("="*70)

    return output_path

# ============================================================================
# BATCH PROCESSING
# ============================================================================

def parse_multiple_cvs(folder_path):
    """Parse all PDFs in a folder"""

    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"❌ No PDF files found in: {folder_path}")
        return

    print("\n" + "="*70)
    print(f"BATCH PARSING: {len(pdf_files)} CVs")
    print("="*70)

    results = []
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"\n\n[{i}/{len(pdf_files)}] {pdf_file}")
        print("-"*70)

        output_path = parse_cv(pdf_path)

        if output_path:
            results.append((pdf_file, "✅ Success", output_path))
        else:
            results.append((pdf_file, "❌ Failed", None))

    print("\n\n" + "="*70)
    print("BATCH PARSING COMPLETE")
    print("="*70)

    for filename, status, output in results:
        if output:
            print(f"{status} {filename} → {os.path.basename(output)}")
        else:
            print(f"{status} {filename}")

    print("="*70)
    successful = sum(1 for _, status, _ in results if "Success" in status)
    print(f"\nSuccessfully parsed: {successful}/{len(pdf_files)}")
    print(f"JSON files saved in: {OUTPUT_FOLDER}/")
    print("="*70)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    test_pdf = r"C:\Users\VICTUS\Downloads\Dhru_cv_compressed.pdf"

    if os.path.exists(test_pdf):
        parse_cv(test_pdf)
    else:
        print(f"❌ File not found: {test_pdf}")
        print("\nUpdate the 'test_pdf' path to your CV file!")