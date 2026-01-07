from dotenv import load_dotenv
load_dotenv()

import PyPDF2
import os
from mistralai import Mistral


api_key = os.getenv("MISTRAL_API_KEY", "")

client = Mistral(api_key=api_key)

def extract_text_from_image_pdf(filepath: str) -> str:
    file_name = os.path.basename(filepath)

    uploaded_pdf = client.files.upload(
        file={
            "file_name": file_name,
            "content": open(filepath, "rb")
        },
        purpose="ocr"
    )

    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        },
        include_image_base64=True
    )

    client.files.delete(file_id=uploaded_pdf.id)

    text = []
    for page in ocr_response.pages:
        if page.markdown:
            text.append(page.markdown)

    return "\n".join(text)



def extract_text_from_text_pdf(filepath: str) -> str:
    text = []

    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text.append(extracted)

    return "\n".join(text)

import os
import PyPDF2
from pathlib import Path

def extract_text_auto(filepath: str) -> str:

    ext = Path(filepath).suffix.lower()

    if ext not in [".pdf"]:
        return extract_text_from_image_pdf(filepath)

    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)

            for i, page in enumerate(reader.pages[:3]):
                extracted = page.extract_text()
                if extracted and extracted.strip():
                    return extract_text_from_text_pdf(filepath)

        return extract_text_from_image_pdf(filepath)

    except Exception as e:
        # If PDF reading fails → fallback to OCR
        print(f"Warning: PDF reading error ({e}) → Using OCR")
        return extract_text_from_image_pdf(filepath)



if __name__ == "__main__":
    path = "data/CV/CV_Brut/CV_Abdellahjpeg.jpeg"
    text = extract_text_auto(path)
    print(text)

