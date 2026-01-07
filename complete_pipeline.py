"""
COMPLETE CV PIPELINE
PDF → Extract Text → Parse CV → Generate Embeddings → Store in Database
"""
import os
import json
import psycopg2
import requests
from sentence_transformers import SentenceTransformer

# Import your existing modules
from text_extraction import extract_text_auto
from step1_parse_cv import extract_cv_info

# ============================================================================
# CONFIGURATION
# ============================================================================

# Database
DB_CONFIG = {
    'host': "aws-0-eu-west-1.pooler.supabase.com",
    'database': "postgres",
    'user': "postgres.pjsvagvojfqenqpjnuqs",
    'password': "your@kpassword",  # Replace with actual password
    'port': "6543"
}

# Load models (only once)
print("Loading AI models...")
skill_model = SentenceTransformer('models/skill-matcher-miniLM')
education_model = SentenceTransformer('models/education-level-matcher-miniLM')
experience_model = SentenceTransformer('models/experience-matcher-miniLM')
print("✓ Models loaded\n")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_coordinates(place_name):
    """Get coordinates using OpenStreetMap"""
    if not place_name:
        return None

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': place_name,
        'format': 'json',
        'limit': 1
    }
    headers = {
        'User-Agent': 'CVPipeline/1.0'
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        data = response.json()

        if len(data) == 0:
            return None

        return {
            'name': data[0]['display_name'],
            'lat': float(data[0]['lat']),
            'lon': float(data[0]['lon'])
        }
    except Exception as e:
        print(f"   ⚠️ Could not geocode location: {e}")
        return None


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def process_cv_complete(pdf_path, cv_id=None):
    """
    Complete pipeline: PDF → Text → Parse → Embeddings → Database

    Args:
        pdf_path: Path to CV PDF file
        cv_id: Optional CV ID (auto-generated if not provided)

    Returns:
        cv_id: ID of inserted CV
    """

    print("=" * 70)
    print(f"PROCESSING CV: {os.path.basename(pdf_path)}")
    print("=" * 70)

    # Generate CV ID if not provided
    if not cv_id:
        import hashlib
        cv_id = "CV_" + hashlib.md5(pdf_path.encode()).hexdigest()[:8]

    # ========================================================================
    # STEP 1: Extract Text from PDF
    # ========================================================================
    print("\n[1/5] Extracting text from PDF...")
    try:
        cv_text = extract_text_auto(pdf_path)
        print(f"   ✓ Extracted {len(cv_text)} characters")
    except Exception as e:
        print(f"   ❌ Text extraction failed: {e}")
        return None

    # ========================================================================
    # STEP 2: Parse CV with Mistral
    # ========================================================================
    print("\n[2/5] Parsing CV with AI...")
    try:
        cv_data, tokens = extract_cv_info(cv_text)
        print(f"   ✓ Parsed CV (used {tokens} tokens)")
        print(f"   ✓ Name: {cv_data.get('name', 'Unknown')}")
        print(f"   ✓ Email: {cv_data.get('email', 'Not found')}")
        print(f"   ✓ Phone: {cv_data.get('phone', 'Not found')}")
    except Exception as e:
        print(f"   ❌ CV parsing failed: {e}")
        return None

    # ========================================================================
    # STEP 3: Generate Embeddings
    # ========================================================================
    print("\n[3/5] Generating embeddings...")

    # Skills
    skills_text = ", ".join(cv_data.get('skills', [])) if cv_data.get('skills') else "No skills listed"
    skills_embedding = skill_model.encode(skills_text, normalize_embeddings=True)
    print(f"   ✓ Skills embedding: {len(skills_embedding)} dimensions")

    # Education
    education_text = cv_data.get('education', 'Not specified')
    education_embedding = education_model.encode(education_text, normalize_embeddings=True)
    print(f"   ✓ Education embedding: {len(education_embedding)} dimensions")

    # Experience
    experience_text = cv_data.get('experience', 'Not specified')
    experience_embedding = experience_model.encode(experience_text, normalize_embeddings=True)
    print(f"   ✓ Experience embedding: {len(experience_embedding)} dimensions")

    # ========================================================================
    # STEP 4: Get Location Coordinates
    # ========================================================================
    print("\n[4/5] Geocoding location...")
    location_data = get_coordinates(cv_data.get('location', ''))
    if location_data:
        print(f"   ✓ Location: {location_data['name']}")
        print(f"   ✓ Coordinates: ({location_data['lat']}, {location_data['lon']})")
    else:
        print(f"   ⚠️ Could not geocode location: {cv_data.get('location', 'Unknown')}")

    # ========================================================================
    # STEP 5: Insert into Database
    # ========================================================================
    print("\n[5/5] Inserting into database...")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO cvs 
            (cv_id, name, email, phone,
             skills_text, education_text, experience_text, location_text,
             skills_embedding, education_embedding, experience_embedding,
             location_lat, location_lon, location_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (cv_id) 
            DO UPDATE SET
                name = EXCLUDED.name,
                email = EXCLUDED.email,
                phone = EXCLUDED.phone,
                skills_text = EXCLUDED.skills_text,
                education_text = EXCLUDED.education_text,
                experience_text = EXCLUDED.experience_text,
                location_text = EXCLUDED.location_text,
                skills_embedding = EXCLUDED.skills_embedding,
                education_embedding = EXCLUDED.education_embedding,
                experience_embedding = EXCLUDED.experience_embedding,
                location_lat = EXCLUDED.location_lat,
                location_lon = EXCLUDED.location_lon,
                location_name = EXCLUDED.location_name,
                updated_at = CURRENT_TIMESTAMP
        ''', (
            cv_id,
            cv_data.get('name', 'Unknown'),
            cv_data.get('email', ''),
            cv_data.get('phone', ''),
            skills_text,
            education_text,
            experience_text,
            cv_data.get('location', ''),
            json.dumps(skills_embedding.tolist()),
            json.dumps(education_embedding.tolist()),
            json.dumps(experience_embedding.tolist()),
            location_data['lat'] if location_data else None,
            location_data['lon'] if location_data else None,
            location_data['name'] if location_data else None
        ))

        conn.commit()
        print(f"   ✓ Inserted CV: {cv_data.get('name', 'Unknown')}")

        # Get total count
        cursor.execute("SELECT COUNT(*) FROM cvs")
        total = cursor.fetchone()[0]
        print(f"   ✓ Total CVs in database: {total}")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"   ❌ Database insertion failed: {e}")
        if 'conn' in locals():
            conn.rollback()
        return None

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ CV PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"CV ID: {cv_id}")
    print(f"Name: {cv_data.get('name', 'Unknown')}")
    print(f"Email: {cv_data.get('email', 'Not found')}")
    print(f"Phone: {cv_data.get('phone', 'Not found')}")
    print(f"Skills: {skills_text[:50]}...")
    print("=" * 70)

    return cv_id


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_multiple_cvs(pdf_folder):
    """Process all PDFs in a folder"""

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

    print("=" * 70)
    print(f"BATCH PROCESSING: {len(pdf_files)} CVs")
    print("=" * 70)

    results = []
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"\n\n[{i}/{len(pdf_files)}] Processing {pdf_file}...")

        cv_id = process_cv_complete(pdf_path)
        if cv_id:
            results.append((pdf_file, cv_id, "✅ Success"))
        else:
            results.append((pdf_file, None, "❌ Failed"))

    # Summary
    print("\n\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    for filename, cv_id, status in results:
        print(f"{status} {filename}")
    print("=" * 70)

    successful = sum(1 for _, _, status in results if "Success" in status)
    print(f"\nSuccessfully processed: {successful}/{len(pdf_files)}")


# ============================================================================
# MAIN - TEST WITH SINGLE CV
# ============================================================================

if __name__ == "__main__":

    # Test with single CV
    test_pdf = r"C:\Users\VICTUS\Downloads\Dhru_cv_compressed.pdf"  # Update this path!

    if os.path.exists(test_pdf):
        process_cv_complete(test_pdf)
    else:
        print(f"❌ File not found: {test_pdf}")
        print("\nUpdate the 'test_pdf' path to your CV file!")

    # To process multiple CVs from a folder:
    # process_multiple_cvs(r"C:\path\to\cv\folder")