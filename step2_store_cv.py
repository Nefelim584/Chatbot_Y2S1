"""
STEP 2: STORE IN DATABASE (Using Environment Variables)
"""
import os
import json
import psycopg2
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION (from .env file)
# ============================================================================

DB_CONFIG = {
    'host': os.getenv("DB_HOST"),
    'database': os.getenv("DB_NAME"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'port': os.getenv("DB_PORT")
}

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

INPUT_FOLDER = "parsed_cvs"

# ============================================================================
# LOAD MODELS
# ============================================================================

print("Loading AI models...")
skill_model = SentenceTransformer('models/skill-matcher-miniLM')
education_model = SentenceTransformer('models/education-level-matcher-miniLM')
experience_model = SentenceTransformer('models/experience-matcher-miniLM')
print("✓ Models loaded\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_coordinates(place_name):
    """Get lat/lon coordinates"""
    if not place_name or place_name.strip() == "":
        return None

    url = "https://nominatim.openstreetmap.org/search"
    params = {'q': place_name, 'format': 'json', 'limit': 1}
    headers = {'User-Agent': 'CVPipeline/1.0'}

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
        print(f"   ⚠️ Geocoding failed: {e}")
        return None

def upload_pdf_to_supabase(pdf_path, cv_id):
    """Upload PDF to Supabase Storage"""
    if not os.path.exists(pdf_path):
        return None

    storage_filename = f"{cv_id}.pdf"
    upload_url = f"{SUPABASE_URL}/storage/v1/object/CVS/{storage_filename}"

    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/pdf"
    }

    try:
        with open(pdf_path, 'rb') as f:
            response = requests.post(upload_url, headers=headers, data=f)

        if response.status_code in [200, 201, 409]:
            pdf_url = f"{SUPABASE_URL}/storage/v1/object/public/CVS/{storage_filename}"
            return pdf_url
        else:
            print(f"   ⚠️ Upload failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"   ⚠️ Upload error: {e}")
        return None

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def store_cv(json_path):
    """Load JSON, generate embeddings, upload PDF, store in database"""

    filename = os.path.basename(json_path)
    print("="*70)
    print(f"STORING CV: {filename}")
    print("="*70)

    # Load JSON
    print("\n[1/5] Loading parsed JSON...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            cv_data = json.load(f)
        print(f"   ✓ Loaded: {cv_data.get('name', 'Unknown')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

    # Generate CV ID
    import hashlib
    cv_id = "CV_" + hashlib.md5(json_path.encode()).hexdigest()[:8]

    # Generate Embeddings
    print("\n[2/5] Generating embeddings...")
    skills_list = cv_data.get('skills', [])
    skills_text = ", ".join(skills_list) if skills_list else "No skills listed"
    skills_embedding = skill_model.encode(skills_text, normalize_embeddings=True)
    print(f"   ✓ Skills: {len(skills_embedding)} dimensions")

    education_text = cv_data.get('education', 'Not specified')
    education_embedding = education_model.encode(education_text, normalize_embeddings=True)
    print(f"   ✓ Education: {len(education_embedding)} dimensions")

    experience_text = cv_data.get('experience', 'Not specified')
    experience_embedding = experience_model.encode(experience_text, normalize_embeddings=True)
    print(f"   ✓ Experience: {len(experience_embedding)} dimensions")

    # Get Location
    print("\n[3/5] Geocoding location...")
    location_text = cv_data.get('location', '')
    location_data = get_coordinates(location_text)

    if location_data:
        print(f"   ✓ {location_data['name']}")
    else:
        print(f"   ⚠️ Could not geocode: {location_text}")

    # Upload PDF
    print("\n[4/5] Uploading PDF...")
    source_path = cv_data['_metadata'].get('source_path', '')
    pdf_url = None

    if source_path and os.path.exists(source_path):
        pdf_url = upload_pdf_to_supabase(source_path, cv_id)
        if pdf_url:
            print(f"   ✓ Uploaded: {cv_id}.pdf")
        else:
            print(f"   ⚠️ Upload failed")
    else:
        print(f"   ⚠️ PDF not found")

    # Insert into Database
    print("\n[5/5] Inserting into database...")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO cvs 
            (cv_id, name, email, phone,
             skills_text, education_text, experience_text, location_text,
             skills_embedding, education_embedding, experience_embedding,
             location_lat, location_lon, location_name, pdf_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                pdf_url = EXCLUDED.pdf_url,
                updated_at = CURRENT_TIMESTAMP
        ''', (
            cv_id,
            cv_data.get('name', 'Unknown'),
            cv_data.get('email', ''),
            cv_data.get('phone', ''),
            skills_text,
            education_text,
            experience_text,
            location_text,
            json.dumps(skills_embedding.tolist()),
            json.dumps(education_embedding.tolist()),
            json.dumps(experience_embedding.tolist()),
            location_data['lat'] if location_data else None,
            location_data['lon'] if location_data else None,
            location_data['name'] if location_data else None,
            pdf_url
        ))

        conn.commit()
        print(f"   ✓ Inserted: {cv_data.get('name', 'Unknown')}")

        cursor.execute("SELECT COUNT(*) FROM cvs")
        total = cursor.fetchone()[0]
        print(f"   ✓ Total CVs: {total}")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"   ❌ Database error: {e}")
        if 'conn' in locals():
            conn.rollback()
        return None

    # Summary
    print("\n" + "="*70)
    print("✅ CV STORED!")
    print("="*70)
    print(f"CV ID: {cv_id}")
    print(f"Name:  {cv_data.get('name', 'Unknown')}")
    if pdf_url:
        print(f"PDF:   {pdf_url}")
    print("="*70)

    return cv_id

def store_multiple_cvs(json_folder=INPUT_FOLDER):
    """Store all parsed JSONs"""

    if not os.path.exists(json_folder):
        print(f"❌ Folder not found: {json_folder}")
        return

    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    if not json_files:
        print(f"❌ No JSON files found")
        return

    print(f"\nBatch storing {len(json_files)} CVs...")

    results = []
    for i, json_file in enumerate(json_files, 1):
        json_path = os.path.join(json_folder, json_file)
        print(f"\n[{i}/{len(json_files)}] {json_file}")
        print("-"*70)

        cv_id = store_cv(json_path)
        results.append((json_file, "✅" if cv_id else "❌", cv_id))

    print("\n" + "="*70)
    print("BATCH COMPLETE")
    print("="*70)
    successful = sum(1 for _, status, _ in results if status == "✅")
    print(f"Success: {successful}/{len(json_files)}")
    print("="*70)

if __name__ == "__main__":

    if os.path.exists(INPUT_FOLDER):
        json_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.json')]
        if json_files:
            store_cv(os.path.join(INPUT_FOLDER, json_files[0]))
        else:
            print("❌ No JSON files found")
    else:
        print("❌ Folder not found")