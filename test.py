"""
View education text and embedding from database
"""
import psycopg2
import json
from dotenv import load_dotenv
import os

load_dotenv()

# Connect
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    port=os.getenv("DB_PORT")
)

cursor = conn.cursor()

# Get one CV
cursor.execute("""
    SELECT cv_id, name, education_text, education_embedding 
    FROM cvs 
    LIMIT 1
""")

cv_id, name, education_text, education_embedding = cursor.fetchone()

# Parse embedding
embedding = json.loads(education_embedding)

print(f"CV ID: {cv_id}")
print(f"Name: {name}")
print(f"\nEducation Text:")
print(f"  {education_text}")
print(f"\nEducation Embedding:")
print(f"  Type: {type(embedding)}")
print(f"  Shape: {len(embedding)} dimensions")
print(f"  First 5 values: {embedding[:5]}")

cursor.close()
conn.close()