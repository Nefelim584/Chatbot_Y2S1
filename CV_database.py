"""
Simple script: Process 1 CV → Insert into Supabase
"""
import psycopg2

DB_CONFIG = {
    'host': "db.pjsvagvojfqenqpjnuqs.supabase.co",
    'database': "postgres",
    'user': "postgres",
    'password': "7030DAKSHA@k",  # Replace with your actual password
    'port': "5432"
}



print("=" * 70)
print("UPDATING DATABASE SCHEMA")
print("=" * 70)

try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Step 1: Drop old table
    print("\n1. Dropping old 'cvs' table (if exists)...")
    cursor.execute("DROP TABLE IF EXISTS cvs CASCADE")
    conn.commit()
    print("   ✓ Old table dropped")

    # Step 2: Create new table with metadata
    print("\n2. Creating new 'cvs' table with metadata...")
    cursor.execute('''
        CREATE TABLE cvs (
            -- Primary key
            cv_id VARCHAR(50) PRIMARY KEY,

            -- Contact information (for recruiters)
            name VARCHAR(200) NOT NULL,
            email VARCHAR(200),
            phone VARCHAR(50),

            -- Original text data
            skills_text TEXT,
            education_text TEXT,
            experience_text TEXT,
            location_text VARCHAR(200),

            -- Embeddings (stored as JSON)
            skills_embedding TEXT,
            education_embedding TEXT,
            experience_embedding TEXT,

            -- Location coordinates
            location_lat FLOAT,
            location_lon FLOAT,
            location_name TEXT,

            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    print("   ✓ New table created")

    # Step 3: Verify table structure
    print("\n3. Verifying table structure...")
    cursor.execute("""
        SELECT column_name, data_type, character_maximum_length
        FROM information_schema.columns 
        WHERE table_name = 'cvs'
        ORDER BY ordinal_position
    """)

    columns = cursor.fetchall()

    print("\n" + "=" * 70)
    print("NEW TABLE STRUCTURE:")
    print("=" * 70)
    for col_name, col_type, max_len in columns:
        if max_len:
            print(f"  {col_name:<30} {col_type}({max_len})")
        else:
            print(f"  {col_name:<30} {col_type}")
    print("=" * 70)

    cursor.close()
    conn.close()

    print("\n✅ Database updated successfully!")
    print("\nNew columns added:")
    print("  - email (for recruiter contact)")
    print("  - phone (for recruiter contact)")
    print("  - location_text (readable location)")
    print("  - location_lat, location_lon, location_name (coordinates)")

except Exception as e:
    print(f"\n❌ Error: {e}")
    if 'conn' in locals():
        conn.rollback()

print("\n" + "=" * 70)