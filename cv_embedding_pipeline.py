"""
UPDATE DATABASE SCHEMA - Add pdf_url column
Run this ONCE to add PDF URL storage capability
"""
import psycopg2

# Database connection
DB_CONFIG = {
    'host': "aws-0-eu-west-1.pooler.supabase.com",
    'database': "postgres",
    'user': "postgres.pjsvagvojfqenqpjnuqs",
    'password': "your@kpassword",  # Replace with your actual password
    'port': "6543"
}

print("="*70)
print("ADDING pdf_url COLUMN TO DATABASE")
print("="*70)

try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Check if column already exists
    cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'cvs' AND column_name = 'pdf_url'
    """)

    exists = cursor.fetchone()

    if exists:
        print("\n✓ Column 'pdf_url' already exists!")
    else:
        print("\n[1/2] Adding pdf_url column...")
        cursor.execute("""
            ALTER TABLE cvs 
            ADD COLUMN pdf_url TEXT
        """)
        conn.commit()
        print("   ✓ Column added!")

    # Verify
    print("\n[2/2] Verifying table structure...")
    cursor.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'cvs'
        ORDER BY ordinal_position
    """)

    columns = cursor.fetchall()

    print("\n" + "="*70)
    print("CURRENT TABLE STRUCTURE:")
    print("="*70)
    for col_name, col_type in columns:
        marker = " ← NEW!" if col_name == 'pdf_url' else ""
        print(f"  {col_name:<30} {col_type}{marker}")
    print("="*70)

    cursor.close()
    conn.close()

    print("\n✅ Database updated successfully!")
    print("\nYou can now:")
    print("1. Store PDF URLs when adding CVs")
    print("2. Retrieve PDFs by clicking the URL")
    print("3. Show PDF links in search results")

except Exception as e:
    print(f"\n❌ Error: {e}")
    if 'conn' in locals():
        conn.rollback()

print("\n" + "="*70)