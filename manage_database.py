"""
CV DATABASE MANAGEMENT (Using Environment Variables)
"""
import os
import json
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from step1_parse_cv import parse_cv
from step2_store_cv import store_cv

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


PARSED_CVS_FOLDER = "parsed_cvs"

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def get_connection():
    """Get database connection"""
    return psycopg2.connect(**DB_CONFIG)

def get_dict_connection():
    """Get connection that returns dicts"""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

def get_cv_count():
    """Get total CVs"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cvs")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count
    except:
        return 0

def test_connection():
    """Test database connection"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return True, version
    except Exception as e:
        return False, str(e)

# ============================================================================
# VIEW CVS
# ============================================================================

def view_all_cvs():
    """Display all CVs"""

    print("\n" + "="*70)
    print("ALL CVs IN DATABASE")
    print("="*70)

    try:
        conn = get_dict_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT cv_id, name, email, phone, location_text, pdf_url, created_at
            FROM cvs
            ORDER BY created_at DESC
        """)

        cvs = cursor.fetchall()

        if not cvs:
            print("\n❌ No CVs in database!")
        else:
            print(f"\nTotal CVs: {len(cvs)}\n")

            for i, cv in enumerate(cvs, 1):
                print(f"{i}. {cv['name']}")
                print(f"   ID: {cv['cv_id']}")
                print(f"   Email: {cv['email'] or 'N/A'}")
                print(f"   Phone: {cv['phone'] or 'N/A'}")
                if cv['pdf_url']:
                    print(f"   PDF: {cv['pdf_url']}")
                print()

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"\n❌ Error: {e}")

    print("="*70)

def search_cv():
    """Search CV by name"""

    print("\n" + "="*70)
    print("SEARCH CV")
    print("="*70)

    search_term = input("\nEnter name: ").strip()

    if not search_term:
        print("❌ Enter a name!")
        return

    try:
        conn = get_dict_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT cv_id, name, email, phone, skills_text, pdf_url
            FROM cvs
            WHERE name ILIKE %s
        """, (f"%{search_term}%",))

        results = cursor.fetchall()

        if not results:
            print(f"\n❌ No CVs found matching '{search_term}'")
        else:
            print(f"\nFound {len(results)} CV(s):\n")

            for i, cv in enumerate(results, 1):
                print(f"{i}. {cv['name']} ({cv['cv_id']})")
                print(f"   Email: {cv['email']}")
                print(f"   Phone: {cv['phone']}")
                if cv['pdf_url']:
                    print(f"   PDF: {cv['pdf_url']}")
                print()

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"\n❌ Error: {e}")

# ============================================================================
# ADD CVS
# ============================================================================

def add_cvs_from_folder():
    """Add CVs from folder"""

    print("\n" + "="*70)
    print("ADD CVs FROM FOLDER")
    print("="*70)

    folder_path = input("\nEnter folder path: ").strip().strip('"').strip("'")

    if not os.path.exists(folder_path):
        print(f"\n❌ Folder not found: {folder_path}")
        return

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"\n❌ No PDFs found in: {folder_path}")
        return

    print(f"\nFound {len(pdf_files)} PDF(s)")
    confirm = input(f"\nProcess all {len(pdf_files)} CVs? (yes/no): ").strip().lower()

    if confirm not in ['yes', 'y']:
        print("❌ Cancelled")
        return



    print("\n" + "="*70)
    print("PROCESSING")
    print("="*70)

    success = 0
    failed = 0

    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"\n[{i}/{len(pdf_files)}] {pdf_file}")
        print("-"*70)

        try:
            json_path = parse_cv(pdf_path)
            if json_path:
                cv_id = store_cv(json_path)
                if cv_id:
                    success += 1
                else:
                    failed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed += 1

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"✅ Success: {success}")
    print(f"❌ Failed: {failed}")
    print(f"Total CVs: {get_cv_count()}")
    print("="*70)

def add_single_cv():
    """Add single CV"""

    print("\n" + "="*70)
    print("ADD SINGLE CV")
    print("="*70)

    pdf_path = input("\nEnter PDF path: ").strip().strip('"').strip("'")

    if not os.path.exists(pdf_path):
        print(f"\n❌ File not found: {pdf_path}")
        return

    from step1_parse_cv import parse_cv
    from step2_store_cv import store_cv

    print("\n" + "-"*70)

    try:
        json_path = parse_cv(pdf_path)
        if json_path:
            cv_id = store_cv(json_path)
            if cv_id:
                print(f"\n✅ CV added!")
                print(f"Total CVs: {get_cv_count()}")
            else:
                print("\n❌ Failed to store!")
        else:
            print("\n❌ Parsing failed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

# ============================================================================
# REMOVE CVS
# ============================================================================

def remove_cv_by_id():
    """Remove CV by ID"""

    print("\n" + "="*70)
    print("REMOVE CV BY ID")
    print("="*70)

    cv_id = input("\nEnter CV ID: ").strip()

    if not cv_id:
        print("❌ Enter CV ID!")
        return

    try:
        conn = get_dict_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM cvs WHERE cv_id = %s", (cv_id,))
        result = cursor.fetchone()

        if not result:
            print(f"\n❌ CV ID not found: {cv_id}")
        else:
            print(f"\nFound: {result['name']}")
            confirm = input("Delete? (yes/no): ").strip().lower()

            if confirm in ['yes', 'y']:
                cursor.execute("DELETE FROM cvs WHERE cv_id = %s", (cv_id,))
                conn.commit()
                print(f"\n✅ Deleted: {result['name']}")
                print(f"Remaining: {get_cv_count()}")
            else:
                print("❌ Cancelled")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"\n❌ Error: {e}")

def clear_all_cvs():
    """Remove ALL CVs"""

    print("\n" + "="*70)
    print("⚠️  CLEAR ALL CVs")
    print("="*70)

    count = get_cv_count()

    if count == 0:
        print("\n❌ Database already empty!")
        return

    print(f"\n⚠️  WARNING: Delete ALL {count} CVs?")
    confirm = input("Type 'DELETE ALL': ").strip()

    if confirm == 'DELETE ALL':
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cvs")
            conn.commit()
            cursor.close()
            conn.close()
            print(f"\n✅ Deleted all {count} CVs")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    else:
        print("❌ Cancelled")

# ============================================================================
# MENU
# ============================================================================

def show_menu():
    """Display menu"""

    print("\n" + "="*70)
    print("CV DATABASE MANAGEMENT")
    print("="*70)

    print(f"\nCurrent CVs: {get_cv_count()}")

    print("\n📊 VIEW:")
    print("  1. View all CVs")
    print("  2. Search CV")

    print("\n➕ ADD:")
    print("  3. Add from folder (batch)")
    print("  4. Add single CV")

    print("\n➖ REMOVE:")
    print("  5. Remove by ID")
    print("  6. Clear all")

    print("\n🔧 OTHER:")
    print("  7. Test connection")
    print("  0. Exit")
    print("-"*70)

def main():
    """Main program"""

    print("\n" + "="*70)
    print("CV DATABASE MANAGEMENT")
    print("="*70)

    connected, info = test_connection()

    if not connected:
        print("\n❌ Database connection failed!")
        print(f"Error: {info}")
        print("\nCheck .env file settings!")
        return

    print("\n✅ Database connected!")

    while True:
        show_menu()

        choice = input("\nChoice (0-7): ").strip()

        if choice == '1':
            view_all_cvs()
        elif choice == '2':
            search_cv()
        elif choice == '3':
            add_cvs_from_folder()
        elif choice == '4':
            add_single_cv()
        elif choice == '5':
            remove_cv_by_id()
        elif choice == '6':
            clear_all_cvs()
        elif choice == '7':
            connected, info = test_connection()
            if connected:
                print(f"\n✅ Connected! {info[:50]}...")
            else:
                print(f"\n❌ Failed: {info}")
        elif choice == '0':
            print("\nGoodbye! 👋")
            break
        else:
            print("\n❌ Invalid choice!")

        input("\nPress Enter...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")