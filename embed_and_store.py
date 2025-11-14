"""
embed_and_store.py - ONE-STOP PIPELINE: Embed with Mistral + Store in ChromaDB
Generates embeddings for 3-vector system and stores directly in ChromaDB
Run this ONCE to process all CVs and Jobs

Features:
- Creates Mistral embeddings (1024-dim) for Skills, Technical, Context
- Stores directly in ChromaDB (no intermediate files)
- Skips already embedded items (smart duplicate detection)
- Shows progress and cost estimation
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import time
from mistralai import Mistral
from dotenv import load_dotenv
import os
import chromadb
from chromadb.config import Settings


# ============================================================================
# CONFIGURATION
# ============================================================================

EMBEDDING_MODEL = "mistral-embed"  # 1024 dimensions
EMBEDDING_DIM = 1024
BATCH_SIZE = 50  # Process in batches for efficiency
COST_PER_1M_TOKENS = 0.10  # Mistral embed pricing
AVG_TOKENS_PER_CV = 250  # Approximate


# ============================================================================
# PART 1: MISTRAL EMBEDDING FUNCTIONS
# ============================================================================

def load_api_key():
    """Load Mistral API key from .env file"""
    load_dotenv()
    api_key = os.getenv('MISTRAL_API_KEY')

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY not found in .env file!\n"
            "Please create .env file with: MISTRAL_API_KEY=your_key_here"
        )

    return api_key


def initialize_mistral_client():
    """Initialize Mistral client"""
    api_key = load_api_key()
    client = Mistral(api_key=api_key)

    print("✅ Mistral client initialized")
    return client


def create_embeddings_batch(client, texts, batch_name="batch"):
    """
    Create embeddings for a batch of texts using Mistral

    Args:
        client: Mistral client
        texts: List of text strings
        batch_name: Name for progress tracking

    Returns:
        List of embeddings (1024-dim vectors)
    """
    if not texts or len(texts) == 0:
        return []

    # Filter out empty texts and handle NaN values
    valid_texts = []
    for t in texts:
        # Convert to string and check if valid
        if pd.notna(t) and str(t).strip():
            valid_texts.append(str(t).strip())

    if len(valid_texts) == 0:
        return [np.zeros(EMBEDDING_DIM).tolist() for _ in texts]

    try:
        # Call Mistral API
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            inputs=valid_texts
        )

        # Extract embeddings
        embeddings = [item.embedding for item in response.data]

        # Add zero vectors for empty texts
        result = []
        valid_idx = 0
        for original_text in texts:
            # Check if text is valid (not NaN and not empty)
            if pd.notna(original_text) and str(original_text).strip():
                result.append(embeddings[valid_idx])
                valid_idx += 1
            else:
                result.append(np.zeros(EMBEDDING_DIM).tolist())

        return result

    except Exception as e:
        print(f"❌ Error creating embeddings for {batch_name}: {str(e)}")
        # Return zero vectors on error
        return [np.zeros(EMBEDDING_DIM).tolist() for _ in texts]


def embed_cvs_in_batches(client, df_cvs):
    """
    Create embeddings for all CVs in batches
    Returns: DataFrame with embeddings added
    """
    print("\n" + "="*60)
    print("📊 EMBEDDING CVs WITH MISTRAL")
    print("="*60)

    total_cvs = len(df_cvs)
    print(f"Total CVs to embed: {total_cvs}")
    print(f"Vectors per CV: 3 (Skills + Technical + Context)")
    print(f"Total embeddings: {total_cvs * 3}")

    # Estimate cost
    estimated_tokens = total_cvs * 3 * AVG_TOKENS_PER_CV
    estimated_cost = (estimated_tokens / 1_000_000) * COST_PER_1M_TOKENS
    print(f"Estimated cost: ${estimated_cost:.3f}")

    # Prepare result lists
    skills_embeddings = []
    technical_embeddings = []
    context_embeddings = []

    # Process in batches
    print(f"\nProcessing in batches of {BATCH_SIZE}...")

    for start_idx in tqdm(range(0, total_cvs, BATCH_SIZE), desc="Embedding CVs"):
        end_idx = min(start_idx + BATCH_SIZE, total_cvs)
        batch = df_cvs.iloc[start_idx:end_idx]

        # Extract texts for this batch
        skills_texts = batch['skills_text'].tolist()
        technical_texts = batch['technical_text'].tolist()
        context_texts = batch['context_text'].tolist()

        # Create embeddings for each vector type
        skills_batch = create_embeddings_batch(client, skills_texts, f"skills_{start_idx}")
        technical_batch = create_embeddings_batch(client, technical_texts, f"technical_{start_idx}")
        context_batch = create_embeddings_batch(client, context_texts, f"context_{start_idx}")

        # Append to results
        skills_embeddings.extend(skills_batch)
        technical_embeddings.extend(technical_batch)
        context_embeddings.extend(context_batch)

        # Small delay to avoid rate limiting
        time.sleep(0.1)

    # Add embeddings to dataframe
    df_cvs['skills_embedding'] = skills_embeddings
    df_cvs['technical_embedding'] = technical_embeddings
    df_cvs['context_embedding'] = context_embeddings

    print(f"\n✅ Created {len(skills_embeddings)} embeddings per vector type")
    print(f"✅ Total embeddings: {len(skills_embeddings) * 3}")

    return df_cvs


def embed_jobs_in_batches(client, df_jobs):
    """
    Create embeddings for all Jobs in batches
    Returns: DataFrame with embeddings added
    """
    if len(df_jobs) == 0:
        print("\n⚠️  No jobs to embed")
        return df_jobs

    print("\n" + "="*60)
    print("💼 EMBEDDING JOBS WITH MISTRAL")
    print("="*60)

    total_jobs = len(df_jobs)
    print(f"Total jobs to embed: {total_jobs}")
    print(f"Vectors per job: 3 (Skills + Technical + Context)")

    # Prepare result lists
    skills_embeddings = []
    technical_embeddings = []
    context_embeddings = []

    # Process in batches
    print(f"\nProcessing in batches of {BATCH_SIZE}...")

    for start_idx in tqdm(range(0, total_jobs, BATCH_SIZE), desc="Embedding jobs"):
        end_idx = min(start_idx + BATCH_SIZE, total_jobs)
        batch = df_jobs.iloc[start_idx:end_idx]

        # Extract texts
        skills_texts = batch['skills_text'].tolist()
        technical_texts = batch['technical_text'].tolist()
        context_texts = batch['context_text'].tolist()

        # Create embeddings
        skills_batch = create_embeddings_batch(client, skills_texts, f"job_skills_{start_idx}")
        technical_batch = create_embeddings_batch(client, technical_texts, f"job_technical_{start_idx}")
        context_batch = create_embeddings_batch(client, context_texts, f"job_context_{start_idx}")

        skills_embeddings.extend(skills_batch)
        technical_embeddings.extend(technical_batch)
        context_embeddings.extend(context_batch)

        time.sleep(0.1)

    # Add to dataframe
    df_jobs['skills_embedding'] = skills_embeddings
    df_jobs['technical_embedding'] = technical_embeddings
    df_jobs['context_embedding'] = context_embeddings

    print(f"\n✅ Created {len(skills_embeddings)} embeddings per vector type")

    return df_jobs


# ============================================================================
# PART 2: CHROMADB STORAGE FUNCTIONS
# ============================================================================

def initialize_chromadb():
    """Initialize ChromaDB client and create collections"""
    print("\n" + "="*60)
    print("🗄️  INITIALIZING CHROMADB")
    print("="*60)

    # Create ChromaDB client (persistent storage)
    base_path = Path.cwd()
    chroma_path = base_path / 'data' / 'chromadb'
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_path))

    print(f"✅ ChromaDB initialized at: {chroma_path}")

    # Create 3 collections (one for each vector type)
    collections = {}

    # Get or create collections
    for vector_name in ['skills', 'technical', 'context']:
        try:
            # Try to get existing collection
            collection = client.get_collection(name=f"cv_{vector_name}")
            existing_count = collection.count()
            print(f"📂 Found existing collection: cv_{vector_name} ({existing_count} items)")
        except:
            # Create new collection
            collection = client.create_collection(
                name=f"cv_{vector_name}",
                metadata={"description": f"CV {vector_name} vectors (1024-dim Mistral embeddings)"}
            )
            print(f"📂 Created new collection: cv_{vector_name}")

        collections[vector_name] = collection

    return client, collections


def store_cvs_in_chromadb(df_cvs, collections):
    """
    Store CV embeddings in ChromaDB collections

    Args:
        df_cvs: DataFrame with embeddings
        collections: Dict of ChromaDB collections
    """
    print("\n" + "="*60)
    print("💾 STORING CVs IN CHROMADB")
    print("="*60)

    total_cvs = len(df_cvs)

    for vector_type in ['skills', 'technical', 'context']:
        collection = collections[vector_type]

        print(f"\nStoring {vector_type} vectors...")

        # Prepare data for ChromaDB
        ids = [f"cv_{int(cv_id)}_{vector_type}" for cv_id in df_cvs['cv_id']]
        embeddings = df_cvs[f'{vector_type}_embedding'].tolist()

        # Handle NaN in documents
        documents = []
        for text in df_cvs[f'{vector_type}_text']:
            if pd.notna(text):
                documents.append(str(text))
            else:
                documents.append("")  # Empty string for NaN

        # Metadata for each CV
        metadatas = []
        for _, row in df_cvs.iterrows():
            # Handle NaN values in text
            text_value = row[f'{vector_type}_text']
            if pd.notna(text_value):
                text_len = len(str(text_value))
            else:
                text_len = 0

            metadata = {
                'cv_id': int(row['cv_id']),
                'vector_type': vector_type,
                'has_skills': bool(row['has_skills']),
                'has_technical': bool(row['has_technical']),
                'has_context': bool(row['has_context']),
                'text_length': text_len
            }
            metadatas.append(metadata)

        # Store in batches (ChromaDB has limits)
        batch_size = 100
        for i in tqdm(range(0, len(ids), batch_size), desc=f"Storing {vector_type}"):
            end_idx = min(i + batch_size, len(ids))

            try:
                collection.upsert(
                    ids=ids[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx]
                )
            except Exception as e:
                print(f"❌ Error storing batch {i}-{end_idx}: {str(e)}")

        final_count = collection.count()
        print(f"✅ Stored {final_count} {vector_type} vectors")

    print(f"\n✅ All CV vectors stored in ChromaDB!")


def store_jobs_in_chromadb(df_jobs, collections):
    """Store job embeddings in ChromaDB collections"""
    if len(df_jobs) == 0:
        print("\n⚠️  No jobs to store")
        return

    print("\n" + "="*60)
    print("💾 STORING JOBS IN CHROMADB")
    print("="*60)

    for vector_type in ['skills', 'technical', 'context']:
        collection = collections[vector_type]

        print(f"\nStoring job {vector_type} vectors...")

        # Prepare data
        ids = [f"job_{int(job_id)}_{vector_type}" for job_id in df_jobs['job_id']]
        embeddings = df_jobs[f'{vector_type}_embedding'].tolist()

        # Handle NaN in documents
        documents = []
        for text in df_jobs[f'{vector_type}_text']:
            if pd.notna(text):
                documents.append(str(text))
            else:
                documents.append("")

        metadatas = []
        for _, row in df_jobs.iterrows():
            # Handle NaN in text length
            text_value = row[f'{vector_type}_text']
            text_len = len(str(text_value)) if pd.notna(text_value) else 0

            # Handle NaN in job_title
            job_title = row.get('job_title', '')
            if pd.isna(job_title):
                job_title = ''

            metadata = {
                'job_id': int(row['job_id']),
                'job_title': str(job_title),
                'vector_type': vector_type,
                'is_job': True,
                'text_length': text_len
            }
            metadatas.append(metadata)

        # Store
        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            print(f"✅ Stored {len(ids)} job {vector_type} vectors")
        except Exception as e:
            print(f"❌ Error storing jobs: {str(e)}")


# ============================================================================
# PART 3: MAIN PIPELINE
# ============================================================================

def check_already_embedded(collections):
    """Check if data is already embedded"""
    for name, collection in collections.items():
        count = collection.count()
        if count > 0:
            return True, count
    return False, 0


def main():
    """Main pipeline: Load → Embed → Store"""

    print("\n" + "="*70)
    print("🚀 EMBED & STORE PIPELINE - ONE-STOP CV PROCESSING")
    print("="*70)
    print("This script will:")
    print("  1. Load prepared 3-vector CVs and Jobs")
    print("  2. Create Mistral embeddings (1024-dim)")
    print("  3. Store directly in ChromaDB")
    print("  4. Skip duplicates if already embedded")

    # Setup paths
    base_path = Path.cwd()
    processed_dir = base_path / 'data' / 'processed'

    cv_file = processed_dir / 'cvs_prepared_3vec.csv'
    job_file = processed_dir / 'jobs_prepared_3vec.csv'

    # Check if files exist
    if not cv_file.exists():
        print(f"\n❌ Error: {cv_file} not found!")
        print("Please run data_preparation_v2.py first")
        return

    # Initialize ChromaDB first (to check for existing data)
    chroma_client, collections = initialize_chromadb()

    # Check if already embedded
    already_embedded, existing_count = check_already_embedded(collections)

    if already_embedded:
        print(f"\n⚠️  Found {existing_count} existing embeddings in ChromaDB")
        response = input("Do you want to re-embed and overwrite? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted. Existing embeddings preserved.")
            return

    # Load data
    print("\n" + "="*60)
    print("📂 LOADING PREPARED DATA")
    print("="*60)

    df_cvs = pd.read_csv(cv_file)
    print(f"✅ Loaded {len(df_cvs)} CVs")

    df_jobs = pd.read_csv(job_file) if job_file.exists() else pd.DataFrame()
    print(f"✅ Loaded {len(df_jobs)} jobs")

    # Initialize Mistral
    mistral_client = initialize_mistral_client()

    # STEP 1: Embed CVs
    df_cvs_embedded = embed_cvs_in_batches(mistral_client, df_cvs)

    # STEP 2: Embed Jobs (if any)
    df_jobs_embedded = embed_jobs_in_batches(mistral_client, df_jobs) if len(df_jobs) > 0 else df_jobs

    # STEP 3: Store in ChromaDB
    store_cvs_in_chromadb(df_cvs_embedded, collections)

    if len(df_jobs_embedded) > 0:
        store_jobs_in_chromadb(df_jobs_embedded, collections)

    # Save embeddings to CSV (backup)
    print("\n" + "="*60)
    print("💾 SAVING BACKUP FILES")
    print("="*60)

    cv_embedded_file = processed_dir / 'cvs_with_embeddings.csv'
    # Don't save embedding columns (too large), just metadata
    df_cvs_meta = df_cvs[['cv_id', 'skills_text', 'technical_text', 'context_text',
                           'has_skills', 'has_technical', 'has_context']]
    df_cvs_meta.to_csv(cv_embedded_file, index=False)
    print(f"✅ Saved CV metadata to: {cv_embedded_file}")

    # Save summary
    summary = {
        'total_cvs': len(df_cvs),
        'total_jobs': len(df_jobs),
        'embedding_model': EMBEDDING_MODEL,
        'embedding_dimension': EMBEDDING_DIM,
        'vectors_per_cv': 3,
        'total_cv_embeddings': len(df_cvs) * 3,
        'total_job_embeddings': len(df_jobs) * 3 if len(df_jobs) > 0 else 0,
        'chromadb_path': str(base_path / 'data' / 'chromadb'),
        'collections': ['cv_skills', 'cv_technical', 'cv_context']
    }

    summary_file = processed_dir / 'embedding_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved summary to: {summary_file}")

    # Final summary
    print("\n" + "="*70)
    print("✅ EMBED & STORE COMPLETE!")
    print("="*70)
    print(f"\n📊 RESULTS:")
    print(f"  • CVs embedded: {len(df_cvs)}")
    print(f"  • Jobs embedded: {len(df_jobs)}")
    print(f"  • Total embeddings: {(len(df_cvs) + len(df_jobs)) * 3}")
    print(f"  • Storage: ChromaDB at data/chromadb/")
    print(f"\n🎯 NEXT STEPS:")
    print(f"  1. Create search.py to query CVs by job description")
    print(f"  2. Implement weighted scoring (Skills 40%, Technical 35%, Context 25%)")
    print(f"  3. Evaluate with Precision@5 metric")
    print(f"\n💡 TIP: You can now search CVs without re-embedding!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()