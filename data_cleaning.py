"""
data_cleaning_v2.py - FIXED: Clean and split dataset with HIGH QUALITY focus
Removes unwanted fields (70-94% missing) and prepares for 3-vector system
FIX: Handles job column names correctly
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import re


def identify_data_type(df):
    """
    Identify whether each row is a CV or a Job Posting
    CVs have: skills, career_objective, educational_institution_name
    Jobs have: job_position_name, skills_required, responsibilities.1
    """
    # Check for job posting columns (with BOM character handling)
    job_cols = ['job_position_name', 'ï»¿job_position_name', 'skills_required',
                'educationaL_requirements', 'responsibilities.1']
    cv_cols = ['skills', 'career_objective', 'educational_institution_name']

    # Create indicators
    df['has_job_data'] = False
    df['has_cv_data'] = False

    for col in job_cols:
        if col in df.columns:
            df['has_job_data'] = df['has_job_data'] | df[col].notna()

    for col in cv_cols:
        if col in df.columns:
            df['has_cv_data'] = df['has_cv_data'] | df[col].notna()

    # Classify each row
    df['data_type'] = 'unknown'
    df.loc[df['has_cv_data'] & ~df['has_job_data'], 'data_type'] = 'cv'
    df.loc[df['has_job_data'] & ~df['has_cv_data'], 'data_type'] = 'job'
    df.loc[df['has_job_data'] & df['has_cv_data'], 'data_type'] = 'mixed'

    return df


def clean_cvs(df_cvs):
    """
    Clean CV data - IMPROVED VERSION
    Focus: Remove unwanted fields and keep only high-quality data
    """
    print("\n📄 CLEANING CV DATA (IMPROVED)")
    print("-" * 40)
    initial_count = len(df_cvs)

    # Drop job-related columns
    job_columns = [
        'job_position_name', 'ï»¿job_position_name',
        'educationaL_requirements', 'experiencere_requirement',
        'age_requirement', 'responsibilities.1',
        'skills_required', 'matched_score'
    ]
    columns_to_drop = [col for col in job_columns if col in df_cvs.columns]
    df_cvs = df_cvs.drop(columns=columns_to_drop)
    print(f"✅ Dropped {len(columns_to_drop)} job-related columns")

    # Drop low-value/privacy columns
    low_value_columns = [
        'address',  # Privacy concern
        'company_urls', 'online_links',  # Mostly empty URLs
        'issue_dates', 'expiry_dates',  # Mostly empty
        'extra_curricular_organization_links'  # URLs not needed
    ]
    columns_to_drop = [col for col in low_value_columns if col in df_cvs.columns]
    df_cvs = df_cvs.drop(columns=columns_to_drop)
    print(f"✅ Dropped {len(columns_to_drop)} low-value columns")

    # 🔴 NEW: Drop UNWANTED high-missing fields (70-94% missing)
    unwanted_columns = [
        # Extra-curricular (66-74% missing)
        'extra_curricular_activity_types',
        'extra_curricular_organization_names',
        'role_positions',
        # Languages (93% missing)
        'languages',
        'proficiency_levels',
        # Certifications (79-86% missing)
        'certification_providers',
        'certification_skills',
        # Educational results (60-64% missing + lots of N/A)
        'educational_results',
        'result_types',
        # Locations (45% missing + lots of N/A text)
        'locations'
    ]

    columns_to_drop = [col for col in unwanted_columns if col in df_cvs.columns]
    df_cvs = df_cvs.drop(columns=columns_to_drop)
    print(f"✅ Dropped {len(columns_to_drop)} unwanted high-missing fields")
    print(f"   (This removes 70-94% missing data and reduces N/A leakage)")

    # Keep CVs that have at least ONE of the main matching criteria
    # Main criteria: skills, education, or experience
    main_fields = ['skills', 'educational_institution_name', 'positions']

    df_cvs['has_skills'] = df_cvs['skills'].notna()
    df_cvs['has_education'] = df_cvs['educational_institution_name'].notna()
    df_cvs['has_experience'] = df_cvs['positions'].notna()

    # Keep if has ANY of the main fields
    df_cvs['keep_cv'] = (df_cvs['has_skills'] | df_cvs['has_education'] | df_cvs['has_experience'])

    rows_before = len(df_cvs)
    df_cvs = df_cvs[df_cvs['keep_cv']].copy()
    df_cvs = df_cvs.drop(columns=['has_skills', 'has_education', 'has_experience', 'keep_cv'])
    removed = rows_before - len(df_cvs)
    print(f"✅ Removed {removed} CVs with NO skills/education/experience")

    # Remove duplicates based on key fields
    df_cvs = df_cvs.drop_duplicates(subset=['skills', 'career_objective'], keep='first')

    # Reset index to use as CV ID
    df_cvs = df_cvs.reset_index(drop=True)
    df_cvs['cv_id'] = df_cvs.index

    final_count = len(df_cvs)
    print(f"✅ Removed {initial_count - final_count} low-quality CVs")
    print(f"✅ Final CV dataset: {final_count} high-quality entries")

    return df_cvs


def clean_jobs(df_jobs):
    """Clean job posting data - FIXED VERSION"""
    print("\n💼 CLEANING JOB POSTING DATA")
    print("-" * 40)
    initial_count = len(df_jobs)

    # First, check what columns actually exist
    print(f"  Columns found: {list(df_jobs.columns)}")

    # Handle BOM character in column name
    if 'ï»¿job_position_name' in df_jobs.columns:
        df_jobs = df_jobs.rename(columns={'ï»¿job_position_name': 'job_position_name'})

    # List of possible job columns to keep
    possible_job_columns = [
        'job_position_name', 'skills_required',
        'educationaL_requirements', 'educational_requirements',
        'experiencere_requirement', 'experience_requirement',
        'age_requirement', 'responsibilities.1', 'responsibilities'
    ]

    # Keep only columns that actually exist
    columns_to_keep = [col for col in possible_job_columns if col in df_jobs.columns]

    if len(columns_to_keep) == 0:
        print("  ⚠️  No job columns found - dataset may not contain job data")
        print("  Creating empty job dataset...")
        return pd.DataFrame(columns=['job_id', 'job_position_name', 'skills_required',
                                     'educational_requirements', 'experience_requirement',
                                     'age_requirement', 'responsibilities'])

    df_jobs = df_jobs[columns_to_keep].copy()
    print(f"  ✅ Kept {len(columns_to_keep)} relevant columns")

    # Rename for consistency
    rename_map = {
        'responsibilities.1': 'responsibilities',
        'educationaL_requirements': 'educational_requirements',
        'experiencere_requirement': 'experience_requirement'
    }
    df_jobs = df_jobs.rename(columns={k: v for k, v in rename_map.items() if k in df_jobs.columns})

    # Check if we have the essential fields after renaming
    if 'job_position_name' not in df_jobs.columns or 'skills_required' not in df_jobs.columns:
        print("  ⚠️  Missing essential job fields after processing")
        print("  Creating empty job dataset...")
        return pd.DataFrame(columns=['job_id', 'job_position_name', 'skills_required',
                                     'educational_requirements', 'experience_requirement',
                                     'age_requirement', 'responsibilities'])

    # Remove jobs without essential fields
    essential_fields = ['job_position_name', 'skills_required']
    for field in essential_fields:
        if field in df_jobs.columns:
            df_jobs = df_jobs[df_jobs[field].notna()]

    if len(df_jobs) == 0:
        print("  ⚠️  No valid jobs after filtering")
        print("  Creating empty job dataset...")
        return pd.DataFrame(columns=['job_id', 'job_position_name', 'skills_required',
                                     'educational_requirements', 'experience_requirement',
                                     'age_requirement', 'responsibilities'])

    # Remove duplicates (only if we have data)
    df_jobs = df_jobs.drop_duplicates(subset=['job_position_name', 'skills_required'], keep='first')

    # Reset index to use as job ID
    df_jobs = df_jobs.reset_index(drop=True)
    df_jobs['job_id'] = df_jobs.index

    final_count = len(df_jobs)
    print(f"✅ Removed {initial_count - final_count} incomplete job postings")
    print(f"✅ Final job dataset: {final_count} complete job postings")

    return df_jobs


def analyze_datasets(df_cvs, df_jobs):
    """Print statistics about both datasets"""
    print("\n📊 DATASET STATISTICS (V2 - HIGH QUALITY)")
    print("=" * 50)

    # CV Statistics
    print("\n📄 CV Dataset:")
    print(f"  • Total CVs: {len(df_cvs)}")
    print(f"  • Columns: {len(df_cvs.columns)}")

    cv_fields = ['skills', 'career_objective', 'educational_institution_name',
                 'positions', 'responsibilities']
    print("\n  Field Coverage:")
    for field in cv_fields:
        if field in df_cvs.columns:
            coverage = df_cvs[field].notna().sum() / len(df_cvs) * 100
            print(f"    - {field:30s}: {coverage:5.1f}%")

    # Job Statistics
    print("\n💼 Job Posting Dataset:")
    print(f"  • Total Jobs: {len(df_jobs)}")
    print(f"  • Columns: {len(df_jobs.columns)}")

    if len(df_jobs) > 0:
        job_fields = ['job_position_name', 'skills_required', 'responsibilities', 'educational_requirements']
        print("\n  Field Coverage:")
        for field in job_fields:
            if field in df_jobs.columns:
                coverage = df_jobs[field].notna().sum() / len(df_jobs) * 100
                print(f"    - {field:30s}: {coverage:5.1f}%")
    else:
        print("  ⚠️  No job data found in dataset")

    print("\n✨ Ready for 3-vector processing!")
    print("  Skills + Technical + Context vectors")


def main():
    """Main function to clean and split the dataset"""

    # Setup paths
    base_path = Path.cwd()
    input_file = base_path / 'data' / 'raw' / 'resume_data.csv'
    output_dir = base_path / 'data' / 'processed'
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("🧹 DATA CLEANING V2 - HIGH QUALITY FOCUS")
    print("=" * 60)
    print("Purpose: Remove unwanted fields & prepare for 3-vector system")

    # Load data
    print(f"\n📂 Loading data from: {input_file}")
    try:
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df)} total rows")
    except FileNotFoundError:
        print(f"❌ Error: File not found!")
        print(f"  Please copy resume_data.csv to: {input_file}")
        return None, None

    # Identify data types
    print("\n🔍 Identifying data types...")
    df = identify_data_type(df)

    # Split by type
    df_cvs = df[df['data_type'].isin(['cv', 'mixed'])].copy()
    df_jobs = df[df['data_type'].isin(['job', 'mixed'])].copy()
    df_unknown = df[df['data_type'] == 'unknown']

    print(f"  • CV rows: {len(df_cvs)}")
    print(f"  • Job rows: {len(df_jobs)}")
    print(f"  • Unknown/Empty rows: {len(df_unknown)} (discarded)")

    # Clean both datasets
    df_cvs_clean = clean_cvs(df_cvs)
    df_jobs_clean = clean_jobs(df_jobs)

    # Analyze results
    analyze_datasets(df_cvs_clean, df_jobs_clean)

    # Save cleaned datasets
    print("\n💾 SAVING CLEANED DATA")
    print("-" * 40)

    # Save CVs
    cv_output = output_dir / 'cvs_clean.csv'
    df_cvs_clean.to_csv(cv_output, index=False)
    print(f"✅ Saved {len(df_cvs_clean)} CVs to: cvs_clean.csv")

    # Save Jobs
    jobs_output = output_dir / 'jobs_clean.csv'
    df_jobs_clean.to_csv(jobs_output, index=False)
    print(f"✅ Saved {len(df_jobs_clean)} jobs to: jobs_clean.csv")

    # Save sample data for inspection
    samples = {
        'sample_cvs': df_cvs_clean.head(2).to_dict('records'),
        'sample_jobs': df_jobs_clean.head(2).to_dict('records') if len(df_jobs_clean) > 0 else [],
        'statistics': {
            'total_cvs': len(df_cvs_clean),
            'total_jobs': len(df_jobs_clean),
            'cv_columns': list(df_cvs_clean.columns),
            'job_columns': list(df_jobs_clean.columns),
            'removed_unwanted_fields': 10,
            'quality_improvements': 'Removed 70-94% missing fields'
        }
    }

    sample_file = output_dir / 'cleaning_summary_v2.json'
    with open(sample_file, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"✅ Saved summary to: cleaning_summary_v2.json")

    print("\n" + "=" * 60)
    print("✅ DATA CLEANING V2 COMPLETE!")
    print("=" * 60)
    print(f"\n📈 IMPROVEMENTS:")
    print(f"  • Removed 10 unwanted fields (70-94% missing)")
    print(f"  • Higher quality data (less N/A)")
    print(f"  • Ready for 3-vector approach")
    print(f"\nNEXT STEP: Run data_preparation_v2.py to create 3-vector format")
    print("  (Skills + Technical + Context)")
    print("=" * 60 + "\n")

    return df_cvs_clean, df_jobs_clean


if __name__ == "__main__":
    main()