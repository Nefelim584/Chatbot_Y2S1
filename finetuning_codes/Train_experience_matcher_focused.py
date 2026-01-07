"""
MODEL 2: EXPERIENCE MATCHER
Trains a model to compare experience/seniority ASSUMING roles already match

Input: Experience descriptions (seniority, years, company)
Example: "Senior, 8 years at Google" vs "Mid-level, 3+ years" → 1.0
         "Junior, 2 years at Meta" vs "Senior, 6+ years" → 0.25

Dataset format: experience_pairs_focused.jsonl
{"cv_experience": "Senior, 8 years at Google", "job_experience": "Mid-level, 3+ years", "score": 1.0}

IMPORTANT: This dataset should ONLY contain examples where roles are similar!
(Role filtering is done by Model 1)
"""

import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import SentencesDataset
from torch.utils.data import DataLoader

# Configuration
DATA_PATH = r"D:\AI_CLINIC\experience_pairs_focused.jsonl"
#BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_MODEL = "sentence-transformers/all-mpnet-base-v2"

#OUTPUT_DIR = "models/experience-matcher-miniLM"
OUTPUT_DIR = "models/experience-matcher-mpnet"
BATCH_SIZE = 16
EPOCHS = 10


def load_training_data(path: str):
    """
    Load experience matching pairs

    Expected format:
    {"cv_experience": "Senior, 8 years at Google", "job_experience": "Mid-level, 3+ years", "score": 1.0}

    Note: These examples should ONLY be for similar roles!
    """
    examples = []

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)

            cv_experience = row["cv_experience"]
            job_experience = row["job_experience"]
            score = float(row["score"])

            examples.append(
                InputExample(
                    texts=[cv_experience, job_experience],
                    label=score
                )
            )

    return examples


def main():
    print("=" * 60)
    print("MODEL 2: EXPERIENCE MATCHER TRAINING")
    print("=" * 60)
    print(f"\nBase model: {BASE_MODEL}")
    print("Purpose: Learn experience/seniority matching")
    print("Note: Assumes roles are already similar (filtered by Model 1)")

    # Load model
    print("\nLoading base model...")
    model = SentenceTransformer(BASE_MODEL)

    # Load training data
    print(f"Loading training data from {DATA_PATH}...")
    train_examples = load_training_data(DATA_PATH)
    print(f"✓ Loaded {len(train_examples)} experience pairs")

    # Create dataset
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    # Loss function
    train_loss = losses.CosineSimilarityLoss(model=model)

    # Training configuration
    warmup_steps = int(0.1 * len(train_dataloader))

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total steps: {len(train_dataloader) * EPOCHS}")

    print("\n" + "=" * 60)
    print("STARTING TRAINING...")
    print("=" * 60 + "\n")

    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )

    print("\n" + "=" * 60)
    print("✓ Training complete!")
    print(f"✓ Saving model to: {OUTPUT_DIR}")
    print("=" * 60)

    # Save
    model.save(OUTPUT_DIR)

    print("\n✓ Model 2 (Experience Matcher) saved successfully!")
    print("\nNext step: Use both models together")
    print("Run: python dual_model_matcher.py")


if __name__ == "__main__":
    main()