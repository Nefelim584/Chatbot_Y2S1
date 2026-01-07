"""
MODEL 2: EDUCATION LEVEL MATCHER
Trains a model to match education levels (with country normalization)

Input: Candidate education level vs Required education level
Example: "Master's in Computer Science, 2 years" vs "Bachelor's required" → 1.0
         "BTech 4 years" vs "Bachelor's required" → 1.0
         "Grande École 5 years" vs "Master's required" → 1.0
         "BSc 3 years" vs "Master's required" → 0.40

Normalizes across countries:
- BTech = BSc = BEng = Licence (France) = Bachelor's
- MTech = MSc = Grande École = Master's
- Associate = Diploma = Sub-Bachelor's

Dataset format: education_level_pairs.jsonl
{"candidate_education": "MTech 2 years", "required_education": "Bachelor's", "score": 1.0}
"""

import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import SentencesDataset
from torch.utils.data import DataLoader

# Configuration
DATA_PATH = r"D:\AI_CLINIC\education_level_pairs.jsonl"
#BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_MODEL = "sentence-transformers/all-mpnet-base-v2"
#OUTPUT_DIR = "models/education-level-matcher-miniLM"
OUTPUT_DIR = "models/education-level-matcher-mpnet"
BATCH_SIZE = 16
EPOCHS = 10


def load_training_data(path: str):
    """
    Load education level pairs

    Expected format:
    {"candidate_education": "MTech 2 years", "required_education": "Bachelor's", "score": 1.0}
    {"candidate_education": "BSc 3 years", "required_education": "Master's", "score": 0.40}
    """
    examples = []

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)

            candidate_education = row["candidate_education"]
            required_education = row["required_education"]
            score = float(row["score"])

            examples.append(
                InputExample(
                    texts=[candidate_education, required_education],
                    label=score
                )
            )

    return examples


def main():
    print("=" * 60)
    print("MODEL 2: EDUCATION LEVEL MATCHER TRAINING")
    print("=" * 60)
    print(f"\nBase model: {BASE_MODEL}")
    print("Purpose: Match education levels across countries")
    print("Normalizes: BTech=BSc=Bachelor's, Grande École=Master's, etc.")

    # Load model
    print("\nLoading base model...")
    model = SentenceTransformer(BASE_MODEL)

    # Load training data
    print(f"Loading training data from {DATA_PATH}...")
    train_examples = load_training_data(DATA_PATH)
    print(f"✓ Loaded {len(train_examples)} education level pairs")

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

    print("\n✓ Model 2 (Education Level Matcher) saved successfully!")
    print("\nNext step: Use both models together")
    print("Run: python dual_education_matcher.py")


if __name__ == "__main__":
    main()