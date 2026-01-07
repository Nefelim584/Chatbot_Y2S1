"""
MODEL 1: DEGREE RELEVANCE MATCHER
Trains a model to match degree fields/majors to job roles

Input: Degree field + Target role
Example: "Computer Science" vs "Software Engineer" → 0.95
         "Mechanical Engineering" vs "Software Engineer" → 0.15
         "BSc Science" vs "Software Engineer" → 0.60 (general science degree)
         "Mathematics" vs "Data Scientist" → 0.85

Dataset format: degree_relevance_pairs.jsonl
{"degree": "Computer Science", "role": "Software Engineer", "score": 0.95}
"""

import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import SentencesDataset
from torch.utils.data import DataLoader

# Configuration
DATA_PATH = r"D:\AI_CLINIC\degree_relevance_pairs.jsonl"
#BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_MODEL = "sentence-transformers/all-mpnet-base-v2"
#OUTPUT_DIR = "models/degree-relevance-matcher-miniLM"
OUTPUT_DIR = "models/degree-relevance-matcher-mpnet"
BATCH_SIZE = 16
EPOCHS = 10


def load_training_data(path: str):
    """
    Load degree relevance pairs

    Expected format:
    {"degree": "Computer Science", "role": "Software Engineer", "score": 0.95}
    {"degree": "BSc Science", "role": "Software Engineer", "score": 0.60}
    """
    examples = []

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)

            degree = row["degree"]
            role = row["role"]
            score = float(row["score"])

            examples.append(
                InputExample(
                    texts=[degree, role],
                    label=score
                )
            )

    return examples


def main():
    print("=" * 60)
    print("MODEL 1: DEGREE RELEVANCE MATCHER TRAINING")
    print("=" * 60)
    print(f"\nBase model: {BASE_MODEL}")
    print("Purpose: Learn which degrees are relevant for which roles")
    print("Handles: Country equivalents, general degrees (BSc Science, Math)")

    # Load model
    print("\nLoading base model...")
    model = SentenceTransformer(BASE_MODEL)

    # Load training data
    print(f"Loading training data from {DATA_PATH}...")
    train_examples = load_training_data(DATA_PATH)
    print(f"✓ Loaded {len(train_examples)} degree-role pairs")

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

    print("\n✓ Model 1 (Degree Relevance Matcher) saved successfully!")
    print("\nNext step: Train Model 2 (Education Level Matcher)")
    print("Run: python train_education_level_matcher.py")


if __name__ == "__main__":
    main()