import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import SentencesDataset
from torch.utils.data import DataLoader

# ===========================================================
# CONFIG
# ===========================================================
DATA_PATH = "education_field_pairs_v2.json"  # <-- your generated JSON (list of dicts)
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "models/education-field-matcher-miniLM"

# BASE_MODEL = "intfloat/e5-small-v2"
# OUTPUT_DIR = "models/education-field-matcher-e5-small"


# ===========================================================
# Load training data (field_1, field_2, score)
# ===========================================================
def load_training_data(path: str):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)  # list of rows

    for row in data:
        field_1 = (row.get("field_1") or "").strip()
        field_2 = (row.get("field_2") or "").strip()
        score = float(row.get("score", 0.0))

        if not field_1 or not field_2:
            continue

        # Clamp score to [0,1] just in case
        if score < 0:
            score = 0.0
        elif score > 1:
            score = 1.0

        examples.append(
            InputExample(
                texts=[field_1, field_2],
                label=score
            )
        )

    return examples


def main():
    print("Loading base model:", BASE_MODEL)
    model = SentenceTransformer(BASE_MODEL)

    print("Loading training data...")
    train_examples = load_training_data(DATA_PATH)
    print(f"Loaded {len(train_examples)} training pairs")

    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)

    train_loss = losses.CosineSimilarityLoss(model=model)

    num_epochs = 5
    warmup_steps = int(0.1 * len(train_dataloader))

    print("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )

    print(f"Saving fine-tuned model to {OUTPUT_DIR}...")
    model.save(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
