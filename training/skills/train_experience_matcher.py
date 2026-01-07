import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import SentencesDataset
from torch.utils.data import DataLoader

DATA_PATH = "experience_position_pairs_v2.json"  # <-- JSON list: {position_1, position_2, score}
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "models/position-matcher-miniLM"

# BASE_MODEL = "intfloat/e5-small-v2"
# OUTPUT_DIR = "models/position-matcher-e5-small"


def load_training_data(path: str):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)  # list of dicts

    for row in data:
        pos_1 = (row.get("position_1") or "").strip()
        pos_2 = (row.get("position_2") or "").strip()
        score = float(row.get("score", 0.0))

        if not pos_1 or not pos_2:
            continue

        # clamp score to [0,1]
        if score < 0:
            score = 0.0
        elif score > 1:
            score = 1.0

        examples.append(
            InputExample(
                texts=[pos_1, pos_2],
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
    