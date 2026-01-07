import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import SentencesDataset
from torch.utils.data import DataLoader

DATA_PATH = "skills_pairs_from_uploaded_list.jsonl"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "models/skill-matcher-miniLM"

# BASE_MODEL = "intfloat/e5-small-v2"
# OUTPUT_DIR = "models/skill-matcher-e5-small"

def load_training_data(path: str):
    examples = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cv_skills = row["cv_skills"]
            job_skills = row["job_skills"]
            score = float(row["score"])

            cv_text = "; ".join(cv_skills)
            job_text = "; ".join(job_skills)

            examples.append(
                InputExample(
                    texts=[cv_text, job_text],
                    label=score  # float in [0,1]
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
