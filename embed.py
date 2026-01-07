from dotenv import load_dotenv
load_dotenv()

import os
from mistralai import Mistral
from sklearn.metrics.pairwise import cosine_similarity   # <-- change

api_key = os.getenv("MISTRAL_API_KEY", "")
model = "mistral-embed"

client = Mistral(api_key=api_key)

def get_text_embedding(inputs):
    embeddings_batch_response = client.embeddings.create(
        model=model,
        inputs=inputs
    )
    return embeddings_batch_response.data[0].embedding

sentences = [
    "public speaking"
]

embeddings = [get_text_embedding([t]) for t in sentences]

reference_sentence = "electrical engineering"
reference_embedding = get_text_embedding([reference_sentence])

for t, e in zip(sentences, embeddings):
    similarity = cosine_similarity([e], [reference_embedding])[0][0]
    print("cosine similarity between public speaking and electrical engineering:")
    print(t, similarity)
