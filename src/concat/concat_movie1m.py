import json
import torch

# 1. Load the embedded data in two JSON files
with open("emb_sum_movielens1m.json", "r") as f:
    emb_sum_data = json.load(f)

with open("movie1m_user_embeddings.json", "r") as f:
    user_kg_emb_data = json.load(f)

# 2. Make sure user IDs are aligned in numerical order
user_ids = sorted(emb_sum_data.keys(), key=lambda x: int(x)) 

# 3. Convert the embedded data to a Tensor and make sure the embedding dimensions are correct
emb_sum_embeddings = torch.tensor([emb_sum_data[user_id] for user_id in user_ids])
user_kg_embeddings = torch.tensor([user_kg_emb_data[user_id] for user_id in user_ids])

# 4. directly splice the two embeddings
final_embeddings = {}
for i, user_id in enumerate(user_ids):
    # Concatenate the two embeddings into a new embedding
    concatenated_embedding = torch.cat((emb_sum_embeddings[i], user_kg_embeddings[i])).tolist()
    final_embeddings[user_id] = concatenated_embedding

# 5. save the final embedding, keeping the insertion order
with open("final_user_embeddings_movie1m.json", "w") as f:
    json.dump(final_embeddings, f, indent=4)

print("final_user_embeddings_movie1m.json")
