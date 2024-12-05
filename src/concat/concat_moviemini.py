import json
import torch

# 1. Load the embedded data in two JSON files
with open("emb_sum_movielens1m.json", "r") as f:
    emb_sum_data = json.load(f)

with open("user_kg_emb_movielens1m.json", "r") as f:
    user_kg_emb_data = json.load(f)

# 2. Ensure that user IDs are aligned
user_ids = sorted(emb_sum_data.keys())
emb_sum_embeddings = torch.tensor([emb_sum_data[user_id] for user_id in user_ids])
user_kg_embeddings = torch.tensor([user_kg_emb_data[user_id] for user_id in user_ids])

# 3. Direct splicing of two embeds
final_embeddings = {}
for i, user_id in enumerate(user_ids):
    # Direct splice of two embeds
    concatenated_embedding = torch.cat((emb_sum_embeddings[i], user_kg_embeddings[i])).tolist()
    final_embeddings[user_id] = concatenated_embedding

# 4.Direct splice of two embeds
with open("final_user_embeddings_concat.json", "w") as f:
    json.dump(final_embeddings, f, indent=4)
print("final_user_embeddings_concat.json")
