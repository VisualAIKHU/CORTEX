import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import h5py
import os

with open("data/RTE_datasets/CLEVR-Change/bef.json", "r") as file:
    data = json.load(file)

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_bert_embeddings_for_all(data):
    results = {}

    for entry in data:
        image_id = entry["image_id"]
        captions = entry["caption"]

        inputs = tokenizer(captions, return_tensors='pt', padding=True, truncation=True, max_length=512)

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy() # shape: (n_sentences, 768)
        results[image_id] = cls_embeddings

    return results
    
# Run BERT embedding
all_embeddings_dict = get_bert_embeddings_for_all(data)

# Save to HDF5 file
hdf5_file = "data/BefCap_chg.h5"
    
all_embeddings = []  # All embeddings
index_map = {}       # Each image's (start_idx, end_idx) information
current_index = 0

for image_id, emb_array in all_embeddings_dict.items():
    num_captions = emb_array.shape[0]  # Number of captions for the image
    all_embeddings.append(emb_array)   # Add embeddings to list
    
    # Record start, end indices
    start_idx = current_index
    end_idx = current_index + num_captions
    index_map[image_id] = (start_idx, end_idx)
    current_index = end_idx


# Concatenate all embeddings into one numpy array
all_embeddings = np.concatenate(all_embeddings, axis=0)  # shape: (total_captions, 768)

# Save to HDF5 file
with h5py.File(hdf5_file, "w") as hf:
    # 1. All embeddings (shape: (total_captions, 768))
    hf.create_dataset("embeddings", data=all_embeddings)

    # 2. Image names list (shape: (num_images,))
    str_dt = h5py.special_dtype(vlen=str)
    image_names = list(index_map.keys())
    image_names_array = np.array(image_names, dtype=str_dt)
    hf.create_dataset("image_names", data=image_names_array)

    # 3. Start, end indices (shape: (num_images, 2))
    start_end_list = np.array([index_map[name] for name in image_names], dtype=np.int32)
    hf.create_dataset("start_end_idx", data=start_end_list)


print(f"HDF5 saved: {hdf5_file}")


