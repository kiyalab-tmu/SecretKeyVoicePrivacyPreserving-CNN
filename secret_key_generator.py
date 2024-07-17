from secret_key import SecretKey, KeyType
import os
import random
from tqdm import tqdm

def generate_keys(key_type: KeyType, num_keys: int, key_dims: int, num_trials: int, key_prefix: str, export_dir: str):
    os.makedirs(export_dir, exist_ok=True)    
    key_files = []
    for i in tqdm(range(num_keys), desc="Generating ..."):
        if key_type == KeyType.ROM:
            key = SecretKey(key_type=KeyType.ROM, key_dims=key_dims)
        elif key_type == KeyType.SHUFFLE:
            key = SecretKey(key_type=KeyType.SHUFFLE, key_dims=key_dims)
        elif key_type == KeyType.FLIP:
            key = SecretKey(key_type=KeyType.FLIP, key_dims=key_dims)
        file_path = os.path.join(export_dir, f"{key_prefix}_{i}.pkl")
        key.save(file_path)
        key_files.append(file_path)

    with open(os.path.join(export_dir, f"trial.txt"), "w") as f:
        k = num_trials - 1 if num_trials > 0 else len(key_files) - 1
        for key1 in key_files:
            for key2 in random.sample(list(set(key_files) - set([key1])), k) + [key1]:
                label = 1 if key1 == key2 else 0
                f.write(f"{label} {key1} {key2}\n")
            if num_trials < 0:
                break