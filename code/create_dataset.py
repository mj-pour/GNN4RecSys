import pandas as pd
import numpy as np
import os
from urllib.request import urlretrieve
import zipfile
from collections import defaultdict

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # code/
DATA_DIR = os.path.join(BASE_DIR, "..", "data")        # data/
OUT_DIR = os.path.join(DATA_DIR, "movielens-100k")     # data/movielens-100k/

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Download MovieLens 100K into data/
zip_path = os.path.join(DATA_DIR, "ml-100k.zip")
extract_path = os.path.join(DATA_DIR, "ml-100k")

if not os.path.exists(extract_path):
    print("Downloading MovieLens 100K...")
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATA_DIR)

# Load ratings
ratings_path = os.path.join(extract_path, "u.data")

df = pd.read_csv(
    ratings_path,
    sep="\t",
    names=["user", "item", "rating", "timestamp"]
)

# 3. Binarize ratings
df = df[df["rating"] >= 4].copy()
df["interaction"] = 1

# Remap user and item IDs to 0...N
unique_users = sorted(df["user"].unique())
unique_items = sorted(df["item"].unique())

user2id = {u: i for i, u in enumerate(unique_users)}
item2id = {m: i for i, m in enumerate(unique_items)}

df["user"] = df["user"].map(user2id)
df["item"] = df["item"].map(item2id)

# Build user → item list
user_items = defaultdict(list)
for u, i in zip(df["user"], df["item"]):
    user_items[u].append(i)

# Train/Test split (80/20 per user)
train_data = {}
test_data = {}

for u, items in user_items.items():
    items = list(set(items))
    np.random.shuffle(items)

    split = int(0.8 * len(items))
    train_data[u] = items[:split]
    test_data[u] = items[split:] if split < len(items) else []

# Write user_list.txt
with open(os.path.join(OUT_DIR, "user_list.txt"), "w") as f:
    f.write("org_id remap_id\n")
    for org, new in user2id.items():
        f.write(f"{org} {new}\n")

# Write item_list.txt
with open(os.path.join(OUT_DIR, "item_list.txt"), "w") as f:
    f.write("org_id remap_id\n")
    for org, new in item2id.items():
        f.write(f"{org} {new}\n")

# Write train.txt
with open(os.path.join(OUT_DIR, "train.txt"), "w") as f:
    for u in sorted(train_data.keys()):
        items = " ".join(str(i) for i in sorted(train_data[u]))
        f.write(f"{u} {items}\n")

# Write test.txt
with open(os.path.join(OUT_DIR, "test.txt"), "w") as f:
    for u in sorted(test_data.keys()):
        items = " ".join(str(i) for i in sorted(test_data[u]))
        f.write(f"{u} {items}\n")

print("Files created in:", OUT_DIR)
