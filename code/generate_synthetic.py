import os
import random
import numpy as np

num_users = 300
num_items = 500
train_interactions = 8000
test_interactions = 2000
output_path = "../data/synthetic"

os.makedirs(output_path, exist_ok=True)

random.seed(2020)
np.random.seed(2020)

train_data = {u: set() for u in range(num_users)}

for _ in range(train_interactions):
    u = random.randint(0, num_users - 1)
    i = random.randint(0, num_items - 1)
    train_data[u].add(i)

test_data = {u: set() for u in range(num_users)}

for _ in range(test_interactions):
    u = random.randint(0, num_users - 1)
    i = random.randint(0, num_items - 1)
    if i not in train_data[u]:  # avoid overlap
        test_data[u].add(i)

with open(os.path.join(output_path, "train.txt"), "w") as f:
    for u in train_data:
        if len(train_data[u]) > 0:
            items = " ".join(map(str, train_data[u]))
            f.write(f"{u} {items}\n")

with open(os.path.join(output_path, "test.txt"), "w") as f:
    for u in test_data:
        if len(test_data[u]) > 0:
            items = " ".join(map(str, test_data[u]))
            f.write(f"{u} {items}\n")

with open(os.path.join(output_path, "user_list.txt"), "w") as f:
    for u in range(num_users):
        f.write(f"{u}\n")

with open(os.path.join(output_path, "item_list.txt"), "w") as f:
    for i in range(num_items):
        f.write(f"{i}\n")

print("Synthetic dataset created successfully!")
print(f"Users: {num_users}")
print(f"Items: {num_items}")
print(f"Train interactions: {train_interactions}")
print(f"Test interactions: {test_interactions}")
