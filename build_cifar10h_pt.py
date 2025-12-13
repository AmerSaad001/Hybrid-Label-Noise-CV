import os
import numpy as np
import torch
import urllib.request

SAVE_PATH = "data/CIFAR-10_human.pt"
URL = "https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/cifar10h-probs.npy"

# -----------------------------------------------------------
# DOWNLOAD THE NP FILE
# -----------------------------------------------------------
print("Downloading CIFAR-10H probabilities (.npy)...")
urllib.request.urlretrieve(URL, "cifar10h_probs.npy")

probs = np.load("cifar10h_probs.npy")  # shape (10000,10)
print("Loaded probs:", probs.shape)

# -----------------------------------------------------------
# CLEAN LABEL: argmax
clean_label = probs.argmax(axis=1)

# -----------------------------------------------------------
# WORSE LABEL: second-highest probability
sorted_idx = np.argsort(probs, axis=1)
worse_label = sorted_idx[:, -2]

# -----------------------------------------------------------
# AGGRE LABELS: sample from distribution
aggre_label1 = []
aggre_label2 = []

for p in probs:
    aggre_label1.append(int(np.random.choice(10, p=p)))
    aggre_label2.append(int(np.random.choice(10, p=p)))

aggre_label1 = np.array(aggre_label1)
aggre_label2 = np.array(aggre_label2)

# -----------------------------------------------------------
# RANDOM LABELS
random_label1 = np.random.randint(0, 10, size=10000)
random_label2 = np.random.randint(0, 10, size=10000)

# -----------------------------------------------------------
# SAVE THE WHOLE STRUCTURE IN A PYTORCH FILE
data = {
    "clean_label": clean_label,
    "worse_label": worse_label,
    "aggre_label1": aggre_label1,
    "aggre_label2": aggre_label2,
    "random_label1": random_label1,
    "random_label2": random_label2
}

torch.save(data, SAVE_PATH)

print("\nDONE!")
print("Saved:", SAVE_PATH)
print("Size:", os.path.getsize(SAVE_PATH), "bytes")
print("Keys:", list(data.keys()))