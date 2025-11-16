import numpy as np

# Load the .npy file
data = np.load("data/CIFAR-10N/CIFAR-10_human_ordered.npy", allow_pickle=True).item()

# Save each label type separately
np.save("data/CIFAR-10N/aggre.npy", data["aggre_label"])
np.save("data/CIFAR-10N/worse.npy", data["worse_label"])
np.save("data/CIFAR-10N/random1.npy", data["random_label1"])
np.save("data/CIFAR-10N/random2.npy", data["random_label2"])
np.save("data/CIFAR-10N/random3.npy", data["random_label3"])

print("DONE — generated aggre.npy, worse.npy, random1.npy, random2.npy, random3.npy")