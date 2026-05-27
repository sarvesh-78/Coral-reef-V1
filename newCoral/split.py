import os
import hashlib

def get_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def scan_folder(folder):
    hashes = {}
    for root, _, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            file_hash = get_hash(path)
            hashes[path] = file_hash
    return hashes

train_hashes = scan_folder("dataset/train")
val_hashes = scan_folder("dataset/val")
test_hashes = scan_folder("dataset/test")

train_set = set(train_hashes.values())
val_set = set(val_hashes.values())
test_set = set(test_hashes.values())

print("Train-Val duplicates:", len(train_set & val_set))
print("Train-Test duplicates:", len(train_set & test_set))
print("Val-Test duplicates:", len(val_set & test_set))
