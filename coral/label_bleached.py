import os

base = "dataset/val"
for folder in os.listdir(base):
    folder_path = os.path.join(base, folder)
    if os.path.isdir(folder_path):
        count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.png','.jpeg'))])
        print(f"{folder}: {count} images")
