import os

def count_images(path):
    total = 0
    for root, dirs, files in os.walk(path):
        total += len(files)
    return total

print("Healthy:", count_images("dataset/train/healthy_corals"))
print("Bleached:", count_images("dataset/train/bleached_corals"))
