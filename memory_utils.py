import json
import os

MEMORY_FILE = "memory.json"

# Hafızayı yükle
def load_memory():

    if not os.path.exists(MEMORY_FILE):
        return []

    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

# Hafızayı kaydet
def save_memory(memory):

    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

# Yeni kayıt ekle
def add_memory(image_name, objects):

    memory = load_memory()

    memory.append({
        "image": image_name,
        "objects": objects
    })

    save_memory(memory)

# Obje ara
def find_object(target):

    memory = load_memory()

    found_results = []

    for item in memory:

        for obj in item["objects"]:

            if target.lower() in obj.lower():

                found_results.append({
                    "image": item["image"],
                    "object": obj
                })

    return found_results