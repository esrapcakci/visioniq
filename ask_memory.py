import sys
import json

target = sys.argv[1].lower()

with open("memory.json", "r") as f:
    memory = json.load(f)

found = False

for item in memory:

    if target in item["object"].lower():

        found = True

        print(f"\n{item['object']} bulundu!")
        print(f"Fotoğraf: {item['image']}")
        print(f"Konum: {item['location']}")

if not found:
    print("\nObje hafızada bulunamadı.")