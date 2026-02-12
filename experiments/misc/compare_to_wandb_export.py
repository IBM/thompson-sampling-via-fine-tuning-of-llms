import csv
import os

# === SETTINGS ===
csv_path = ""
folder_path = ""

# === 1. Extract names from CSV ===
csv_names = set()
with open(csv_path, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if "Name" in row:
            name = row["Name"].strip()
            csv_names.add(name)

# === 2. Extract names from folder ===
folder_names = set()
for filename in os.listdir(folder_path):
    if filename.endswith(".pt"):
        # Remove -BO-rewards and .pt suffix
        base = filename.replace("-BO-rewards", "").replace(".pt", "")
        folder_names.add(base)

# === 3. Compute differences ===
missing_in_folder = csv_names - folder_names
extra_in_folder = folder_names - csv_names

# === 4. Print results ===
print("\n✅ Names in CSV but missing in folder:")
print(sorted(missing_in_folder))

print("\n⚠️ Files in folder but not listed in CSV:")
print(sorted(extra_in_folder))

print("\n💀 Remove files from folder that are not in CSV:")
print('rm', ' '.join([extra+'*' for extra in sorted(extra_in_folder)]))