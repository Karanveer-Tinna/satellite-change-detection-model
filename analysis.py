from PIL import Image
import os, csv
from pathlib import Path

class_dict = {}
class_names = []

with open("Dataset/Main_Dataset/class_dict.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rgb = (int(row["r"]), int(row["g"]), int(row["b"]))
        name = row["name"]
        class_dict[rgb] = name
        class_names.append(name)

dataset_prepared = Path("Dataset/Prepared_Dataset")
out_csv = "full_eda_tiles.csv"
splits = ["train","val","test"]
# class_dict = {...}  # read from Main_Dataset/class_dict.csv, map rgb->name

with open(out_csv, "w", newline="") as fout:
    writer = csv.writer(fout)
    header = ["tile","split"] + [f"pct_{name}" for name in class_names]
    writer.writerow(header)
    for split in splits:
        mask_dir = dataset_prepared / split / "masks"
        for fname in os.listdir(mask_dir):
            if not fname.lower().endswith((".png",".jpg")): continue
            img = Image.open(mask_dir / fname).convert("RGB")
            w,h = img.size
            total = w*h
            colors = img.getcolors(maxcolors=100000)
            row = [fname, split] + [0]*len(class_names)
            for cnt,color in colors:
                name = class_dict.get(tuple(color), "UNKNOWN")
                row[2 + class_names.index(name)] = cnt/total*100
            writer.writerow(row)