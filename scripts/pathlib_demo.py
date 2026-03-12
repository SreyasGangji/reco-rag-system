from pathlib import Path

data_dir = Path("data")
raw_dir = data_dir / "raw"
ml_100k_dir = raw_dir / "ml-100k"
dataset_file = ml_100k_dir / "u.data"

print(data_dir)
print(raw_dir)

print(type(dataset_file))
print(ml_100k_dir.suffix)
print(dataset_file.name)
print(dataset_file.stem)
print(dataset_file.parent)
p = Path("data/raw/ml-100k/u.data")

print(p.name)
print(p.parent)