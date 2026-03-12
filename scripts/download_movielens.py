import zipfile
import urllib.request
from pathlib import Path

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "ml-100k.zip"

    # 1) Download (only if not already downloaded)
    if not zip_path.exists():
        print(f"Downloading MovieLens 100k from:\n{MOVIELENS_100K_URL}")
        urllib.request.urlretrieve(MOVIELENS_100K_URL, zip_path)
        print(f"Saved zip to: {zip_path}")
    else:
        print(f"Zip already exists: {zip_path}")

    # 2) Extract (only if not already extracted)
    extract_dir = raw_dir / "ml-100k"
    if extract_dir.exists():
        print(f"Already extracted: {extract_dir}")
        return

    print("Extracting zip...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(raw_dir)

    if not extract_dir.exists():
        raise RuntimeError("Extraction failed: 'ml-100k' folder not found after unzip")

    print(f"Extracted to: {extract_dir}")

if __name__ == "__main__":
    main()