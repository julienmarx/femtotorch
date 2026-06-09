import urllib.request
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
# fallback if that ever fails: "https://ossci-datasets.s3.amazonaws.com/mnist/"

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

def main():
    # Resolve absolute path of the script, move up two levels, and target the 'data/mnist' directory
    dest = Path(__file__).resolve().parent.parent / "data" / "mnist" # script lives in scripts/, so parent.parent is the project root
    dest.mkdir(parents=True, exist_ok=True) #

    for name in FILES:
        target = dest / name
        if target.exists():
            print(f"skip {name} (already there)")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(BASE_URL + name, target) # source is BASE_URL + name ; destination is target

    print(f"done -> {dest}")

if __name__ == "__main__":
    main()