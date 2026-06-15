import urllib.request
from pathlib import Path

BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
# fallback if that ever fails: "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

def main():
    # script lives in scripts/, so parent.parent is the project root
    dest = Path(__file__).resolve().parent.parent / "data" / "fashion_mnist"
    dest.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        target = dest / name
        if target.exists():
            print(f"skip {name} (already there)")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(BASE_URL + name, target)

    print(f"done -> {dest}")

if __name__ == "__main__":
    main()