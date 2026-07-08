import urllib.request
from pathlib import Path

BASE_URL = "https://data.brainchip.com/dataset-mirror/cifar10/" # fast mirror (~4 MB/s); the canonical https://www.cs.toronto.edu/~kriz/ is throttled to ~1 KB/s

FILES = [
    "cifar-10-python.tar.gz",
]

def main():
    # Resolve absolute path of the script, move up two levels, and target the 'data/cifar10' directory
    dest = Path(__file__).resolve().parent.parent / "data" / "cifar10" # script lives in scripts/, so parent.parent is the project root
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
