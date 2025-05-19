import random, shutil, pathlib

def subset(split, n):
    imgs = pathlib.Path(f"data/images/{split}")
    lbls = pathlib.Path(f"data/labels/{split}")
    dsti = pathlib.Path(f"data/images/{split}_mini"); dsti.mkdir(parents=True, exist_ok=True)
    dstl = pathlib.Path(f"data/labels/{split}_mini"); dstl.mkdir(parents=True, exist_ok=True)

    keep = random.sample(list(imgs.glob("*.jpg")), n)
    copied = 0
    for p in keep:
        l = lbls / f"{p.stem}.txt"
        if l.exists():                          
            shutil.copy2(p, dsti / p.name)
            shutil.copy2(l, dstl / l.name)
            copied += 1
    print(f"{split}: copied {copied} pairs")

if __name__ == "__main__":
    subset("train", 5000)     
    subset("val",   2500)    
