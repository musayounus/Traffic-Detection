import pathlib, fileinput

root = pathlib.Path("data/labels")
txts = list(root.rglob("*.txt"))
for lf in txts:
    with fileinput.FileInput(lf, inplace=True, backup=".bak", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:                      
                continue
            cls = int(parts[0]) - 1            
            parts[0] = str(cls)
            print(" ".join(parts))
print(f"✅  shifted IDs in {len(txts)} label files")
