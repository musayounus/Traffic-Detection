import pathlib, fileinput

root = pathlib.Path("data/labels")
txts = list(root.rglob("*.txt"))
bad = 0

for lf in txts:
    new_lines = []
    with open(lf, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if int(float(parts[0])) < 0:
                bad += 1
                continue  # skip negative class IDs
            new_lines.append(" ".join(parts))
    with open(lf, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

print(f"✅ removed lines with negative class IDs from {bad} files")
