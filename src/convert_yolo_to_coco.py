from pathlib import Path
import json, cv2, tqdm

CATS = ["car", "bus", "truck", "motorcycle"]   

def convert(split):
    img_dir = Path(f"data/images/{split}")
    lab_dir = Path(f"data/labels/{split}")
    images, annotations, ann_id = [], [], 1

    for img_id, img_path in enumerate(tqdm.tqdm(sorted(img_dir.glob("*.jpg")), desc=split), 1):
        h, w = cv2.imread(str(img_path)).shape[:2]
        images.append({"id": img_id, "file_name": img_path.name, "height": h, "width": w})

        txt_path = lab_dir / f"{img_path.stem}.txt"
        if not txt_path.exists():   
            continue                         

        for line in open(txt_path):
            cls, xc, yc, bw, bh = map(float, line.split())
            x1 = (xc - bw/2) * w
            y1 = (yc - bh/2) * h
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cls) + 1,    
                "bbox": [x1, y1, bw*w, bh*h],
                "area": bw*bh*w*h,
                "iscrowd": 0
            })
            ann_id += 1

    cats = [{"id": i+1, "name": n} for i, n in enumerate(CATS)]
    out = {"images": images, "annotations": annotations, "categories": cats}
    json.dump(out, open(f"data/{split}.coco.json", "w"))
    print(f"✅  wrote data/{split}.coco.json  |  {len(images)} images, {len(annotations)} boxes")

if __name__ == "__main__":
    for sp in ["train", "val"]:
        convert(sp)