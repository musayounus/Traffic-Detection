from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2, time, argparse, collections, json

MODEL_PATH = "weights_clean/yolov8n-fresh/weights/best.pt"
SAVE_COUNTS_PATH = "results/final_counts.json"

def main(src):
    model = YOLO(MODEL_PATH)
    tracker = DeepSort(max_age=30, n_init=2)
    cap = cv2.VideoCapture(0 if src.isdigit() else src)
    counts = collections.defaultdict(set)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        res = model(frame, conf=0.6, iou=0.5, verbose=False)[0]

        dets = []
        for b, cls, conf in zip(res.boxes, res.boxes.cls, res.boxes.conf):
            x1, y1, x2, y2 = b.xyxy[0]
            w, h = (x2 - x1).item(), (y2 - y1).item()
            if w * h < 900:  # Skip tiny boxes
                continue
            dets.append([[x1.item(), y1.item(), w, h], conf.item(), int(cls.item())])

        tracks = tracker.update_tracks(dets, frame=frame)

        for trk in tracks:
            if not trk.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, trk.to_ltrb())
            cls = int(getattr(trk, "det_class", 0))
            label = model.names[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}-{trk.track_id}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            counts[label].add(trk.track_id)

        overlay = "  ".join(f"{k}:{len(v)}" for k, v in counts.items())
        fps = 1.0 / (time.time() - t0 + 1e-6)
        cv2.putText(frame, f"{overlay} | {fps:.1f} FPS", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        total = sum(len(v) for v in counts.values())
        density = "Low" if total < 10 else "Medium" if total < 25 else "High"
        cv2.putText(frame, f"Density: {density}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Traffic", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    result = {k: len(v) for k, v in counts.items()}
    with open(SAVE_COUNTS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"✅ Saved vehicle counts to {SAVE_COUNTS_PATH}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0", help="0 = webcam or path to video file")
    main(p.parse_args().source)
