from pathlib import Path
import csv

from jetson_inference import detectNet
from jetson_utils import loadImage, saveImage


BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"
RESULT_DIR = BASE_DIR / "results"
IMAGE_NAMES = ("image1.jpg", "image2.jpg")
THRESHOLDS = (0.3, 0.7)


def main():
    for name in IMAGE_NAMES:
        if not (IMAGE_DIR / name).is_file():
            raise FileNotFoundError("Input image not found: " + str(IMAGE_DIR / name))

    RESULT_DIR.mkdir(exist_ok=True)
    detection_rows = []
    summary_rows = []

    for threshold in THRESHOLDS:
        # Same detector as the teaching example; confidence is applied by detectNet.
        net = detectNet("ssd-mobilenet-v2", threshold=threshold)

        for name in IMAGE_NAMES:
            image_path = IMAGE_DIR / name
            # Load the original separately for each threshold: Detect overlays boxes.
            image = loadImage(str(image_path))
            if image is None:
                raise RuntimeError("Could not load image: " + str(image_path))

            detections = net.Detect(image)
            summary_rows.append([name, threshold, len(detections)])
            print("\n{} | threshold {:.1f} | {} objects".format(
                name, threshold, len(detections)
            ))

            for detection in detections:
                class_id = int(detection.ClassID)
                class_name = net.GetClassDesc(class_id)
                confidence = float(detection.Confidence)
                left = float(detection.Left)
                top = float(detection.Top)
                right = float(detection.Right)
                bottom = float(detection.Bottom)
                width = right - left
                height = bottom - top
                area = width * height
                center_x = (left + right) / 2
                center_y = (top + bottom) / 2

                detection_rows.append([
                    name, threshold, class_id, class_name, confidence,
                    left, top, right, bottom, width, height, area,
                    center_x, center_y,
                ])
                print(
                    "  ClassID={} ({}) Confidence={:.3f} "
                    "Left={:.1f} Top={:.1f} Right={:.1f} Bottom={:.1f} "
                    "Width={:.1f} Height={:.1f} Area={:.1f} "
                    "Center=({:.1f}, {:.1f})".format(
                        class_id, class_name, confidence,
                        left, top, right, bottom, width, height, area,
                        center_x, center_y,
                    )
                )

            suffix = "03" if threshold == 0.3 else "07"
            result_path = RESULT_DIR / "{}_t{}.jpg".format(image_path.stem, suffix)
            saveImage(str(result_path), image)
            print("  Saved " + str(result_path))

    with (RESULT_DIR / "detections.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Image", "Threshold", "ClassID", "Class", "Confidence",
            "Left", "Top", "Right", "Bottom", "Width", "Height", "Area",
            "CenterX", "CenterY",
        ])
        writer.writerows(detection_rows)

    with (RESULT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Threshold", "Number of detected objects"])
        writer.writerows(summary_rows)

    print("\nDone. Results saved to " + str(RESULT_DIR))


if __name__ == "__main__":
    main()
