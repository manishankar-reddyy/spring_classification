import argparse
from collections import deque
import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_DIR = Path(r"C:\Users\manis\OneDrive\Documents\Spring Analysis Project")
MODEL_PATH = PROJECT_DIR / "my_model.pt"
CALIBRATION_PATH = PROJECT_DIR / "spring_calibration.json"
OUTPUT_DIR = PROJECT_DIR / "output"
WINDOW_NAME = "Spring Detector"
HISTORY_LENGTH = 15
MIN_ROW_FILL_RATIO = 0.18
MIN_OD_ROW_FILL_RATIO = 0.25
OD_WIDTH_PERCENTILE = 90

SPRING_SPECS = [
    {"family": "Mod-2 BOSTHS", "type": "Outer", "height_mm": 253.0, "od_mm": 136.0},
    {"family": "Mod-2 BOSTHS", "type": "Inner", "height_mm": 222.0, "od_mm": 87.0},
    {"family": "Mod-2 BOSTHS", "type": "Snubber", "height_mm": 304.0, "od_mm": 104.0},
    {"family": "Mod-1 BOSTHS", "type": "Outer", "height_mm": 253.0, "od_mm": 136.0},
    {"family": "Mod-1 BOSTHS", "type": "Inner", "height_mm": 225.0, "od_mm": 85.0},
    {"family": "Mod-1 BOSTHS", "type": "Snubber", "height_mm": 304.0, "od_mm": 104.0},
    {"family": "HS", "type": "Outer", "height_mm": 260.0, "od_mm": 137.0},
    {"family": "HS", "type": "Inner", "height_mm": 243.0, "od_mm": 88.0},
    {"family": "HS", "type": "Snubber", "height_mm": 293.0, "od_mm": 104.0},
    {"family": "NL", "type": "Outer", "height_mm": 260.0, "od_mm": 140.0},
    {"family": "NL", "type": "Inner", "height_mm": 262.0, "od_mm": 86.0},
    {"family": "NL", "type": "Snubber", "height_mm": 294.0, "od_mm": 98.0},
    {"family": "LCCF20", "type": "Outer", "height_mm": 260.0, "od_mm": 140.0},
    {"family": "LCCF20", "type": "Inner", "height_mm": 243.0, "od_mm": 103.5},
    {"family": "LCCF20", "type": "Snubber", "height_mm": 248.0, "od_mm": 103.5},
]

HEIGHT_TOL_MM = 18.0
OD_TOL_MM = 12.0
HEIGHT_SCALE_MM = 8.0
OD_SCALE_MM = 6.0
AMBIGUITY_SCORE_GAP = 0.35


def classify_family_type_from_height_od(height_mm: float, od_mm: float | None) -> dict:
    if od_mm is None or od_mm <= 0:
        return {
            "family": "Unknown",
            "type": "Unknown",
            "confidence": 0.0,
            "alerts": ["OD_REQUIRED"],
        }

    scored = []
    for spec in SPRING_SPECS:
        height_error = abs(height_mm - spec["height_mm"])
        od_error = abs(od_mm - spec["od_mm"])
        score = (height_error / HEIGHT_SCALE_MM) + (od_error / OD_SCALE_MM)
        item = dict(spec)
        item["height_error_mm"] = height_error
        item["od_error_mm"] = od_error
        item["score"] = score
        scored.append(item)

    scored.sort(key=lambda item: item["score"])
    best = scored[0]
    if best["height_error_mm"] > HEIGHT_TOL_MM or best["od_error_mm"] > OD_TOL_MM:
        return {
            "family": "Unknown",
            "type": "Unknown",
            "confidence": 0.0,
            "alerts": ["MEASUREMENT_OUT_OF_RANGE"],
        }

    close_matches = [
        item for item in scored
        if item["score"] <= best["score"] + AMBIGUITY_SCORE_GAP
    ]
    families = []
    types = []
    for item in close_matches:
        if item["family"] not in families:
            families.append(item["family"])
        if item["type"] not in types:
            types.append(item["type"])

    alerts: list[str] = []
    if len(types) == 1 and len(families) > 1:
        family_name = "/".join(families)
        spring_type = types[0]
        alerts.append("FAMILY_AMBIGUOUS")
        confidence = 0.55
    elif len(families) == 1 and len(types) > 1:
        family_name = families[0]
        spring_type = "Unknown"
        alerts.append("TYPE_AMBIGUOUS")
        confidence = 0.4
    elif len(families) > 1 and len(types) > 1:
        family_name = "Unknown"
        spring_type = "Unknown"
        alerts.append("FAMILY_TYPE_AMBIGUOUS")
        confidence = 0.25
    else:
        family_name = best["family"]
        spring_type = best["type"]
        confidence = 0.92
        if best["height_error_mm"] > 4.0 or best["od_error_mm"] > 4.0:
            confidence = 0.75

    return {
        "family": family_name,
        "type": spring_type,
        "confidence": round(confidence, 2),
        "alerts": alerts,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect spring and estimate height from webcam, image, or video.")
    parser.add_argument("--source",default="camera:0",help=("Input source. Use 'camera:0' for webcam, or provide an image/video path. ""Default: camera:0"),)
    parser.add_argument("--conf", type=float, default=0.75, help="Confidence threshold. Default: 0.75")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration mode using webcam only.")
    parser.add_argument("--reference-mm",type=float,default=100.0,help="Known reference object height in millimeters for calibration.",)
    parser.add_argument("--label",default="SPRING",help="Display label to show with height. Default: SPRING",)
    parser.add_argument("--resolution",default="1280x720",help="Display resolution in WxH format. Default: 1280x720",)
    return parser.parse_args()


def parse_resolution(resolution: str) -> tuple[int, int]:
    width_str, height_str = resolution.lower().split("x")
    return int(width_str), int(height_str)


def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    print("Loading model...")
    model = YOLO(str(MODEL_PATH), task="detect")
    print(f"Classes: {model.names}")
    return model


def save_calibration(reference_mm: float, pixels_per_mm: float, frame_width: int, frame_height: int) -> None:
    data = {
        "reference_height_mm": reference_mm,
        "pixels_per_mm": pixels_per_mm,
        "frame_width": frame_width,
        "frame_height": frame_height,
    }
    CALIBRATION_PATH.write_text(json.dumps(data, indent=2))


def load_calibration() -> dict:
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {CALIBRATION_PATH}. Run this script with --calibrate first."
        )

    data = json.loads(CALIBRATION_PATH.read_text())
    pixels_per_mm = float(data["pixels_per_mm"])
    if pixels_per_mm <= 0:
        raise ValueError("Invalid pixels_per_mm value in calibration file.")
    return data


def parse_source(source: str) -> tuple[str, str | int]:
    if source.lower().startswith("camera:"):
        return "camera", int(source.split(":", 1)[1])

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {path}")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
    ext = path.suffix.lower()

    if ext in image_exts:
        return "image", str(path)
    if ext in video_exts:
        return "video", str(path)

    raise ValueError(f"Unsupported source type: {path}")


def open_capture(source_kind: str, source_value: str | int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source_value)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {source_kind} source: {source_value}")
    print(f"{source_kind.capitalize()} ready")
    return cap


def get_best_detection(result, min_conf: float):
    best_box = None
    best_area = -1.0

    for box in result.boxes:
        conf = float(box.conf[0].item())
        if conf < min_conf:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_area = area
            best_box = box

    return best_box


def extract_spring_mask(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy.astype(int)
    h, w = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    mask = np.zeros((h, w), dtype=np.uint8)
    if x2 <= x1 or y2 <= y1:
        return mask

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, roi_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(roi_mask) > 127:
        roi_mask = cv2.bitwise_not(roi_mask)

    kernel = np.ones((3, 3), np.uint8)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask[y1:y2, x1:x2] = roi_mask
    return mask


def refine_spring_height(frame: np.ndarray, xyxy: np.ndarray):
    x1, y1, x2, y2 = xyxy.astype(int)
    h, w = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(mask) > 127:
        mask = cv2.bitwise_not(mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    row_fill = np.count_nonzero(mask, axis=1) / max(1, mask.shape[1])
    valid_rows = np.where(row_fill > MIN_ROW_FILL_RATIO)[0]
    if len(valid_rows) == 0:
        return None

    top_local = int(valid_rows[0])
    bottom_local = int(valid_rows[-1])
    height_px = float(bottom_local - top_local + 1)
    if height_px < 20:
        return None

    return {
        "top_y": y1 + top_local,
        "bottom_y": y1 + bottom_local,
        "height_px": height_px,
    }


def refine_outer_diameter(mask: np.ndarray, xyxy: np.ndarray):
    x1, y1, x2, y2 = xyxy.astype(int)
    h, w = mask.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    roi_mask = mask[y1:y2, x1:x2]
    if roi_mask.size == 0:
        return None

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    od_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    od_mask = cv2.morphologyEx(od_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    row_fill = np.count_nonzero(od_mask, axis=1) / max(1, od_mask.shape[1])
    valid_rows = np.where(row_fill > MIN_OD_ROW_FILL_RATIO)[0]
    if len(valid_rows) == 0:
        return None

    row_samples = []
    for row_idx in valid_rows:
        cols = np.where(od_mask[row_idx] > 0)[0]
        if len(cols) < 2:
            continue
        left = int(cols.min())
        right = int(cols.max())
        width = float(right - left + 1)
        row_samples.append((int(row_idx), left, right, width))

    if not row_samples:
        return None

    widths = np.array([sample[3] for sample in row_samples], dtype=float)
    diameter_px = float(np.percentile(widths, OD_WIDTH_PERCENTILE))
    if diameter_px < 20:
        return None

    mid_row = float(np.median(valid_rows))
    candidate_rows = [
        sample for sample in row_samples
        if sample[3] >= diameter_px - 2.0
    ]
    if not candidate_rows:
        candidate_rows = row_samples

    best_row, left_local, right_local, representative_width = min(
        candidate_rows,
        key=lambda sample: (abs(sample[0] - mid_row), -sample[3]),
    )

    return {
        "diameter_px": diameter_px,
        "representative_width_px": float(representative_width),
        "line_y": y1 + int(best_row),
        "left_x": x1 + left_local,
        "right_x": x1 + right_local,
    }


def format_height_label(label_name: str, height_mm: float) -> str:
    return f"{label_name} | {height_mm:.2f}mm"


def draw_result(frame: np.ndarray, xyxy: np.ndarray, text: str) -> np.ndarray:
    display = frame.copy()
    x1, y1, x2, y2 = xyxy.astype(int)
    color = (0, 255, 0)

    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    box_x1 = min(max(0, x2 + 8), max(0, display.shape[1] - text_w - 10))
    box_y1 = max(0, y1)
    box_x2 = box_x1 + text_w + 8
    box_y2 = box_y1 + text_h + baseline + 8

    cv2.rectangle(display, (box_x1, box_y1), (box_x2, box_y2), color, cv2.FILLED)
    cv2.putText(
        display,
        text,
        (box_x1 + 4, box_y2 - baseline - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return display


def save_detection_image(frame: np.ndarray, label_name: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = label_name.replace(" ", "_")
    output_path = OUTPUT_DIR / f"{safe_label}_{timestamp}.jpg"
    cv2.imwrite(str(output_path), frame)
    return output_path


def draw_help(frame: np.ndarray, lines) -> None:
    y = 25
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28


def fit_for_display(frame: np.ndarray, display_width: int, display_height: int) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    scale = min(display_width / frame_w, display_height / frame_h)
    new_w = max(1, int(frame_w * scale))
    new_h = max(1, int(frame_h * scale))

    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)

    x_offset = (display_width - new_w) // 2
    y_offset = (display_height - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def process_frame(
    frame: np.ndarray,
    model: YOLO,
    min_conf: float,
    pixels_per_mm: float,
    label_name: str,
    height_history_px: deque,
    od_history_px: deque,
):
    result = model.predict(source=frame, conf=min_conf, verbose=False)[0]
    best_box = get_best_detection(result, min_conf)
    display = frame.copy()
    overlay_text = None

    if best_box is not None:
        xyxy = best_box.xyxy[0].cpu().numpy()
        mask = extract_spring_mask(frame, xyxy)
        refined = refine_spring_height(frame, xyxy)
        refined_od = refine_outer_diameter(mask, xyxy)
        height_px = refined["height_px"] if refined is not None else float(xyxy[3] - xyxy[1])

        height_history_px.append(height_px)
        smoothed_height_px = float(np.median(height_history_px))
        height_mm = smoothed_height_px / pixels_per_mm
        if refined_od is not None:
            od_history_px.append(refined_od["diameter_px"])
        elif len(od_history_px) > 0:
            od_history_px.clear()

        outer_diameter_mm = None
        if len(od_history_px) > 0:
            outer_diameter_mm = float(np.median(od_history_px)) / pixels_per_mm

        result = classify_family_type_from_height_od(height_mm, outer_diameter_mm)
        family_text = result["family"] if result["family"] != "Unknown" else label_name
        type_text = result["type"] if result["type"] != "Unknown" else "Unknown"
        overlay_text = f"{family_text} | {type_text.upper()} | H {height_mm:.2f}mm"
        if outer_diameter_mm is not None:
            overlay_text += f" | OD {outer_diameter_mm:.2f}mm"
        display = draw_result(display, xyxy, overlay_text)

        if refined is not None:
            cv2.line(
                display,
                (int(xyxy[0]), refined["top_y"]),
                (int(xyxy[2]), refined["top_y"]),
                (0, 255, 0),
                2,
            )
            cv2.line(
                display,
                (int(xyxy[0]), refined["bottom_y"]),
                (int(xyxy[2]), refined["bottom_y"]),
                (0, 255, 0),
                2,
            )
        if refined_od is not None:
            cv2.line(
                display,
                (refined_od["left_x"], refined_od["line_y"]),
                (refined_od["right_x"], refined_od["line_y"]),
                (255, 255, 0),
                2,
            )
        alerts = ",".join(result["alerts"]) if result["alerts"] else "clean"
        cv2.putText(
            display,
            f"Rules: {family_text} | {type_text} | conf {result['confidence']:.2f} | {alerts}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        height_history_px.clear()
        od_history_px.clear()
        cv2.putText(
            display,
            "No spring detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return display, overlay_text, best_box


def warn_if_resolution_mismatch(frame: np.ndarray, calibration: dict) -> None:
    calib_w = calibration.get("frame_width")
    calib_h = calibration.get("frame_height")
    if calib_w is None or calib_h is None:
        return

    frame_h, frame_w = frame.shape[:2]
    if frame_w != calib_w or frame_h != calib_h:
        print(
            "Warning: frame resolution "
            f"{frame_w}x{frame_h} does not match calibration resolution {calib_w}x{calib_h}. "
            "Height in mm may be inaccurate."
        )


def run_calibration(
    model: YOLO,
    cap: cv2.VideoCapture,
    min_conf: float,
    reference_mm: float,
    display_width: int,
    display_height: int,
) -> None:
    print("Calibration mode started.")
    print("Place a known-height object where the spring will stand.")
    print("Press 'c' to save calibration.")
    print("Press 'q' to quit.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, display_width, display_height)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Unable to read frame from camera.")
            break

        result = model.predict(source=frame, conf=min_conf, verbose=False)[0]
        best_box = get_best_detection(result, min_conf)
        display = frame.copy()

        draw_help(
            display,
            [
                f"Reference height: {reference_mm:.2f} mm",
                "Place reference object at spring position",
                "Press c to save calibration",
                "Press q to quit",
            ],
        )

        if best_box is not None:
            xyxy = best_box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy.astype(int)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            refined = refine_spring_height(frame, xyxy)
            height_px = refined["height_px"] if refined is not None else float(y2 - y1)

            cv2.putText(
                display,
                f"Measured: {height_px:.1f} px",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                display,
                "No detected object for calibration",
                (10, display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(WINDOW_NAME, fit_for_display(display, display_width, display_height))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c") and best_box is not None:
            xyxy = best_box.xyxy[0].cpu().numpy()
            refined = refine_spring_height(frame, xyxy)
            height_px = refined["height_px"] if refined is not None else float(xyxy[3] - xyxy[1])

            if height_px <= 0:
                print("Calibration failed because measured height was zero.")
                continue

            pixels_per_mm = height_px / reference_mm
            frame_height, frame_width = frame.shape[:2]
            save_calibration(reference_mm, pixels_per_mm, frame_width, frame_height)
            print(f"Calibration saved: {CALIBRATION_PATH}")
            print(f"Reference height: {reference_mm:.2f} mm")
            print(f"Measured pixels: {height_px:.2f} px")
            print(f"Pixels per mm: {pixels_per_mm:.4f}")
            print(f"Calibration frame size: {frame_width}x{frame_height}")
            break


def run_single_image(
    model: YOLO,
    image_path: Path,
    min_conf: float,
    calibration: dict,
    label_name: str,
    display_width: int,
    display_height: int,
) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    warn_if_resolution_mismatch(frame, calibration)
    height_history_px = deque(maxlen=HISTORY_LENGTH)
    od_history_px = deque(maxlen=HISTORY_LENGTH)
    display, overlay_text, _ = process_frame(
        frame, model, min_conf, float(calibration["pixels_per_mm"]), label_name, height_history_px, od_history_px
    )

    if overlay_text is not None:
        print(overlay_text)
    else:
        print("No spring detected in the image.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, display_width, display_height)
    cv2.imshow(WINDOW_NAME, fit_for_display(display, display_width, display_height))
    cv2.waitKey(0)


def run_stream(
    model: YOLO,
    cap: cv2.VideoCapture,
    min_conf: float,
    calibration: dict,
    label_name: str,
    display_width: int,
    display_height: int,
) -> None:
    print("-" * 60)
    print("Detecting spring...")

    last_printed_text = None
    height_history_px = deque(maxlen=HISTORY_LENGTH)
    od_history_px = deque(maxlen=HISTORY_LENGTH)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, display_width, display_height)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Reached end of source or unable to read frame.")
            break

        if last_printed_text is None and len(height_history_px) == 0:
            warn_if_resolution_mismatch(frame, calibration)
        display, overlay_text, best_box = process_frame(
            frame, model, min_conf, float(calibration["pixels_per_mm"]), label_name, height_history_px, od_history_px
        )

        if overlay_text is not None:
            if overlay_text != last_printed_text:
                print(overlay_text)
                last_printed_text = overlay_text

        cv2.imshow(WINDOW_NAME, fit_for_display(display, display_width, display_height))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s") and best_box is not None:
            saved_path = save_detection_image(display, label_name)
            print(f"Saved: {saved_path}")
            print("-" * 60)
            print("Detecting spring...")

def main() -> None:
    args = parse_args()
    display_width, display_height = parse_resolution(args.resolution)
    model = load_model()
    source_kind, source_value = parse_source(args.source)

    cap = None
    try:
        if args.calibrate:
            if source_kind != "camera":
                raise ValueError("Calibration mode supports webcam only. Use --source camera:0")
            cap = open_capture(source_kind, source_value)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
            run_calibration(model, cap, args.conf, args.reference_mm, display_width, display_height)
            return

        calibration = load_calibration()

        if source_kind == "image":
            run_single_image(
                model,
                Path(source_value),
                args.conf,
                calibration,
                args.label,
                display_width,
                display_height,
            )
            return

        cap = open_capture(source_kind, source_value)
        if source_kind == "camera":
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

        run_stream(
            model,
            cap,
            args.conf,
            calibration,
            args.label,
            display_width,
            display_height,
        )
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
