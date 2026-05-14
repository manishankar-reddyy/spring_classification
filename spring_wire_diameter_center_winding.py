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
WINDOW_NAME = "Center Winding Wire Diameter"
HISTORY_LENGTH = 10

SIDE_RANGES = {
    "left": (0.35, 0.62),
    "right": (0.68, 0.86),
}
PROBE_WIDTH_PX = 5
ROW_FILL_THRESHOLD = 0.60
MIN_WIRE_RUN_PX = 5
MAX_WIRE_RUN_RATIO = 0.24
MIN_ROW_FILL_RATIO = 0.18
MIN_OD_ROW_FILL_RATIO = 0.25
OD_WIDTH_PERCENTILE = 90
CENTER_WINDOW_RATIO = 0.22
EDGE_AVOID_RATIO = 0.04
TARGET_ROW_FILL_RATIO = 0.22
TARGET_ROW_WIDTH_PERCENTILE = 90
WINDING_SEARCH_Y_RANGE = (0.22, 0.70)
TARGET_Y_RATIO_BY_SIDE = {
    "right": 0.36,
    "left": 0.58,
}
TARGET_HEIGHT_PERCENTILE_BY_SIDE = {
    "right": 75,
    "left": 55,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure only spring wire diameter from the height of the center winding body."
    )
    parser.add_argument(
        "--source",
        default="camera:0",
        help="Use 'camera:0' for webcam, or provide an image/video path. Default: camera:0",
    )
    parser.add_argument("--conf", type=float, default=0.75, help="YOLO confidence threshold. Default: 0.75")
    parser.add_argument(
        "--side",
        choices=("auto", "left", "right"),
        default="auto",
        help="Side override. Auto uses OD/height: outer=right, inner/snubber=left. Default: auto",
    )
    parser.add_argument(
        "--resolution",
        default="1280x720",
        help="Display resolution in WxH format. Default: 1280x720",
    )
    return parser.parse_args()


def parse_resolution(resolution: str) -> tuple[int, int]:
    width_str, height_str = resolution.lower().split("x")
    return int(width_str), int(height_str)


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


def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    print("Loading model...")
    model = YOLO(str(MODEL_PATH), task="detect")
    print(f"Classes: {model.names}")
    return model


def load_calibration() -> dict:
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {CALIBRATION_PATH}. Run calibration in one of the main scripts first."
        )

    data = json.loads(CALIBRATION_PATH.read_text())
    pixels_per_mm = float(data["pixels_per_mm"])
    if pixels_per_mm <= 0:
        raise ValueError("Invalid pixels_per_mm value in calibration file.")
    return data


def open_capture(source_kind: str, source_value: str | int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source_value)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {source_kind} source: {source_value}")
    print(f"{source_kind.capitalize()} ready")
    return cap


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


def save_detection_image(frame: np.ndarray, label_name: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"{label_name}_{timestamp}.jpg"
    cv2.imwrite(str(output_path), frame)
    return output_path


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


def get_green_bbox_fallback(frame: np.ndarray) -> np.ndarray | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([45, 120, 120]), np.array([85, 255, 255]))
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 40 or h < 80:
            continue
        if h / max(1, w) < 1.2:
            continue
        boxes.append((x, y, x + w, y + h, w * h))
    if not boxes:
        return None
    x1, y1, x2, y2, _ = max(boxes, key=lambda item: item[4])
    return np.array([x1, y1, x2, y2], dtype=float)


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

    open_kernel = np.ones((3, 3), np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask[y1:y2, x1:x2] = roi_mask
    return mask


def extract_wire_body_mask(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
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
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    blue_overlay = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 130) & (hsv[:, :, 1] > 70)
    green_overlay = (hsv[:, :, 0] >= 40) & (hsv[:, :, 0] <= 90) & (hsv[:, :, 1] > 70)
    yellow_overlay = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 40) & (hsv[:, :, 1] > 70)
    colored_overlay = blue_overlay | green_overlay | yellow_overlay

    body_mask = ((gray < 125) & (~colored_overlay)).astype(np.uint8) * 255
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask[y1:y2, x1:x2] = body_mask
    return mask


def extract_runs(binary_rows: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(binary_rows):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(binary_rows) - 1))
    return runs


def find_center_winding_row(roi_mask: np.ndarray) -> int | None:
    h, w = roi_mask.shape[:2]
    if h == 0 or w == 0:
        return None

    row_samples = []
    row_fill = np.count_nonzero(roi_mask, axis=1) / max(1, w)
    valid_rows = np.where(row_fill >= TARGET_ROW_FILL_RATIO)[0]
    for row in valid_rows:
        cols = np.where(roi_mask[row] > 0)[0]
        if len(cols) < 2:
            continue
        left = int(cols.min())
        right = int(cols.max())
        width = right - left + 1
        row_samples.append((int(row), left, right, float(width)))

    if not row_samples:
        return None

    widths = np.array([sample[3] for sample in row_samples], dtype=float)
    strong_width = float(np.percentile(widths, TARGET_ROW_WIDTH_PERCENTILE))
    strong_rows = [sample for sample in row_samples if sample[3] >= strong_width - 2.0]
    if not strong_rows:
        strong_rows = row_samples

    center_y = h / 2.0
    best_row = min(strong_rows, key=lambda sample: (abs(sample[0] - center_y), -sample[3]))
    return int(best_row[0])


def refine_spring_height(mask: np.ndarray, xyxy: np.ndarray) -> dict | None:
    x1, y1, x2, y2 = xyxy.astype(int)
    h, w = mask.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    roi_mask = mask[y1:y2, x1:x2]
    row_fill = np.count_nonzero(roi_mask, axis=1) / max(1, roi_mask.shape[1])
    valid_rows = np.where(row_fill > MIN_ROW_FILL_RATIO)[0]
    if len(valid_rows) == 0:
        return None

    top_local = int(valid_rows[0])
    bottom_local = int(valid_rows[-1])
    height_px = float(bottom_local - top_local + 1)
    if height_px < 20:
        return None

    return {
        "height_px": height_px,
        "top_y": y1 + top_local,
        "bottom_y": y1 + bottom_local,
    }


def refine_outer_diameter(mask: np.ndarray, xyxy: np.ndarray) -> dict | None:
    x1, y1, x2, y2 = xyxy.astype(int)
    h, w = mask.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    roi_mask = mask[y1:y2, x1:x2]
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    od_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    od_mask = cv2.morphologyEx(od_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    row_fill = np.count_nonzero(od_mask, axis=1) / max(1, od_mask.shape[1])
    valid_rows = np.where(row_fill > MIN_OD_ROW_FILL_RATIO)[0]
    row_samples = []
    for row_idx in valid_rows:
        cols = np.where(od_mask[row_idx] > 0)[0]
        if len(cols) < 2:
            continue
        left = int(cols.min())
        right = int(cols.max())
        row_samples.append((int(row_idx), left, right, float(right - left + 1)))

    if not row_samples:
        return None

    widths = np.array([sample[3] for sample in row_samples], dtype=float)
    diameter_px = float(np.percentile(widths, OD_WIDTH_PERCENTILE))
    if diameter_px < 20:
        return None

    mid_row = float(np.median(valid_rows))
    candidate_rows = [sample for sample in row_samples if sample[3] >= diameter_px - 2.0]
    if not candidate_rows:
        candidate_rows = row_samples
    best_row, left_local, right_local, _ = min(
        candidate_rows,
        key=lambda sample: (abs(sample[0] - mid_row), -sample[3]),
    )

    return {
        "diameter_px": diameter_px,
        "line_y": y1 + int(best_row),
        "left_x": x1 + left_local,
        "right_x": x1 + right_local,
    }


def classify_type_from_od_height(od_mm: float | None, height_mm: float | None) -> str:
    if od_mm is not None and od_mm > 120.0:
        return "OUTER"
    if height_mm is not None and height_mm > 280.0:
        return "SNUBBER"
    return "INNER"


def side_for_type(spring_type: str) -> str:
    return "right" if spring_type == "OUTER" else "left"


def measure_side_center_winding(roi_mask: np.ndarray, side: str) -> dict | None:
    if roi_mask.size == 0 or cv2.countNonZero(roi_mask) == 0:
        return None

    h, w = roi_mask.shape[:2]
    x_low_ratio, x_high_ratio = SIDE_RANGES[side]
    x_start = max(0, int(w * x_low_ratio))
    x_end = min(w - 1, int(w * x_high_ratio))
    if x_end <= x_start:
        return None

    center_y = h / 2.0
    y_search_start = int(h * WINDING_SEARCH_Y_RANGE[0])
    y_search_end = int(h * WINDING_SEARCH_Y_RANGE[1])
    max_run_px = max(10, int(h * MAX_WIRE_RUN_RATIO))
    min_x = int(w * EDGE_AVOID_RATIO)
    max_x = int(w * (1.0 - EDGE_AVOID_RATIO))
    preferred_y = h * TARGET_Y_RATIO_BY_SIDE[side]

    candidates = []
    for probe_x in range(x_start, x_end + 1):
        if probe_x < min_x or probe_x > max_x:
            continue

        x1 = max(0, probe_x - PROBE_WIDTH_PX // 2)
        x2 = min(w, probe_x + PROBE_WIDTH_PX // 2 + 1)
        probe = roi_mask[:, x1:x2]
        if probe.shape[1] < 2:
            continue

        row_fill = np.count_nonzero(probe, axis=1) / max(1, probe.shape[1])
        runs = extract_runs(row_fill >= ROW_FILL_THRESHOLD)
        for start, end in runs:
            run_height = end - start + 1
            if run_height < MIN_WIRE_RUN_PX or run_height > max_run_px:
                continue
            run_center = (start + end) / 2.0
            distance_to_center = abs(run_center - center_y)
            if run_center < y_search_start or run_center > y_search_end:
                continue

            candidates.append(
                {
                    "side": side,
                    "x": probe_x,
                    "start": start,
                    "end": end,
                    "height_px": float(run_height),
                    "center_y": run_center,
                    "distance_to_center": distance_to_center,
                    "distance_to_preferred_y": abs(run_center - preferred_y),
                }
            )

    if not candidates:
        return None

    heights = np.array([c["height_px"] for c in candidates], dtype=float)
    target_height = float(np.percentile(heights, TARGET_HEIGHT_PERCENTILE_BY_SIDE[side]))
    height_tolerance = max(4.0, target_height * 0.20)
    strong = [c for c in candidates if abs(c["height_px"] - target_height) <= height_tolerance]
    if not strong:
        strong = candidates
    selected = min(
        strong,
        key=lambda c: (
            abs(c["height_px"] - target_height),
            c["distance_to_preferred_y"],
            -c["x"] if side == "right" else c["x"],
        ),
    )
    selected["final_height_px"] = selected["height_px"]
    selected["mode"] = "image_detected"
    selected["raw_height_px"] = selected["height_px"]
    return selected


def measure_wire_diameter(mask: np.ndarray, xyxy: np.ndarray, side: str) -> dict | None:
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

    selected = measure_side_center_winding(roi_mask, side)
    if selected is None:
        return None

    return {
        "wire_diameter_px": selected["final_height_px"],
        "raw_wire_diameter_px": selected["raw_height_px"],
        "line_x": x1 + int(selected["x"]),
        "line_top_y": y1 + int(selected["start"]),
        "line_bottom_y": y1 + int(selected["end"]),
        "side": selected["side"],
        "distance_to_center": selected["distance_to_center"],
        "mode": selected["mode"],
    }


def draw_result(frame: np.ndarray, xyxy: np.ndarray, text: str) -> np.ndarray:
    display = frame.copy()
    x1, y1, x2, y2 = xyxy.astype(int)
    color = (0, 255, 0)
    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
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
        0.75,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return display


def process_frame(
    frame: np.ndarray,
    model: YOLO,
    min_conf: float,
    pixels_per_mm: float,
    wire_history_px: deque,
    side_mode: str,
):
    result = model.predict(source=frame, conf=min_conf, verbose=False)[0]
    best_box = get_best_detection(result, min_conf)
    display = frame.copy()
    overlay_text = None

    xyxy = None
    if best_box is not None:
        xyxy = best_box.xyxy[0].cpu().numpy()
    else:
        xyxy = get_green_bbox_fallback(frame)

    if xyxy is None:
        wire_history_px.clear()
        cv2.putText(display, "No spring detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return display, overlay_text, best_box

    spring_mask = extract_spring_mask(frame, xyxy)
    body_mask = extract_wire_body_mask(frame, xyxy)

    height_info = refine_spring_height(spring_mask, xyxy)
    od_info = refine_outer_diameter(spring_mask, xyxy)
    height_mm = height_info["height_px"] / pixels_per_mm if height_info is not None else None
    od_mm = od_info["diameter_px"] / pixels_per_mm if od_info is not None else None
    spring_type = classify_type_from_od_height(od_mm, height_mm)
    selected_side = side_for_type(spring_type) if side_mode == "auto" else side_mode
    measurement = measure_wire_diameter(body_mask, xyxy, selected_side)

    wire_diameter_mm = None
    if measurement is not None:
        wire_history_px.append(measurement["wire_diameter_px"])
        wire_diameter_mm = float(np.median(wire_history_px)) / pixels_per_mm
    else:
        wire_history_px.clear()

    overlay_text = f"{spring_type} | WIRE DIA"
    if wire_diameter_mm is None:
        overlay_text += " | UNAVAILABLE"
    else:
        overlay_text += f" | {wire_diameter_mm:.2f}mm"
    if od_mm is not None:
        overlay_text += f" | OD {od_mm:.1f}mm"
    if height_mm is not None:
        overlay_text += f" | H {height_mm:.1f}mm"

    display = draw_result(display, xyxy, overlay_text)
    if measurement is not None:
        cv2.line(
            display,
            (measurement["line_x"], measurement["line_top_y"]),
            (measurement["line_x"], measurement["line_bottom_y"]),
            (0, 200, 255),
            3,
        )
        cv2.putText(
            display,
            (
                f"side {measurement['side']} | "
                f"raw {measurement['raw_wire_diameter_px'] / pixels_per_mm:.1f}mm | "
                f"type {spring_type} | "
                f"{measurement['mode']}"
            ),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
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
            "Wire diameter in mm may be inaccurate."
        )


def run_single_image(
    model: YOLO,
    image_path: Path,
    min_conf: float,
    calibration: dict,
    display_width: int,
    display_height: int,
    side_mode: str,
) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    warn_if_resolution_mismatch(frame, calibration)
    wire_history_px = deque(maxlen=HISTORY_LENGTH)
    display, overlay_text, _ = process_frame(
        frame,
        model,
        min_conf,
        float(calibration["pixels_per_mm"]),
        wire_history_px,
        side_mode,
    )

    print(overlay_text or "No spring detected in the image.")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, display_width, display_height)
    cv2.imshow(WINDOW_NAME, fit_for_display(display, display_width, display_height))
    cv2.waitKey(0)


def run_stream(
    model: YOLO,
    cap: cv2.VideoCapture,
    min_conf: float,
    calibration: dict,
    display_width: int,
    display_height: int,
    side_mode: str,
) -> None:
    print("-" * 60)
    print("Measuring center-winding wire diameter...")
    print("Use --side right for outer springs, --side left for inner/snubber, or --side auto.")
    last_printed_text = None
    wire_history_px = deque(maxlen=HISTORY_LENGTH)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, display_width, display_height)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Reached end of source or unable to read frame.")
            break

        if last_printed_text is None and len(wire_history_px) == 0:
            warn_if_resolution_mismatch(frame, calibration)

        display, overlay_text, best_box = process_frame(
            frame,
            model,
            min_conf,
            float(calibration["pixels_per_mm"]),
            wire_history_px,
            side_mode,
        )

        if overlay_text is not None and overlay_text != last_printed_text:
            print(overlay_text)
            last_printed_text = overlay_text

        cv2.imshow(WINDOW_NAME, fit_for_display(display, display_width, display_height))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and best_box is not None and overlay_text is not None:
            saved_path = save_detection_image(display, "WIRE_DIA_CENTER")
            print(f"Saved: {saved_path}")


def main() -> None:
    args = parse_args()
    display_width, display_height = parse_resolution(args.resolution)
    model = load_model()
    calibration = load_calibration()
    source_kind, source_value = parse_source(args.source)

    cap = None
    try:
        if source_kind == "image":
            run_single_image(
                model,
                Path(source_value),
                args.conf,
                calibration,
                display_width,
                display_height,
                args.side,
            )
            return

        cap = open_capture(source_kind, source_value)
        if source_kind == "camera":
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

        run_stream(model, cap, args.conf, calibration, display_width, display_height, args.side)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
