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
WINDOW_NAME = "Bogie Colour + Height + OD + WD + Type"
HISTORY_LENGTH = 15
MIN_ROW_FILL_RATIO = 0.18
MIN_OD_ROW_FILL_RATIO = 0.25
OD_WIDTH_PERCENTILE = 90
WIRE_PROBE_WIDTH = 5
MIN_WIRE_RUN_PX = 5
MAX_WIRE_RUN_RATIO = 0.24
WIRE_ROW_FILL_THRESHOLD = 0.6
WIRE_SIDE_RANGES = {
    "left": (0.35, 0.62),
    "right": (0.68, 0.86),
}
WIRE_CONTOUR_SIDE_RANGES = {
    "left": (0.02, 0.38),
    "right": (0.62, 0.98),
}
WIRE_WINDING_SEARCH_Y_RANGE = (0.22, 0.70)
WIRE_TARGET_Y_RATIO_BY_SIDE = {
    "right": 0.36,
    "left": 0.58,
}
WIRE_TARGET_HEIGHT_PERCENTILE_BY_SIDE = {
    "right": 75,
    "left": 55,
}
MIN_REASONABLE_WD_MM = 8.0

COLOUR_RULES = [
    ((90, 115, 60, 200, 80, 230), "CASNUB-22HS Mod-1 BOSTHS"),
    ((0, 30, 0, 60, 60, 140), "CASNUB-22HS"),
    ((0, 180, 0, 40, 0, 70), "CASNUB 22W/22NLB"),
    ((40, 80, 80, 255, 60, 200), "Cast Steel Bogie"),
    ((20, 38, 100, 255, 120, 255), "Cast Steel Bogie"),
]

SAMPLE_TUNED_RULES = [
    ((95, 120, 0, 18, 90, 180), "CASNUB 22W/22NLB", "Black"),
    ((95, 120, 19, 55, 90, 180), "CASNUB-22HS", "Grey"),
    ((95, 120, 56, 170, 90, 180), "CASNUB-22HS Mod-1 BOSTHS", "Blue"),
]

FAMILY_ALIASES = {
    "CASNUB 22W/22NLB": "NL",
    "CASNUB-22HS": "HS",
    "CASNUB-22HS Mod-1 BOSTHS": "Mod-1 BOSTHS",
    "Cast Steel Bogie": "LCCF20",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect spring, classify family by color, and measure height, OD, and wire diameter.")
    parser.add_argument("--source",default="camera:0",help="Use 'camera:0' for webcam, or provide an image/video path. Default: camera:0",)
    parser.add_argument("--conf", type=float, default=0.75, help="Confidence threshold. Default: 0.75")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration mode using webcam only.")
    parser.add_argument("--reference-mm",type=float,default=100.0,help="Known reference object height in millimeters for calibration.",)
    parser.add_argument("--resolution",default="1280x720",help="Display resolution in WxH format. Default: 1280x720",)
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

def open_capture(source_kind: str, source_value: str | int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source_value)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {source_kind} source: {source_value}")
    print(f"{source_kind.capitalize()} ready")
    return cap

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
        raise FileNotFoundError(f"Calibration file not found: {CALIBRATION_PATH}. Run this script with --calibrate first.")

    data = json.loads(CALIBRATION_PATH.read_text())
    pixels_per_mm = float(data["pixels_per_mm"])
    if pixels_per_mm <= 0:
        raise ValueError("Invalid pixels_per_mm value in calibration file.")
    return data

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
    safe_label = label_name.replace(" ", "_").replace("/", "_")
    output_path = OUTPUT_DIR / f"{safe_label}_{timestamp}.jpg"
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

    kernel = np.ones((3, 3), np.uint8)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
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

def classify_colour(mask: np.ndarray, image_bgr: np.ndarray) -> tuple[str, dict | None]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    patch = hsv[mask > 0]

    if len(patch) < 100:
        return "Unknown", None

    clean = patch[(patch[:, 1] > 30) | (patch[:, 2] > 40)]
    if len(clean) < 100:
        return "Unknown", None

    mean_h = float(np.median(clean[:, 0]))
    mean_s = float(np.median(clean[:, 1]))
    mean_v = float(np.median(clean[:, 2]))
    sat_q75 = float(np.percentile(clean[:, 1], 75))
    high_sat_ratio = float(np.mean(clean[:, 1] >= 70))

    family = "Unknown"
    rule_name = "none"

    for (hl, hh, sl, sh, vl, vh), family_name, profile_name in SAMPLE_TUNED_RULES:
        if hl <= mean_h <= hh and sl <= mean_s <= sh and vl <= mean_v <= vh:
            family = family_name
            rule_name = profile_name
            break

    if family == "Unknown":
        for (hl, hh, sl, sh, vl, vh), family_name in COLOUR_RULES:
            if hl <= mean_h <= hh and sl <= mean_s <= sh and vl <= mean_v <= vh:
                family = family_name
                rule_name = "document_rule"
                break

    if family == "Unknown" and 90 <= mean_h <= 120:
        if mean_s <= 18:
            family = "CASNUB 22W/22NLB"
            rule_name = "sat_fallback_black"
        elif mean_s <= 55:
            family = "CASNUB-22HS"
            rule_name = "sat_fallback_grey"
        elif mean_s >= 70:
            family = "CASNUB-22HS Mod-1 BOSTHS"
            rule_name = "sat_fallback_blue"

    if (family == "CASNUB-22HS" and 90 <= mean_h <= 120 and (sat_q75 >= 60 or high_sat_ratio >= 0.15)):
        family = "CASNUB-22HS Mod-1 BOSTHS"
        rule_name = "blue_dominance_override"

    return family, {
        "median_h": mean_h,
        "median_s": mean_s,
        "median_v": mean_v,
        "sat_q75": sat_q75,
        "high_sat_ratio": high_sat_ratio,
        "pixel_count": int(len(clean)),
        "rule_name": rule_name,
    }

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
    candidate_rows = [sample for sample in row_samples if sample[3] >= diameter_px - 2.0]
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

def _extract_runs(binary_rows: np.ndarray) -> list[tuple[int, int]]:
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

def refine_wire_diameter(mask: np.ndarray, xyxy: np.ndarray, spring_type: str):
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

    if cv2.countNonZero(roi_mask) == 0:
        return None

    strip_side = "right" if spring_type == "OUTER" else "left"
    x_range = WIRE_SIDE_RANGES[strip_side]
    probe_width = min(WIRE_PROBE_WIDTH, max(3, roi_mask.shape[1]))
    max_run_px = max(10, int(roi_mask.shape[0] * MAX_WIRE_RUN_RATIO))
    preferred_y = roi_mask.shape[0] * WIRE_TARGET_Y_RATIO_BY_SIDE[strip_side]
    y_search_start = int(roi_mask.shape[0] * WIRE_WINDING_SEARCH_Y_RANGE[0])
    y_search_end = int(roi_mask.shape[0] * WIRE_WINDING_SEARCH_Y_RANGE[1])
    x_start = max(0, int(roi_mask.shape[1] * x_range[0]))
    x_end = min(roi_mask.shape[1] - 1, int(roi_mask.shape[1] * x_range[1]))
    if x_end <= x_start:
        return None

    candidates = []
    for probe_center_x in range(x_start, x_end + 1):
        probe_x1 = max(0, probe_center_x - probe_width // 2)
        probe_x2 = min(roi_mask.shape[1], probe_center_x + probe_width // 2 + 1)
        probe = roi_mask[:, probe_x1:probe_x2]
        if probe.shape[1] < 2:
            continue

        row_fraction = np.count_nonzero(probe, axis=1) / max(1, probe.shape[1])
        binary_rows = row_fraction >= WIRE_ROW_FILL_THRESHOLD
        runs = _extract_runs(binary_rows)

        for start, end in runs:
            run_height = end - start + 1
            if run_height < MIN_WIRE_RUN_PX or run_height > max_run_px:
                continue
            run_center = (start + end) / 2.0
            if run_center < y_search_start or run_center > y_search_end:
                continue
            candidates.append(
                {
                    "probe_center_x": probe_center_x,
                    "start": start,
                    "end": end,
                    "height": float(run_height),
                    "center_y": run_center,
                    "distance_to_preferred_y": abs(run_center - preferred_y),
                }
            )

    if not candidates:
        return None

    heights = np.array([item["height"] for item in candidates], dtype=float)
    target_height = float(np.percentile(heights, WIRE_TARGET_HEIGHT_PERCENTILE_BY_SIDE[strip_side]))
    height_tolerance = max(4.0, target_height * 0.20)
    central_widths = heights[np.abs(heights - target_height) <= height_tolerance]
    if len(central_widths) == 0:
        central_widths = heights
    representative = min(
        candidates,
        key=lambda item: (
            abs(item["height"] - target_height),
            item["distance_to_preferred_y"],
            -item["probe_center_x"] if strip_side == "right" else item["probe_center_x"],
        ),
    )
    wire_diameter_px = float(representative["height"])

    return {
        "wire_diameter_px": wire_diameter_px,
        "peak_count": len(candidates),
        "line_top_y": y1 + int(representative["start"]),
        "line_bottom_y": y1 + int(representative["end"]),
        "line_x": x1 + int(representative["probe_center_x"]),
        "all_widths_px": [float(item["height"]) for item in candidates],
        "central_widths_px": [float(v) for v in central_widths.tolist()],
        "strip_side": strip_side,
        "measurement_method": "image_detected",
    }

def refine_wire_diameter_contour_fallback(mask: np.ndarray, xyxy: np.ndarray, spring_type: str):
    x1, y1, x2, y2 = xyxy.astype(int)
    h, w = mask.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    roi_mask = mask[y1:y2, x1:x2]
    if roi_mask.size == 0 or cv2.countNonZero(roi_mask) == 0:
        return None

    strip_side = "right" if spring_type == "OUTER" else "left"
    x_low_ratio, x_high_ratio = WIRE_CONTOUR_SIDE_RANGES[strip_side]
    side_x1 = int(roi_mask.shape[1] * x_low_ratio)
    side_x2 = int(roi_mask.shape[1] * x_high_ratio)
    y_search_start = int(roi_mask.shape[0] * WIRE_WINDING_SEARCH_Y_RANGE[0])
    y_search_end = int(roi_mask.shape[0] * WIRE_WINDING_SEARCH_Y_RANGE[1])
    max_height_px = max(18, int(roi_mask.shape[0] * 0.12))

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    clean_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if cw < 20 or ch < MIN_WIRE_RUN_PX or ch > max_height_px:
            continue
        center_y = cy + ch / 2.0
        if center_y < y_search_start or center_y > y_search_end:
            continue
        overlap = max(0, min(cx + cw, side_x2) - max(cx, side_x1))
        if overlap < max(6, min(cw, side_x2 - side_x1) * 0.20):
            continue
        area = cv2.contourArea(contour)
        if area < cw * ch * 0.25:
            continue
        candidates.append(
            {
                "x": cx,
                "y": cy,
                "w": cw,
                "h": float(ch),
                "center_y": center_y,
                "overlap": overlap,
                "area": area,
            }
        )

    if not candidates:
        return None

    heights = np.array([item["h"] for item in candidates], dtype=float)
    target_height = float(np.percentile(heights, 55))
    height_tolerance = max(5.0, target_height * 0.25)
    central_widths = heights[np.abs(heights - target_height) <= height_tolerance]
    if len(central_widths) == 0:
        central_widths = heights

    preferred_y = roi_mask.shape[0] * WIRE_TARGET_Y_RATIO_BY_SIDE[strip_side]
    representative = min(
        candidates,
        key=lambda item: (
            abs(item["h"] - target_height),
            abs(item["center_y"] - preferred_y),
            -item["overlap"],
        ),
    )

    line_x = representative["x"] + representative["w"] // 2
    return {
        "wire_diameter_px": float(representative["h"]),
        "peak_count": len(candidates),
        "line_top_y": y1 + int(representative["y"]),
        "line_bottom_y": y1 + int(representative["y"] + representative["h"] - 1),
        "line_x": x1 + int(line_x),
        "all_widths_px": [float(item["h"]) for item in candidates],
        "central_widths_px": [float(v) for v in central_widths.tolist()],
        "strip_side": strip_side,
        "measurement_method": "contour_fallback",
    }

def classify_type_from_od_height(od_mm: float | None, height_mm: float) -> str:
    if od_mm is not None and od_mm > 120.0:
        return "OUTER"
    if height_mm > 280.0:
        return "SNUBBER"
    return "INNER"

def choose_wire_diameter_mm(
    family: str,
    spring_type: str,
    od_mm: float | None,
    image_wire_mm: float | None,
) -> tuple[float | None, str]:
    if image_wire_mm is not None:
        return image_wire_mm, "image_detected"
    return None, "missing"

def draw_result(frame: np.ndarray, xyxy: np.ndarray, text: str) -> np.ndarray:
    display = frame.copy()
    x1, y1, x2, y2 = xyxy.astype(int)
    color = (0, 255, 0)

    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    box_x1 = min(max(0, x2 + 8), max(0, display.shape[1] - text_w - 10))
    box_y1 = max(0, y1)
    box_x2 = box_x1 + text_w + 8
    box_y2 = box_y1 + text_h + baseline + 8

    cv2.rectangle(display, (box_x1, box_y1), (box_x2, box_y2), color, cv2.FILLED)
    cv2.putText(display,text,(box_x1 + 4, box_y2 - baseline - 4),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 0, 0),2,cv2.LINE_AA,)
    return display

def process_frame(
    frame: np.ndarray,
    model: YOLO,
    min_conf: float,
    pixels_per_mm: float,
    height_history_px: deque,
    od_history_px: deque,
    wire_history_px: deque,
):
    result = model.predict(source=frame, conf=min_conf, verbose=False)[0]
    best_box = get_best_detection(result, min_conf)
    display = frame.copy()
    overlay_text = None

    xyxy = best_box.xyxy[0].cpu().numpy() if best_box is not None else get_green_bbox_fallback(frame)

    if xyxy is not None:
        mask = extract_spring_mask(frame, xyxy)
        wire_body_mask = extract_wire_body_mask(frame, xyxy)
        family, hsv_stats = classify_colour(mask, frame)

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

        spring_type = classify_type_from_od_height(outer_diameter_mm, height_mm)
        refined_wd = refine_wire_diameter(wire_body_mask, xyxy, spring_type)
        if refined_wd is not None and refined_wd["wire_diameter_px"] / pixels_per_mm < MIN_REASONABLE_WD_MM:
            contour_wd = refine_wire_diameter_contour_fallback(wire_body_mask, xyxy, spring_type)
            if contour_wd is not None:
                refined_wd = contour_wd
        elif refined_wd is None:
            refined_wd = refine_wire_diameter_contour_fallback(wire_body_mask, xyxy, spring_type)

        if refined_wd is not None:
            wire_history_px.append(refined_wd["wire_diameter_px"])
        elif len(wire_history_px) > 0:
            wire_history_px.clear()

        image_wire_diameter_mm = None
        if len(wire_history_px) > 0:
            image_wire_diameter_mm = float(np.median(wire_history_px)) / pixels_per_mm

        wire_diameter_mm, wire_mode = choose_wire_diameter_mm(
            family,
            spring_type,
            outer_diameter_mm,
            image_wire_diameter_mm,
        )
        overlay_text = f"{family} | {spring_type} | H {height_mm:.2f}mm"
        if outer_diameter_mm is not None:
            overlay_text += f" | OD {outer_diameter_mm:.2f}mm"
        if wire_diameter_mm is not None:
            overlay_text += f" | WD {wire_diameter_mm:.2f}mm"
        display = draw_result(display, xyxy, overlay_text)

        if refined is not None:
            cv2.line(display, (int(xyxy[0]), refined["top_y"]), (int(xyxy[2]), refined["top_y"]), (0, 255, 0), 2)
            cv2.line(display, (int(xyxy[0]), refined["bottom_y"]), (int(xyxy[2]), refined["bottom_y"]), (0, 255, 0), 2)
        if refined_od is not None:
            cv2.line(
                display,
                (refined_od["left_x"], refined_od["line_y"]),
                (refined_od["right_x"], refined_od["line_y"]),
                (255, 255, 0),
                2,
            )
        if refined_wd is not None:
            cv2.line(
                display,
                (refined_wd["line_x"], refined_wd["line_top_y"]),
                (refined_wd["line_x"], refined_wd["line_bottom_y"]),
                (0, 200, 255),
                2,
            )

        if hsv_stats is not None:
            od_text = ""
            wd_text = ""
            if outer_diameter_mm is not None:
                od_text = f" | OD {outer_diameter_mm:.1f}mm"
            if wire_diameter_mm is not None:
                wd_text = f" | WD {wire_diameter_mm:.1f}mm ({wire_mode})"
            cv2.putText(
                display,
                (
                    f"HSV median: {hsv_stats['median_h']:.0f}, {hsv_stats['median_s']:.0f}, "
                    f"{hsv_stats['median_v']:.0f} | pixels: {hsv_stats['pixel_count']} | "
                    f"{hsv_stats['rule_name']} | type {spring_type}{od_text}{wd_text}"
                ),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if refined_wd is not None:
                cv2.putText(
                    display,
                    (
                        f"WD side: {refined_wd['strip_side']} | "
                        f"target widths: {','.join(f'{v:.0f}' for v in refined_wd['central_widths_px'][:8])}"
                    ),
                    (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
    else:
        height_history_px.clear()
        od_history_px.clear()
        wire_history_px.clear()
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

        cv2.putText(display, f"Reference height: {reference_mm:.2f} mm", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "Place reference object at spring position", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "Press c to save calibration | q to quit", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        if best_box is not None:
            xyxy = best_box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy.astype(int)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            refined = refine_spring_height(frame, xyxy)
            height_px = refined["height_px"] if refined is not None else float(y2 - y1)
            cv2.putText(display, f"Measured: {height_px:.1f} px", (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(display, "No detected object for calibration", (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

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
    display_width: int,
    display_height: int,
) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    warn_if_resolution_mismatch(frame, calibration)
    height_history_px = deque(maxlen=HISTORY_LENGTH)
    od_history_px = deque(maxlen=HISTORY_LENGTH)
    wire_history_px = deque(maxlen=HISTORY_LENGTH)
    display, overlay_text, _ = process_frame(
        frame,
        model,
        min_conf,
        float(calibration["pixels_per_mm"]),
        height_history_px,
        od_history_px,
        wire_history_px,
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
    display_width: int,
    display_height: int,
) -> None:
    print("-" * 60)
    print("Detecting spring...")

    last_printed_text = None
    height_history_px = deque(maxlen=HISTORY_LENGTH)
    od_history_px = deque(maxlen=HISTORY_LENGTH)
    wire_history_px = deque(maxlen=HISTORY_LENGTH)

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
            frame,
            model,
            min_conf,
            float(calibration["pixels_per_mm"]),
            height_history_px,
            od_history_px,
            wire_history_px,
        )

        if overlay_text is not None and overlay_text != last_printed_text:
            print(overlay_text)
            last_printed_text = overlay_text

        cv2.imshow(WINDOW_NAME, fit_for_display(display, display_width, display_height))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s") and best_box is not None and overlay_text is not None:
            save_label = overlay_text.split("|")[0].strip()
            saved_path = save_detection_image(display, save_label)
            print(f"Saved: {saved_path}")

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
            run_single_image(model, Path(source_value), args.conf, calibration, display_width, display_height)
            return

        cap = open_capture(source_kind, source_value)
        if source_kind == "camera":
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

        run_stream(model, cap, args.conf, calibration, display_width, display_height)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
