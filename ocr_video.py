"""
PRISM — extract on-screen text from videos into newline-separated strings.

Default engine: RapidOCR (ONNX), usually stronger than EasyOCR on scene text.
Optional: PaddleOCR if paddlepaddle + paddleocr are installed.

Frames are sampled at a fixed rate (default ≥ 0.5 FPS) over the whole clip.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np


def _native_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        return 25.0
    return float(fps)


def _frame_step(native_fps: float, target_fps: float) -> int:
    """Read every `step` frames to approximate `target_fps` samples per second."""
    target_fps = max(float(target_fps), 0.05)
    native_fps = max(float(native_fps), 1.0)
    return max(1, int(round(native_fps / target_fps)))


def _normalize_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def dedupe_preserve_order(lines: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in lines:
        t = raw.strip()
        if not t:
            continue
        k = _normalize_key(t)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def _parse_rapid_result(result) -> List[str]:
    if not result:
        return []
    texts: List[str] = []
    for item in result:
        if not item or len(item) < 2:
            continue
        payload = item[1]
        if isinstance(payload, (list, tuple)) and len(payload) >= 1:
            texts.append(str(payload[0]).strip())
        elif isinstance(payload, str):
            texts.append(payload.strip())
    return [t for t in texts if t]


def _parse_paddle_result(result) -> List[str]:
    if not result or result[0] is None:
        return []
    texts: List[str] = []
    for line in result[0]:
        if not line or len(line) < 2:
            continue
        cell = line[1]
        if isinstance(cell, (list, tuple)) and len(cell) >= 1:
            texts.append(str(cell[0]).strip())
    return [t for t in texts if t]


def _make_rapid_engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _make_paddle_engine(lang: str = "en"):
    from paddleocr import PaddleOCR

    return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)


def extract_lines_from_bgr_frame(engine: str, ocr_handle, frame_bgr: np.ndarray) -> List[str]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if engine == "rapid":
        result, _ = ocr_handle(rgb)
        return _parse_rapid_result(result)
    if engine == "paddle":
        result = ocr_handle.ocr(rgb, cls=True)
        return _parse_paddle_result(result)
    raise ValueError(f"Unknown OCR engine: {engine}")


def extract_ocr_from_video(
    video_path: Path,
    *,
    sample_fps: float = 0.5,
    engine: str = "rapid",
    lang: str = "en",
    ocr_handle=None,
) -> str:
    """
    Run OCR on frames sampled at ``sample_fps`` (e.g. 0.5 = one frame every ~2s at 25fps).

    Returns a single string: unique non-empty lines in first-seen order, joined by newlines.
    """
    engine = engine.lower().strip()
    if engine not in {"rapid", "paddle"}:
        raise ValueError("engine must be 'rapid' or 'paddle'")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return ""

    native = _native_fps(cap)
    step = _frame_step(native, sample_fps)

    lazy = ocr_handle is None
    if lazy:
        if engine == "rapid":
            ocr_handle = _make_rapid_engine()
        else:
            ocr_handle = _make_paddle_engine(lang=lang)

    all_lines: List[str] = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            all_lines.extend(extract_lines_from_bgr_frame(engine, ocr_handle, frame))
        i += 1

    cap.release()

    merged = dedupe_preserve_order(all_lines)
    return "\n".join(merged)


def extract_ocr_from_video_with_progress(
    video_path: Path,
    *,
    sample_fps: float = 0.5,
    engine: str = "rapid",
    lang: str = "en",
    ocr_handle=None,
    on_frame: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Same as extract_ocr_from_video but optional callback on_frame(frame_index, total_to_read)."""
    engine = engine.lower().strip()
    if engine not in {"rapid", "paddle"}:
        raise ValueError("engine must be 'rapid' or 'paddle'")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return ""

    native = _native_fps(cap)
    step = _frame_step(native, sample_fps)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    total_pick = (n_frames + step - 1) // step if n_frames > 0 else None

    lazy = ocr_handle is None
    if lazy:
        if engine == "rapid":
            ocr_handle = _make_rapid_engine()
        else:
            ocr_handle = _make_paddle_engine(lang=lang)

    all_lines: List[str] = []
    i = 0
    picked = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            if on_frame:
                on_frame(picked, total_pick if total_pick is not None else -1)
            all_lines.extend(extract_lines_from_bgr_frame(engine, ocr_handle, frame))
            picked += 1
        i += 1

    cap.release()

    merged = dedupe_preserve_order(all_lines)
    return "\n".join(merged)
