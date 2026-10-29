import base64
import math
import os
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectionConfig:
    """블랙박스 모션 및 접촉/충격 감지 설정"""
    # 정규화된 관심 영역 (0.0 ~ 1.0)
    roi_x: float = 0.0
    roi_y: float = 0.05  # 상단 블랙박스 시계/텍스트 자막 영역 기본 제외 가능
    roi_w: float = 1.0
    roi_h: float = 0.90
    # 초당 분석할 프레임 수 (기본 4fps: 1분 영상을 240프레임만 스캔하여 초고속 분석)
    sample_fps: float = 4.0
    # 픽셀 변화 민감도 (1 ~ 100, 높을수록 작은 움직임도 감지)
    sensitivity: int = 65
    # 최소 이벤트 병합 간격(초) - 이 시간 이내에 다시 움직이면 하나의 장면으로 묶음
    merge_gap_sec: float = 2.5
    # 최소 움직임 면적 비율 (ROI 대비 %)
    min_area_ratio: float = 0.35
    # 차량 미세 흔들림(접촉/충격) 감지 활성화 여부
    detect_shake: bool = True


@dataclass
class DetectedEvent:
    """감지된 단일 의심 장면 정보"""
    event_id: str
    file_name: str
    file_index: int
    start_time: float
    end_time: float
    peak_time: float
    duration: float
    score: float
    area_pct: float
    shake_score: float
    is_impact_suspected: bool
    category: str  # "impact" | "close_motion" | "motion"
    label: str     # "충격/접촉 의심" | "근접 큰 움직임" | "움직임 감지"
    bbox: Dict[str, float]  # 전체 화면 기준 정규화 좌표 {x, y, w, h}
    thumbnail_data_url: str


def _encode_thumbnail(
    frame_bgr: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
    bboxes_in_roi: List[Tuple[int, int, int, int]],
    is_impact: bool,
    peak_time: float,
) -> str:
    """감지된 순간의 프레임에 ROI 및 움직임 박스를 그려 JPEG Data URL로 반환"""
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]
    rx, ry, rw, rh = roi_rect

    # 1. ROI 영역 테두리 표시 (청록색 점선 느낌의 얇은 테두리)
    if rx > 2 or ry > 2 or rw < w - 4 or rh < h - 4:
        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (255, 200, 0), 2)

    # 2. 움직임 바운딩 박스 표시
    box_color = (0, 60, 255) if is_impact else (0, 215, 255)
    for (bx, by, bw, bh) in bboxes_in_roi:
        gx, gy = rx + bx, ry + by
        cv2.rectangle(vis, (gx, gy), (gx + bw, gy + bh), box_color, 2)

    # 3. 충격 의심 시 화면 외곽 경고 테두리 표시
    if is_impact:
        cv2.rectangle(vis, (2, 2), (w - 3, h - 3), (0, 40, 255), 4)

    # 썸네일 리사이즈 (가로 480px 기준)
    target_w = 480
    if w > target_w:
        scale = target_w / float(w)
        vis = cv2.resize(vis, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def analyze_video_file(
    video_path: str,
    file_name: str,
    file_index: int,
    config: DetectionConfig,
    progress_callback: Optional[Callable[[float, int], None]] = None,
) -> Tuple[List[DetectedEvent], float]:
    """
    단일 블랙박스 동영상 파일을 스캔하여 움직임/접촉 의심 구간 리스트와 총 재생시간(초)을 반환합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or math.isnan(fps):
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / fps if total_frames > 0 else 0.0

    frame_step = max(1, int(round(fps / max(1.0, config.sample_fps))))

    # 민감도(1~100)에 따른 픽셀 차이 임계값 계산 (민감도 65 -> diff threshold 약 22)
    diff_thresh = int(np.clip(55 - (config.sensitivity * 0.48), 8, 50))
    min_area_pct = max(0.08, config.min_area_ratio * ((110 - config.sensitivity) / 50.0))

    prev_gray_roi: Optional[np.ndarray] = None
    prev_full_small: Optional[np.ndarray] = None
    bg_avg_roi: Optional[np.ndarray] = None

    # 활성 이벤트 누적 상태
    raw_segments: List[Dict] = []
    current_seg: Optional[Dict] = None

    frame_idx = 0
    processed_steps = 0

    # 분석용 축소 해상도 기준 (가로 최대 640px로 정규화하여 초고속 연산)
    analysis_width = 640

    while True:
        # frame_step 만큼 건너뛰며 읽기
        if frame_idx > 0 and frame_step > 1:
            for _ in range(frame_step - 1):
                if not cap.grab():
                    break
                frame_idx += 1

        ret, frame = cap.read()
        if not ret or frame is None:
            break

        timestamp = frame_idx / fps
        frame_idx += 1
        processed_steps += 1

        orig_h, orig_w = frame.shape[:2]
        scale = min(1.0, analysis_width / float(orig_w))
        if scale < 1.0:
            small = cv2.resize(
                frame,
                (int(orig_w * scale), int(orig_h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = frame

        sh, sw = small.shape[:2]

        # ROI 픽셀 좌표 계산
        rx = int(np.clip(config.roi_x * sw, 0, sw - 10))
        ry = int(np.clip(config.roi_y * sh, 0, sh - 10))
        rw = int(np.clip(config.roi_w * sw, 10, sw - rx))
        rh = int(np.clip(config.roi_h * sh, 10, sh - ry))

        gray_full = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # 블랙박스 상/하단 타임스탬프(시계 초 단위 깜빡임) 노이즈 억제를 위해 가우시안 블러 적용
        gray_full_blur = cv2.GaussianBlur(gray_full, (5, 5), 0)
        gray_roi = gray_full_blur[ry : ry + rh, rx : rx + rw]

        if prev_gray_roi is None or bg_avg_roi is None:
            prev_gray_roi = gray_roi
            prev_full_small = gray_full_blur
            bg_avg_roi = gray_roi.astype("float32")
            continue

        # 1. 적응형 배경 모델과의 차이 + 직전 프레임과의 차이 결합
        cv2.accumulateWeighted(gray_roi, bg_avg_roi, 0.12)
        bg_diff = cv2.absdiff(gray_roi, cv2.convertScaleAbs(bg_avg_roi))
        frame_diff = cv2.absdiff(gray_roi, prev_gray_roi)
        combined_diff = cv2.max(bg_diff, frame_diff)

        _, thresh = cv2.threshold(combined_diff, diff_thresh, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # 2. 전체 화면 미세 흔들림(충격/접촉) 점수 계산
        shake_score = 0.0
        if config.detect_shake and prev_full_small is not None:
            # 전체 화면의 급격한 엣지 이동 감지
            global_diff = cv2.absdiff(gray_full_blur, prev_full_small)
            global_active_ratio = float(np.count_nonzero(global_diff > (diff_thresh + 6))) / float(sw * sh)
            # 화면 전반(35% 이상)이 순간적으로 동시에 흔들렸다면 차량 충격/접촉 또는 급격한 진동 의심
            if 0.18 <= global_active_ratio <= 0.85:
                shake_score = min(100.0, global_active_ratio * 180.0)

        prev_gray_roi = gray_roi
        prev_full_small = gray_full_blur

        # 3. 윤곽선(Contour) 추출 및 유효 움직임 박스 계산
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = float(rw * rh)
        min_contour_area = max(24.0, roi_area * (min_area_pct / 100.0) * 0.25)

        valid_boxes_small: List[Tuple[int, int, int, int]] = []
        active_area_sum = 0.0
        largest_box = (0, 0, 0, 0)
        largest_box_area = 0.0

        for cnt in contours:
            c_area = cv2.contourArea(cnt)
            if c_area < min_contour_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # 너무 얇고 작은 시계 숫자 변화 같은 영역 무시
            if bh < 6 or bw < 6:
                continue
            valid_boxes_small.append((bx, by, bw, bh))
            active_area_sum += c_area
            if c_area > largest_box_area:
                largest_box_area = c_area
                largest_box = (bx, by, bw, bh)

        area_pct = (active_area_sum / roi_area) * 100.0 if roi_area > 0 else 0.0
        is_motion = (area_pct >= min_area_pct and len(valid_boxes_small) > 0) or (shake_score >= 40.0)

        # 종합 의심 점수 (0 ~ 100)
        motion_score = min(100.0, (area_pct / max(0.2, min_area_pct)) * 22.0 + shake_score * 0.65)

        if is_motion:
            # 원본 프레임 기준 좌표로 환산
            inv_scale = 1.0 / scale
            orig_roi_rect = (
                int(rx * inv_scale),
                int(ry * inv_scale),
                int(rw * inv_scale),
                int(rh * inv_scale),
            )
            orig_boxes = [
                (
                    int(bx * inv_scale),
                    int(by * inv_scale),
                    int(bw * inv_scale),
                    int(bh * inv_scale),
                )
                for (bx, by, bw, bh) in valid_boxes_small
            ]

            # 대표 바운딩 박스 (정규화 0~1)
            if largest_box_area > 0:
                lbx, lby, lbw, lbh = largest_box
                norm_bbox = {
                    "x": round((rx + lbx) / float(sw), 4),
                    "y": round((ry + lby) / float(sh), 4),
                    "w": round(lbw / float(sw), 4),
                    "h": round(lbh / float(sh), 4),
                }
            else:
                norm_bbox = {
                    "x": round(config.roi_x, 4),
                    "y": round(config.roi_y, 4),
                    "w": round(config.roi_w, 4),
                    "h": round(config.roi_h, 4),
                }

            if current_seg is None:
                current_seg = {
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "peak_time": timestamp,
                    "max_score": motion_score,
                    "max_area_pct": area_pct,
                    "max_shake": shake_score,
                    "peak_frame": frame.copy(),
                    "roi_rect": orig_roi_rect,
                    "peak_boxes": orig_boxes,
                    "norm_bbox": norm_bbox,
                }
            else:
                current_seg["end_time"] = timestamp
                current_seg["max_shake"] = max(current_seg["max_shake"], shake_score)
                if motion_score >= current_seg["max_score"]:
                    current_seg["max_score"] = motion_score
                    current_seg["max_area_pct"] = max(current_seg["max_area_pct"], area_pct)
                    current_seg["peak_time"] = timestamp
                    current_seg["peak_frame"] = frame.copy()
                    current_seg["roi_rect"] = orig_roi_rect
                    current_seg["peak_boxes"] = orig_boxes
                    current_seg["norm_bbox"] = norm_bbox
        else:
            if current_seg is not None:
                if (timestamp - current_seg["end_time"]) > config.merge_gap_sec:
                    raw_segments.append(current_seg)
                    current_seg = None

        if progress_callback and processed_steps % 12 == 0 and total_frames > 0:
            pct = min(99.0, (frame_idx / float(total_frames)) * 100.0)
            progress_callback(pct, len(raw_segments) + (1 if current_seg else 0))

    if current_seg is not None:
        raw_segments.append(current_seg)

    cap.release()

    # 세그먼트를 최종 DetectedEvent 목록으로 변환
    events: List[DetectedEvent] = []
    for idx, seg in enumerate(raw_segments):
        score = round(min(99.0, max(15.0, seg["max_score"])), 1)
        area_pct = round(seg["max_area_pct"], 2)
        shake = round(seg["max_shake"], 1)
        is_impact = bool(shake >= 38.0 or area_pct >= 14.0)

        if shake >= 38.0:
            category = "impact"
            label = "충격/접촉 의심 (화면 흔들림)"
        elif area_pct >= 8.0 or score >= 72.0:
            category = "close_motion"
            label = "근접 큰 움직임 (밀착 통행)"
        else:
            category = "motion"
            label = "주변 움직임 감지"

        thumb_url = _encode_thumbnail(
            seg["peak_frame"],
            seg["roi_rect"],
            seg["peak_boxes"],
            is_impact=is_impact,
            peak_time=seg["peak_time"],
        )

        # 전후 여유 시간 부여
        start_t = max(0.0, round(seg["start_time"] - 1.0, 2))
        end_t = round(min(duration_sec if duration_sec > 0 else seg["end_time"] + 1.0, seg["end_time"] + 1.0), 2)
        dur = round(max(0.5, end_t - start_t), 2)

        events.append(
            DetectedEvent(
                event_id=f"ev_{file_index}_{idx + 1}",
                file_name=file_name,
                file_index=file_index,
                start_time=start_t,
                end_time=end_t,
                peak_time=round(seg["peak_time"], 2),
                duration=dur,
                score=score,
                area_pct=area_pct,
                shake_score=shake,
                is_impact_suspected=is_impact,
                category=category,
                label=label,
                bbox=seg["norm_bbox"],
                thumbnail_data_url=thumb_url,
            )
        )

    if progress_callback:
        progress_callback(100.0, len(events))

    return events, round(duration_sec, 2)


def event_to_dict(ev: DetectedEvent) -> Dict:
    return asdict(ev)
