import mimetypes
import os
import shutil
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from detector import DetectionConfig, analyze_video_file, event_to_dict

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMP_DIR = BASE_DIR / "temp_jobs"
TEMP_DIR.mkdir(exist_ok=True)

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

app = FastAPI(
    title="블랙박스 범인 잡기 API",
    description="다수의 블랙박스 영상에서 움직임 및 차량 접촉/충격 의심 장면을 자동 추출하는 백엔드 서버",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 메모리 내 작업 상태 저장소
JOBS: Dict[str, Dict] = {}
JOBS_LOCK = threading.Lock()


class FolderAnalyzeRequest(BaseModel):
    folder_path: str
    roi_x: float = 0.0
    roi_y: float = 0.05
    roi_w: float = 1.0
    roi_h: float = 0.90
    sample_fps: float = 4.0
    sensitivity: int = 65
    merge_gap_sec: float = 2.5
    detect_shake: bool = True


def _run_batch_job(job_id: str, file_entries: List[Dict], config: DetectionConfig):
    """백그라운드 스레드에서 블랙박스 영상들을 순차 분석하며 실시간으로 이벤트를 추가합니다."""
    total_files = len(file_entries)
    start_ts = time.time()

    for idx, entry in enumerate(file_entries):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job or job.get("status") == "cancelled":
                return
            job["current_file_index"] = idx
            job["current_file_name"] = entry["name"]
            job["current_file_progress"] = 0.0

        def on_progress(pct: float, found_in_file: int):
            with JOBS_LOCK:
                j = JOBS.get(job_id)
                if not j:
                    return
                j["current_file_progress"] = round(pct, 1)
                overall = ((idx + (pct / 100.0)) / max(1, total_files)) * 100.0
                j["overall_progress"] = round(min(99.5, overall), 1)

        try:
            events, duration_sec = analyze_video_file(
                video_path=entry["path"],
                file_name=entry["name"],
                file_index=idx,
                config=config,
                progress_callback=on_progress,
            )
            event_dicts = [event_to_dict(ev) for ev in events]
            for ev_d in event_dicts:
                ev_d["video_url"] = f"/api/video/{job_id}/{idx}"

            with JOBS_LOCK:
                j = JOBS.get(job_id)
                if not j or j.get("status") == "cancelled":
                    return
                j["files"][idx]["duration"] = duration_sec
                j["files"][idx]["event_count"] = len(event_dicts)
                j["files"][idx]["analyzed"] = True
                j["events"].extend(event_dicts)
                j["completed_files"] = idx + 1
                j["overall_progress"] = round(((idx + 1) / max(1, total_files)) * 100.0, 1)
        except Exception as exc:
            with JOBS_LOCK:
                j = JOBS.get(job_id)
                if j:
                    j["errors"].append(f"{entry['name']}: {str(exc)}")

    with JOBS_LOCK:
        j = JOBS.get(job_id)
        if j and j.get("status") != "cancelled":
            j["status"] = "completed"
            j["overall_progress"] = 100.0
            j["current_file_progress"] = 100.0
            j["elapsed_sec"] = round(time.time() - start_ts, 1)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "engine": "python-opencv",
        "message": "OpenCV 로컬 정밀 분석 엔진이 활성화되어 있습니다.",
    }


@app.post("/api/analyze-upload")
async def analyze_uploaded_files(
    files: List[UploadFile] = File(...),
    roi_x: float = Form(0.0),
    roi_y: float = Form(0.05),
    roi_w: float = Form(1.0),
    roi_h: float = Form(0.90),
    sample_fps: float = Form(4.0),
    sensitivity: int = Form(65),
    merge_gap_sec: float = Form(2.5),
    detect_shake: bool = Form(True),
):
    if not files:
        raise HTTPException(status_code=400, detail="분석할 영상 파일이 없습니다.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    file_entries = []
    for idx, up in enumerate(files):
        safe_name = os.path.basename(up.filename or f"video_{idx}.mp4")
        dest_path = job_dir / f"{idx:04d}_{safe_name}"
        with open(dest_path, "wb") as out_f:
            shutil.copyfileobj(up.file, out_f)
        file_entries.append(
            {
                "index": idx,
                "name": safe_name,
                "path": str(dest_path),
                "size": os.path.getsize(dest_path),
                "duration": 0.0,
                "event_count": 0,
                "analyzed": False,
                "video_url": f"/api/video/{job_id}/{idx}",
            }
        )

    config = DetectionConfig(
        roi_x=roi_x,
        roi_y=roi_y,
        roi_w=roi_w,
        roi_h=roi_h,
        sample_fps=sample_fps,
        sensitivity=sensitivity,
        merge_gap_sec=merge_gap_sec,
        detect_shake=detect_shake,
    )

    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "running",
            "total_files": len(file_entries),
            "completed_files": 0,
            "current_file_index": 0,
            "current_file_name": file_entries[0]["name"],
            "current_file_progress": 0.0,
            "overall_progress": 0.0,
            "files": file_entries,
            "events": [],
            "errors": [],
            "elapsed_sec": 0.0,
            "is_temp": True,
        }

    worker = threading.Thread(
        target=_run_batch_job,
        args=(job_id, file_entries, config),
        daemon=True,
    )
    worker.start()

    return {"job_id": job_id, "total_files": len(file_entries)}


@app.post("/api/analyze-folder")
def analyze_local_folder(req: FolderAnalyzeRequest):
    folder = Path(req.folder_path).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"폴더를 찾을 수 없습니다: {req.folder_path}",
        )

    video_paths = sorted(
        [
            p
            for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )
    if not video_paths:
        raise HTTPException(
            status_code=400,
            detail="해당 폴더 내에 지원하는 동영상 파일(MP4, AVI, MOV, MKV)이 없습니다.",
        )

    job_id = uuid.uuid4().hex[:12]
    file_entries = []
    for idx, vp in enumerate(video_paths):
        file_entries.append(
            {
                "index": idx,
                "name": vp.name,
                "path": str(vp),
                "size": vp.stat().st_size,
                "duration": 0.0,
                "event_count": 0,
                "analyzed": False,
                "video_url": f"/api/video/{job_id}/{idx}",
            }
        )

    config = DetectionConfig(
        roi_x=req.roi_x,
        roi_y=req.roi_y,
        roi_w=req.roi_w,
        roi_h=req.roi_h,
        sample_fps=req.sample_fps,
        sensitivity=req.sensitivity,
        merge_gap_sec=req.merge_gap_sec,
        detect_shake=req.detect_shake,
    )

    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "running",
            "total_files": len(file_entries),
            "completed_files": 0,
            "current_file_index": 0,
            "current_file_name": file_entries[0]["name"],
            "current_file_progress": 0.0,
            "overall_progress": 0.0,
            "files": file_entries,
            "events": [],
            "errors": [],
            "elapsed_sec": 0.0,
            "is_temp": False,
        }

    worker = threading.Thread(
        target=_run_batch_job,
        args=(job_id, file_entries, config),
        daemon=True,
    )
    worker.start()

    return {"job_id": job_id, "total_files": len(file_entries)}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
        return job


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
        job["status"] = "cancelled"
    return {"status": "cancelled"}


@app.get("/api/video/{job_id}/{file_index}")
def stream_video(job_id: str, file_index: int, request: Request):
    """브라우저 비디오 플레이어에서 구간 탐색(Seek)이 즉시 가능하도록 HTTP Range 스트리밍을 지원합니다."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or file_index < 0 or file_index >= len(job["files"]):
            raise HTTPException(status_code=404, detail="영상을 찾을 수 없습니다.")
        file_path = job["files"][file_index]["path"]

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="실제 영상 파일이 존재하지 않습니다.")

    file_size = os.path.getsize(file_path)
    content_type, _ = mimetypes.guess_type(file_path)
    if not content_type:
        content_type = "video/mp4"

    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(file_path, media_type=content_type)

    try:
        byte_range = range_header.strip().lower().replace("bytes=", "")
        start_str, end_str = byte_range.split("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
    except ValueError:
        return FileResponse(file_path, media_type=content_type)

    start = max(0, start)
    end = min(file_size - 1, end)
    chunk_size = (end - start) + 1

    def iterfile():
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = chunk_size
            while remaining > 0:
                read_len = min(1024 * 512, remaining)
                data = f.read(read_len)
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
    }
    return StreamingResponse(
        iterfile(),
        status_code=206,
        media_type=content_type,
        headers=headers,
    )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"
    print("=" * 62)
    print(" 🚗 블랙박스 범인 잡기 (Dashcam Culprit Finder) 서버 시작")
    print(f" 🌐 브라우저 접속 주소: {url}")
    print("=" * 62)
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="info")
