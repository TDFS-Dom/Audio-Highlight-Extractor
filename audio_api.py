from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
import librosa
import numpy as np
import shutil
import tempfile
import os
import time
import json
from datetime import datetime
from pydub import AudioSegment
import asyncio
import logging
import unicodedata
import re
from scipy import signal
from scipy.signal import find_peaks
from utils_audio import (process_audio_file)

from utils_sql import (
    write_log, save_job, update_job_status, save_highlight,
    get_processing_jobs, get_job_highlights, get_job_details
)
from schema import init_db

# Cấu hình API với metadata cho documentation
app = FastAPI(
    title="Audio Highlight API",
    description="API xử lý và phân tích audio để tìm các highlight trong bài hát",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("audio_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("audio_processor")

# Gọi hàm khởi tạo DB khi ứng dụng khởi động
init_db()


@app.post("/highlight", 
    summary="Tạo highlight cho một file audio",
    description="Xử lý file audio và tạo các đoạn highlight dựa trên phân tích âm nhạc",
    response_description="Thông tin về các highlight được tạo")
async def get_highlight(
    audio_file: UploadFile = File(..., description="File audio cần xử lý (MP3, WAV)"),
    segment_duration: float = 40.0, 
    num_segments: int = 2
):
    # Tạo thư mục output nếu chưa tồn tại
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
        shutil.copyfileobj(audio_file.file, temp_audio)
        audio_path = temp_audio.name

    # Xử lý file audio - truyền num_segments để xác định số lượng chorus cần lấy
    # Số lượng chorus = num_segments - 1 (vì luôn lấy 1 main)
    result = await process_audio_file(audio_path, output_dir, audio_file.filename, num_segments, segment_duration)
    
    # Xóa file tạm sau khi xử lý xong
    try:
        os.unlink(audio_path)
    except:
        pass
        
    return result

# Endpoint mới để xử lý nhiều file cùng lúc
@app.post("/highlight_multiple",
    summary="Tạo highlight cho nhiều file audio",
    description="Xử lý nhiều file audio cùng lúc và tạo các đoạn highlight cho từng file",
    response_description="Danh sách kết quả highlight cho mỗi file")
async def get_multiple_highlights(
    audio_files: list[UploadFile] = File(..., description="Danh sách các file audio cần xử lý"),
    segment_duration: float = 40.0, 
    num_segments: int = 2
):
    # Tạo thư mục output nếu chưa tồn tại
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for audio_file in audio_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            shutil.copyfileobj(audio_file.file, temp_audio)
            audio_path = temp_audio.name
        
        # Truyền num_segments để xác định số lượng chorus cần lấy
        # Số lượng chorus = num_segments - 1 (vì luôn lấy 1 main)
        result = await process_audio_file(audio_path, output_dir, audio_file.filename, num_segments, segment_duration)
        
        # Không cần gán type thủ công nữa vì process_audio_file đã xử lý
        
        results.append({
            "filename": audio_file.filename,
            "result": result
        })
        
        # Xóa file tạm sau khi xử lý xong
        try:
            os.unlink(audio_path)
        except:
            pass
    
    return {"results": results}

# Endpoint để download file highlight audio
@app.get("/download_highlight",
    summary="Tải file highlight audio",
    description="Tải về file audio highlight đã được tạo trước đó",
    response_class=FileResponse,
    response_description="File audio highlight")
async def download_highlight(file_path: str):
    if not os.path.exists(file_path):
        return {"error": "File không tồn tại"}
    
    # Lấy tên file từ đường dẫn
    filename = os.path.basename(file_path)
    
    # Xác định media_type dựa trên phần mở rộng của file
    if filename.endswith('.mp3'):
        media_type = "audio/mpeg"
    elif filename.endswith('.wav'):
        media_type = "audio/wav"
    else:
        media_type = "audio/mpeg"  # Mặc định là MP3
    
    # Xử lý tên file để tránh lỗi Unicode khi encode sang latin-1
    # Thay thế các ký tự Unicode không hợp lệ bằng ASCII
    filename_ascii = unicodedata.normalize('NFKD', filename)
    filename_ascii = re.sub(r'[^\x00-\x7F]+', '_', filename_ascii)
    
    return FileResponse(
        path=file_path, 
        media_type=media_type, 
        filename=filename_ascii,
        headers={"Content-Disposition": f"attachment; filename={filename_ascii}"}
    )

# Endpoint để lấy danh sách các job đã xử lý
@app.get("/jobs",
    summary="Danh sách các job xử lý",
    description="Lấy danh sách tất cả các job xử lý audio đã và đang thực hiện",
    response_description="Danh sách các job")
async def get_jobs():
    jobs = get_processing_jobs()
    result = []
    for job in jobs:
        job_id, file_name, status, start_time, end_time, num_segments, segment_duration, output_path = job
        result.append({
            "job_id": job_id,
            "file_name": file_name,
            "status": status,
            "start_time": start_time,
            "end_time": end_time,
            "num_segments": num_segments,
            "segment_duration": segment_duration,
            "output_path": output_path
        })
    return {"jobs": result}

# Endpoint để lấy thông tin chi tiết của một job
@app.get("/job/{job_id}",
    summary="Chi tiết job xử lý",
    description="Lấy thông tin chi tiết của một job xử lý cụ thể bao gồm các highlight và log",
    response_description="Chi tiết job và các highlight")
async def get_job(job_id: int):
    job = get_job_details(job_id)
    
    if not job:
        return {"error": "Job không tồn tại"}
    
    job_id, file_name, status, start_time, end_time, num_segments, segment_duration, output_path, error_message = job
    
    # Lấy danh sách highlights
    highlights = get_job_highlights(job_id)
    highlight_list = []
    
    for highlight in highlights:
        highlight_index, highlight_time, highlight_file, score = highlight
        highlight_list.append({
            "highlight_index": highlight_index,
            "highlight_time": highlight_time,
            "highlight_file": highlight_file,
            "score": score
        })
    
    # Đọc log file nếu có
    log_content = ""
    log_file = os.path.join(os.getcwd(), "logs", f"{os.path.splitext(file_name)[0]}_log.txt")
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            log_content = f.read()
    
    return {
        "job_id": job_id,
        "file_name": file_name,
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "num_segments": num_segments,
        "segment_duration": segment_duration,
        "output_path": output_path,
        "error_message": error_message,
        "highlights": highlight_list,
        "log": log_content
    }