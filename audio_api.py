from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
import librosa
import numpy as np
import shutil
import tempfile
import os
import time
import json
import sqlite3
from datetime import datetime
from pydub import AudioSegment
import asyncio
import logging

app = FastAPI()

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

# Khởi tạo cơ sở dữ liệu SQLite
def init_db():
    conn = sqlite3.connect('audio_processing.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS processing_jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT,
        status TEXT,
        start_time TEXT,
        end_time TEXT,
        num_segments INTEGER,
        segment_duration REAL,
        output_path TEXT,
        error_message TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS highlights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER,
        highlight_index INTEGER,
        highlight_time REAL,
        highlight_file TEXT,
        score REAL,
        FOREIGN KEY (job_id) REFERENCES processing_jobs (id)
    )
    ''')
    
    conn.commit()
    conn.close()

# Gọi hàm khởi tạo DB khi ứng dụng khởi động
init_db()

# Hàm lưu thông tin job vào DB
def save_job(file_name, status, start_time, num_segments, segment_duration, output_path=None, end_time=None, error_message=None):
    conn = sqlite3.connect('audio_processing.db')
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO processing_jobs (file_name, status, start_time, end_time, num_segments, segment_duration, output_path, error_message)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (file_name, status, start_time, end_time, num_segments, segment_duration, output_path, error_message))
    job_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return job_id

# Hàm cập nhật trạng thái job
def update_job_status(job_id, status, end_time=None, output_path=None, error_message=None):
    conn = sqlite3.connect('audio_processing.db')
    cursor = conn.cursor()
    
    update_fields = ["status = ?"]
    params = [status]
    
    if end_time:
        update_fields.append("end_time = ?")
        params.append(end_time)
    
    if output_path:
        update_fields.append("output_path = ?")
        params.append(output_path)
    
    if error_message:
        update_fields.append("error_message = ?")
        params.append(error_message)
    
    params.append(job_id)
    
    cursor.execute(f'''
    UPDATE processing_jobs SET {", ".join(update_fields)} WHERE id = ?
    ''', params)
    
    conn.commit()
    conn.close()

# Hàm lưu thông tin highlight
def save_highlight(job_id, highlight_index, highlight_time, highlight_file, score):
    conn = sqlite3.connect('audio_processing.db')
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO highlights (job_id, highlight_index, highlight_time, highlight_file, score)
    VALUES (?, ?, ?, ?, ?)
    ''', (job_id, highlight_index, highlight_time, highlight_file, score))
    conn.commit()
    conn.close()

# Hàm lấy thông tin job đang xử lý
def get_processing_jobs():
    conn = sqlite3.connect('audio_processing.db')
    cursor = conn.cursor()
    cursor.execute('''
    SELECT id, file_name, status, start_time, end_time, num_segments, segment_duration, output_path
    FROM processing_jobs
    ORDER BY start_time DESC
    ''')
    jobs = cursor.fetchall()
    conn.close()
    return jobs

# Hàm lấy highlights của một job
def get_job_highlights(job_id):
    conn = sqlite3.connect('audio_processing.db')
    cursor = conn.cursor()
    cursor.execute('''
    SELECT highlight_index, highlight_time, highlight_file, score
    FROM highlights
    WHERE job_id = ?
    ORDER BY highlight_index
    ''', (job_id,))
    highlights = cursor.fetchall()
    conn.close()
    return highlights

# Hàm ghi log vào file txt
def write_log(file_name, message):
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{os.path.splitext(file_name)[0]}_log.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

# Thêm hàm xử lý background để tránh timeout khi xử lý nhiều file
async def process_audio_file(audio_file_path, output_dir, file_name, num_segments, segment_duration):
    # Tạo job trong database
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    job_id = save_job(file_name, "PROCESSING", start_time, num_segments, segment_duration)
    
    # Ghi log bắt đầu xử lý
    write_log(file_name, f"Bắt đầu xử lý file {file_name} - Job ID: {job_id}")
    logger.info(f"Bắt đầu xử lý file {file_name} - Job ID: {job_id}")
    
    try:
        # Tải file âm thanh
        write_log(file_name, "Đang tải file âm thanh...")
        y, sr = librosa.load(audio_file_path, sr=None)
        write_log(file_name, f"Đã tải file âm thanh: {len(y)/sr:.2f} giây, sample rate: {sr}Hz")
        
        # Tính toán nhiều đặc trưng âm thanh
        write_log(file_name, "Đang tính toán đặc trưng âm thanh...")
        
        # 1. Năng lượng (RMS)
        rms = librosa.feature.rms(y=y)[0]
        write_log(file_name, f"Đã tính toán RMS: min={np.min(rms):.4f}, max={np.max(rms):.4f}, mean={np.mean(rms):.4f}")
        
        # 2. Spectral Centroid - đặc trưng cho "độ sáng" của âm thanh
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid = (spectral_centroid - np.mean(spectral_centroid)) / np.std(spectral_centroid)
        write_log(file_name, "Đã tính toán Spectral Centroid")
        
        # 3. Spectral Contrast - đặc trưng cho sự khác biệt giữa peak và valley trong spectrum
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=0)
        contrast = (contrast - np.mean(contrast)) / np.std(contrast)
        write_log(file_name, "Đã tính toán Spectral Contrast")
        
        # 4. Onset Strength - phát hiện sự thay đổi đột ngột trong âm thanh
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_env = (onset_env - np.mean(onset_env)) / np.std(onset_env)
        write_log(file_name, "Đã tính toán Onset Strength")
        
        # Kết hợp các đặc trưng (có thể điều chỉnh trọng số)
        combined_feature = rms * 0.4 + spectral_centroid * 0.2 + contrast * 0.2 + onset_env * 0.2
        write_log(file_name, "Đã kết hợp các đặc trưng")
        
        # Áp dụng cửa sổ trượt để tìm đoạn có tổng đặc trưng cao nhất
        frame_length = int(segment_duration * sr)
        hop_length = int(sr * 0.5)  # Bước nhảy 0.5 giây
        write_log(file_name, f"Thiết lập cửa sổ trượt: frame_length={frame_length}, hop_length={hop_length}")
        
        # Chuyển đổi combined_feature thành cùng độ dài với y nếu cần
        if len(combined_feature) < len(y):
            combined_feature = np.interp(
                np.linspace(0, 1, len(y)), 
                np.linspace(0, 1, len(combined_feature)), 
                combined_feature
            )
            write_log(file_name, "Đã nội suy đặc trưng kết hợp")
        
        # Tìm đoạn có tổng đặc trưng cao nhất
        write_log(file_name, "Đang tìm đoạn có đặc trưng cao nhất...")
        scores = []
        for i in range(0, len(y) - frame_length, hop_length):
            segment = combined_feature[i:i+frame_length]
            scores.append((i, np.mean(segment)))
        write_log(file_name, f"Đã tính toán điểm số cho {len(scores)} đoạn")
        
        # Phân tích cấu trúc bài hát để phát hiện các đoạn lặp lại
        write_log(file_name, "Đang phân tích cấu trúc bài hát...")
        # Sử dụng chroma features để phát hiện cấu trúc hòa âm
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Tính ma trận tự tương quan để tìm các đoạn tương tự nhau
        write_log(file_name, "Đang tính ma trận tự tương quan...")
        similarity_matrix = librosa.segment.recurrence_matrix(
            chroma, width=int(segment_duration * sr / hop_length), mode='affinity'
        )
        
        # Phân đoạn bài hát dựa trên sự thay đổi trong đặc trưng
        write_log(file_name, "Đang phân đoạn bài hát...")
        boundaries = librosa.segment.agglomerative(similarity_matrix, int(len(y) / (sr * 10)))
        boundary_times = librosa.frames_to_time(boundaries, sr=sr)
        
        # Chuyển đổi boundaries thành các đoạn
        segments = []
        for i in range(len(boundary_times) - 1):
            start_time_seg = boundary_times[i]
            end_time_seg = boundary_times[i+1]
            segments.append((start_time_seg, end_time_seg))
        write_log(file_name, f"Đã phân đoạn bài hát thành {len(segments)} đoạn")
        
        # Sắp xếp theo điểm số giảm dần
        scores.sort(key=lambda x: x[1], reverse=True)
        write_log(file_name, "Đã sắp xếp các đoạn theo điểm số")
        
        # Lấy n đoạn có điểm cao nhất và đảm bảo chúng không trùng lặp về mặt cấu trúc
        top_segments = []
        # Tăng khoảng cách tối thiểu lên 200% độ dài đoạn để đảm bảo các đoạn cách xa nhau
        min_distance_samples = int(segment_duration * sr * 2.0)
        
        # Nếu file quá ngắn, điều chỉnh khoảng cách tối thiểu
        if len(y) < min_distance_samples * num_segments:
            min_distance_samples = max(int(len(y) / (num_segments * 1.5)), int(segment_duration * sr * 0.8))
            write_log(file_name, f"Điều chỉnh khoảng cách tối thiểu xuống {min_distance_samples} mẫu do file ngắn")
        
        # Tạo danh sách các đoạn đã chọn để kiểm tra tính tương tự
        selected_features = []
        
        write_log(file_name, "Đang chọn các đoạn highlight...")
        for start_sample, score in scores:
            start_time_hl = start_sample / sr
            
            # Kiểm tra khoảng cách với các đoạn đã chọn
            is_far_enough = True
            for selected_start, _ in top_segments:
                if abs(start_sample - selected_start) < min_distance_samples:
                    is_far_enough = False
                    break
            
            if not is_far_enough:
                continue
            
            # Kiểm tra xem đoạn này có thuộc cùng một phân đoạn cấu trúc với đoạn nào đã chọn không
            current_segment_idx = -1
            for i, (seg_start, seg_end) in enumerate(segments):
                if seg_start <= start_time_hl < seg_end:
                    current_segment_idx = i
                    break
            
            # Kiểm tra tính tương tự về mặt âm nhạc với các đoạn đã chọn
            is_unique = True
            if len(selected_features) > 0:
                # Trích xuất đặc trưng của đoạn hiện tại
                end_sample = min(start_sample + frame_length, len(y))
                current_segment = y[start_sample:end_sample]
                current_chroma = librosa.feature.chroma_stft(y=current_segment, sr=sr)
                
                # So sánh với các đoạn đã chọn
                for prev_chroma in selected_features:
                    # Tính độ tương tự cosine
                    similarity = np.mean(
                        [np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2)) 
                         for c1, c2 in zip(current_chroma, prev_chroma) if np.linalg.norm(c1) > 0 and np.linalg.norm(c2) > 0]
                    )
                    
                    # Nếu độ tương tự quá cao (> 0.85), coi là trùng lặp
                    if similarity > 0.85:
                        is_unique = False
                        break
            
            if is_far_enough and is_unique:
                # Lưu đặc trưng của đoạn này để so sánh sau
                end_sample = min(start_sample + frame_length, len(y))
                current_segment = y[start_sample:end_sample]
                current_chroma = librosa.feature.chroma_stft(y=current_segment, sr=sr)
                selected_features.append(current_chroma)
                
                top_segments.append((start_sample, score))
                write_log(file_name, f"Đã chọn đoạn highlight tại {start_time_hl:.2f}s với điểm số {score:.4f}")
                if len(top_segments) >= num_segments:
                    break
        
        # Nếu không tìm đủ số đoạn không trùng lặp, giảm ngưỡng tương tự
        if len(top_segments) < num_segments and len(scores) > num_segments:
            write_log(file_name, f"Không tìm đủ {num_segments} đoạn không trùng lặp, đang giảm ngưỡng tương tự...")
            for start_sample, score in scores:
                if len(top_segments) >= num_segments:
                    break
                    
                # Kiểm tra xem đoạn này đã được chọn chưa
                already_selected = any(start_sample == selected_start for selected_start, _ in top_segments)
                
                if already_selected:
                    continue
                    
                # Kiểm tra khoảng cách với các đoạn đã chọn
                is_far_enough = True
                for selected_start, _ in top_segments:
                    if abs(start_sample - selected_start) < min_distance_samples // 2:  # Giảm khoảng cách
                        is_far_enough = False
                        break
                
                if is_far_enough:
                    top_segments.append((start_sample, score))
                    write_log(file_name, f"Đã chọn đoạn highlight bổ sung tại {start_sample/sr:.2f}s với điểm số {score:.4f}")
        
        # Sắp xếp lại theo thứ tự thời gian
        top_segments.sort(key=lambda x: x[0])
        write_log(file_name, f"Đã chọn tổng cộng {len(top_segments)} đoạn highlight")
        
        highlights = []
        write_log(file_name, "Đang xuất các đoạn highlight...")
        for idx, (start_sample, score) in enumerate(top_segments):
            start_time_hl = start_sample / sr
            
            audio_segment = AudioSegment.from_file(audio_file_path)
            start_ms = int(start_time_hl * 1000)
            end_ms = min(len(audio_segment), start_ms + int(segment_duration * 1000))
            highlight_audio = audio_segment[start_ms:end_ms]
            
            # Tạo tên file với timestamp để đảm bảo tính duy nhất
            timestamp = int(start_time_hl)
            # Tạo thư mục con cho file này
            file_folder_name = os.path.splitext(os.path.basename(file_name))[0]
            file_output_dir = os.path.join(output_dir, file_folder_name)
            os.makedirs(file_output_dir, exist_ok=True)
            
            highlight_output_path = os.path.join(file_output_dir, f"highlight_{idx+1}_{timestamp}.mp3")
            highlight_audio.export(highlight_output_path, format="mp3")
            write_log(file_name, f"Đã xuất highlight {idx+1} tại {start_time_hl:.2f}s vào file {highlight_output_path}")
            
            # Lưu thông tin highlight vào database
            save_highlight(job_id, idx+1, float(start_time_hl), highlight_output_path, float(score))
            
            highlights.append({
                "highlight_time": float(start_time_hl),
                "highlight_file": highlight_output_path,
                "score": float(score),
                "duration": segment_duration
            })
        
        # Cập nhật trạng thái job thành công
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        update_job_status(job_id, "COMPLETED", end_time, file_output_dir)
        write_log(file_name, f"Hoàn thành xử lý file {file_name} - Job ID: {job_id}")
        logger.info(f"Hoàn thành xử lý file {file_name} - Job ID: {job_id}")
        
        return {
            "num_segments": len(highlights),
            "highlights": highlights,
            "job_id": job_id
        }
        
    except Exception as e:
        # Ghi log lỗi
        error_message = str(e)
        write_log(file_name, f"Lỗi khi xử lý file {file_name}: {error_message}")
        logger.error(f"Lỗi khi xử lý file {file_name}: {error_message}", exc_info=True)
        
        # Cập nhật trạng thái job thất bại
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        update_job_status(job_id, "FAILED", end_time, error_message=error_message)
        
        # Trả về thông báo lỗi
        return {
            "error": f"Error processing audio: {error_message}",
            "job_id": job_id
        }

@app.post("/highlight")
async def get_highlight(audio_file: UploadFile, num_segments: int = 1, segment_duration: float = 5.0):
    # Tạo thư mục output nếu chưa tồn tại
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
        shutil.copyfileobj(audio_file.file, temp_audio)
        audio_path = temp_audio.name

    # Xử lý file audio
    result = await process_audio_file(audio_path, output_dir, audio_file.filename, num_segments, segment_duration)
    
    # Xóa file tạm sau khi xử lý xong
    try:
        os.unlink(audio_path)
    except:
        pass
        
    return result

# Endpoint mới để xử lý nhiều file cùng lúc
@app.post("/highlight_multiple")
async def get_multiple_highlights(audio_files: list[UploadFile], num_segments: int = 1, segment_duration: float = 5.0):
    # Tạo thư mục output nếu chưa tồn tại
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for audio_file in audio_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            shutil.copyfileobj(audio_file.file, temp_audio)
            audio_path = temp_audio.name
        
        # Xử lý file audio
        result = await process_audio_file(audio_path, output_dir, audio_file.filename, num_segments, segment_duration)
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
@app.get("/download_highlight")
async def download_highlight(file_path: str):
    if not os.path.exists(file_path):
        return {"error": "File không tồn tại"}
    
    # Đảm bảo filename có đuôi .mp3
    filename = os.path.basename(file_path)
    if not filename.endswith('.mp3'):
        filename += '.mp3'
    
    return FileResponse(
        path=file_path, 
        media_type="audio/mpeg", 
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# Endpoint để lấy danh sách các job đã xử lý
@app.get("/jobs")
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
@app.get("/job/{job_id}")
async def get_job(job_id: int):
    conn = sqlite3.connect('audio_processing.db')
    cursor = conn.cursor()
    cursor.execute('''
    SELECT id, file_name, status, start_time, end_time, num_segments, segment_duration, output_path, error_message
    FROM processing_jobs
    WHERE id = ?
    ''', (job_id,))
    job = cursor.fetchone()
    conn.close()
    
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
