import sqlite3
import json
import os
from datetime import datetime
import logging
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
# Đường dẫn đến file database
DB_PATH = 'audio_processing.db'

def write_log(file_name, log_content):
    """Ghi log vào file và console"""
    # Tạo thư mục logs nếu chưa tồn tại
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Tạo tên file log dựa trên tên file âm thanh
    base_filename = os.path.splitext(os.path.basename(file_name))[0]
    log_file = os.path.join(logs_dir, f"{base_filename}_log.txt")
    
    # Thời gian hiện tại
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # Định dạng log
    log_entry = f"[{timestamp}] {log_content}\n"
    
    # Ghi vào file
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)
    
    # In ra console để debug
    print(log_entry.strip())

def save_job(file_name, status, start_time, num_segments, segment_duration, output_path=None, end_time=None, error_message=None):
    """Lưu thông tin job xử lý vào database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO processing_jobs (file_name, status, start_time, end_time, num_segments, segment_duration, output_path, error_message)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (file_name, status, start_time, end_time, num_segments, segment_duration, output_path, error_message))
    job_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return job_id

def update_job_status(job_id, status, end_time=None, output_path=None, error_message=None):
    """Cập nhật trạng thái của job xử lý"""
    conn = sqlite3.connect(DB_PATH)
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

def save_highlight(job_id, highlight_index, highlight_time, highlight_file, score):
    """Lưu thông tin highlight vào database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO highlights (job_id, highlight_index, highlight_time, highlight_file, score)
    VALUES (?, ?, ?, ?, ?)
    ''', (job_id, highlight_index, highlight_time, highlight_file, score))
    conn.commit()
    conn.close()

def get_processing_jobs():
    """Lấy danh sách các job đang xử lý từ database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    SELECT id, file_name, status, start_time, end_time, num_segments, segment_duration, output_path
    FROM processing_jobs
    ORDER BY start_time DESC
    ''')
    jobs = cursor.fetchall()
    conn.close()
    return jobs

def get_job_highlights(job_id):
    """Lấy danh sách highlights của một job từ database"""
    conn = sqlite3.connect(DB_PATH)
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

def get_job_details(job_id):
    """Lấy thông tin chi tiết của một job từ database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    SELECT id, file_name, status, start_time, end_time, num_segments, segment_duration, output_path, error_message
    FROM processing_jobs
    WHERE id = ?
    ''', (job_id,))
    job = cursor.fetchone()
    conn.close()
    return job

# Thêm các hàm tương tác với SQL khác:
# - Các hàm kết nối database
# - Các hàm thực thi truy vấn
# - Các hàm lưu trữ kết quả 