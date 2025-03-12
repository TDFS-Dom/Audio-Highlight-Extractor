import sqlite3
from datetime import datetime

# Đường dẫn đến file database
DB_PATH = 'audio_processing.db'

# Khởi tạo cơ sở dữ liệu SQLite
def init_db():
    conn = sqlite3.connect(DB_PATH)
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

# Hàm lưu thông tin job vào DB
def save_job(file_name, status, start_time, num_segments, segment_duration, output_path=None, end_time=None, error_message=None):
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

# Hàm cập nhật trạng thái job
def update_job_status(job_id, status, end_time=None, output_path=None, error_message=None):
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

# Hàm lưu thông tin highlight
def save_highlight(job_id, highlight_index, highlight_time, highlight_file, score):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO highlights (job_id, highlight_index, highlight_time, highlight_file, score)
    VALUES (?, ?, ?, ?, ?)
    ''', (job_id, highlight_index, highlight_time, highlight_file, score))
    conn.commit()
    conn.close()

# Hàm lấy thông tin job đang xử lý
def get_processing_jobs():
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

# Hàm lấy highlights của một job
def get_job_highlights(job_id):
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

# Hàm lấy chi tiết của một job
def get_job_details(job_id):
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