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
import unicodedata
import re
from scipy import signal
from scipy.signal import find_peaks

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
        write_log(file_name, f"Đã tính toán RMS: min={float(np.min(rms)):.4f}, max={float(np.max(rms)):.4f}, mean={float(np.mean(rms)):.4f}")
        
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
        
        # 5. Thêm Tempo và Beat Strength - phát hiện nhịp điệu mạnh
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        # Chuyển đổi tempo từ numpy.ndarray sang Python float
        tempo = float(tempo)
        beat_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        beat_strength = (beat_strength - np.mean(beat_strength)) / np.std(beat_strength)
        write_log(file_name, f"Đã tính toán Tempo ({tempo:.2f} BPM) và Beat Strength")
        
        # 6. Thêm Harmonic-Percussive Source Separation để phát hiện phần giai điệu và nhịp
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_strength = librosa.feature.rms(y=y_harmonic)[0]
        percussive_strength = librosa.feature.rms(y=y_percussive)[0]
        harmonic_strength = (harmonic_strength - np.mean(harmonic_strength)) / np.std(harmonic_strength)
        percussive_strength = (percussive_strength - np.mean(percussive_strength)) / np.std(percussive_strength)
        write_log(file_name, "Đã tính toán Harmonic-Percussive Separation")
        
        # 7. Thêm Mel-frequency cepstral coefficients (MFCCs) để phát hiện đặc trưng âm sắc
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = np.var(mfcc, axis=1)  # Tính phương sai của mỗi hệ số MFCC
        mfcc_var = (mfcc_var - np.min(mfcc_var)) / (np.max(mfcc_var) - np.min(mfcc_var))  # Chuẩn hóa
        mfcc_dynamic = np.mean(mfcc_var)  # Đo lường sự thay đổi trong âm sắc
        write_log(file_name, "Đã tính toán MFCC và độ biến thiên âm sắc")
        
        # Kết hợp các đặc trưng với trọng số tối ưu hơn
        # Tăng trọng số cho RMS, Beat Strength và Percussive Strength để ưu tiên đoạn sôi động
        combined_feature = (
            rms * 0.3 +                  # Năng lượng âm thanh
            spectral_centroid * 0.1 +    # Độ sáng của âm thanh
            contrast * 0.1 +             # Sự khác biệt giữa peak và valley
            onset_env * 0.15 +           # Sự thay đổi đột ngột
            beat_strength * 0.2 +        # Độ mạnh của nhịp
            percussive_strength * 0.15   # Độ mạnh của phần nhịp điệu
        )
        
        # Áp dụng bộ lọc trung bình động để làm mịn đặc trưng
        window_size = 100  # Kích thước cửa sổ trung bình động
        combined_feature = np.convolve(combined_feature, np.ones(window_size)/window_size, mode='same')
        write_log(file_name, "Đã kết hợp và làm mịn các đặc trưng")
        
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
        
        # Thêm phân tích cấu trúc bài hát để phát hiện chorus/điệp khúc
        write_log(file_name, "Đang phân tích cấu trúc bài hát để phát hiện điệp khúc...")
        
        # Sử dụng chroma features để phát hiện cấu trúc hòa âm
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        
        # Tính ma trận tự tương quan để tìm các đoạn lặp lại (điệp khúc)
        write_log(file_name, "Đang tính ma trận tự tương quan...")
        similarity_matrix = librosa.segment.recurrence_matrix(
            chroma, width=int(segment_duration * sr / hop_length), mode='affinity'
        )
        
        # Tính tổng tương quan cho mỗi frame để phát hiện đoạn lặp lại nhiều nhất
        chorus_scores = np.sum(similarity_matrix, axis=1)
        chorus_scores = (chorus_scores - np.min(chorus_scores)) / (np.max(chorus_scores) - np.min(chorus_scores))
        
        # Chuyển đổi chorus_scores thành cùng độ dài với y
        if len(chorus_scores) < len(y) // hop_length:
            chorus_scores = np.repeat(chorus_scores, hop_length)
        else:
            chorus_scores = np.interp(
                np.linspace(0, 1, len(y)), 
                np.linspace(0, 1, len(chorus_scores) * hop_length), 
                np.repeat(chorus_scores, hop_length)
            )
        
        write_log(file_name, "Đã tính điểm số cho các đoạn điệp khúc tiềm năng")
        
        # Kết hợp điểm số đặc trưng với điểm số điệp khúc
        combined_score = combined_feature * 0.7 + chorus_scores[:len(combined_feature)] * 0.3
        
        # Tính điểm số cho mỗi đoạn
        for i in range(0, len(y) - frame_length, hop_length):
            segment = combined_score[i:i+frame_length]
            
            # Tính điểm trung bình của đoạn
            mean_score = np.mean(segment)
            
            # Tính thêm độ biến thiên để ưu tiên đoạn có sự thay đổi
            variation_score = np.std(segment) * 0.3
            
            # Tính điểm cuối cùng
            final_score = mean_score + variation_score
            
            scores.append((i, final_score))
        
        write_log(file_name, f"Đã tính toán điểm số cho {len(scores)} đoạn")
        
        # Sắp xếp theo điểm số giảm dần
        scores.sort(key=lambda x: x[1], reverse=True)
        write_log(file_name, "Đã sắp xếp các đoạn theo điểm số")
        
        # Lấy 2 đoạn có điểm cao nhất và đảm bảo chúng không trùng lặp về mặt cấu trúc
        num_segments = 2  # Luôn lấy 2 đoạn hay nhất
        top_segments = []
        # Tăng khoảng cách tối thiểu lên 200% độ dài đoạn để đảm bảo các đoạn cách xa nhau
        min_distance_samples = int(segment_duration * sr * 2.0)
        
        # Nếu file quá ngắn, điều chỉnh khoảng cách tối thiểu
        if len(y) < min_distance_samples * num_segments:
            min_distance_samples = max(int(len(y) / (num_segments * 1.5)), int(segment_duration * sr * 0.8))
            write_log(file_name, f"Điều chỉnh khoảng cách tối thiểu xuống {min_distance_samples} mẫu do file ngắn")
        
        # Tạo danh sách các đoạn đã chọn để kiểm tra tính tương tự
        selected_features = []
        
        write_log(file_name, "Đang chọn các đoạn highlight bắt đầu từ cao trào...")
        
        # Phát hiện cao trào trong bài hát
        # Sử dụng kết hợp năng lượng, onset và percussive để tìm cao trào
        climax_feature = (
            rms * 0.4 +                  # Năng lượng âm thanh
            onset_env * 0.3 +            # Sự thay đổi đột ngột
            percussive_strength * 0.3    # Độ mạnh của phần nhịp điệu
        )
        
        # Đảm bảo climax_feature có cùng độ dài với y
        if len(climax_feature) < len(y):
            climax_feature = np.interp(
                np.linspace(0, 1, len(y)), 
                np.linspace(0, 1, len(climax_feature)), 
                climax_feature
            )
        
        # Áp dụng bộ lọc để làm nổi bật các đỉnh
        # Đảm bảo window_length là số lẻ và nhỏ hơn độ dài của climax_feature
        window_length = min(51, len(climax_feature) - 2)
        if window_length % 2 == 0:
            window_length -= 1
        
        if window_length >= 3:  # Savgol filter yêu cầu window_length >= 3
            try:
                climax_feature = signal.savgol_filter(climax_feature, window_length, 3)
            except Exception as e:
                write_log(file_name, f"Lỗi khi áp dụng savgol_filter: {str(e)}, sử dụng dữ liệu gốc")
        
        # Tìm các đỉnh cục bộ (local maxima) - đây là các cao trào tiềm năng
        try:
            # Đảm bảo distance là số nguyên dương
            min_peak_distance = max(1, int(10 * sr / hop_length))  # Giảm xuống 10 giây
            peaks, _ = find_peaks(climax_feature, 
                                 height=np.mean(climax_feature) + 0.3*np.std(climax_feature),  # Giảm ngưỡng
                                 distance=min_peak_distance)
            
            write_log(file_name, f"Đã phát hiện {len(peaks)} cao trào tiềm năng")
        except Exception as e:
            write_log(file_name, f"Lỗi khi tìm peaks: {str(e)}, sử dụng phương pháp thay thế")
            # Phương pháp thay thế: lấy các điểm có giá trị cao nhất
            num_peaks = 10  # Số lượng đỉnh cần tìm
            peak_indices = np.argsort(climax_feature)[-num_peaks:]
            peaks = sorted(peak_indices)
            write_log(file_name, f"Đã phát hiện {len(peaks)} cao trào tiềm năng bằng phương pháp thay thế")
        
        # Chuyển đổi vị trí peak từ index của climax_feature sang vị trí mẫu trong y
        peak_samples = []
        for peak in peaks:
            # Đảm bảo peak nằm trong phạm vi của climax_feature
            if peak < len(climax_feature):
                # Chuyển đổi từ index trong climax_feature sang vị trí mẫu
                sample_pos = int(peak * hop_length)
                # Đảm bảo có đủ dữ liệu cho một đoạn hoàn chỉnh
                if sample_pos + frame_length <= len(y):
                    # Lấy giá trị tại vị trí peak
                    # Đảm bảo chuyển đổi thành Python float nguyên thủy, không phải numpy.float
                    peak_value = float(climax_feature[peak])
                    peak_samples.append((sample_pos, peak_value))
        
        # Sắp xếp các cao trào theo độ mạnh giảm dần
        peak_samples.sort(key=lambda x: x[1], reverse=True)
        write_log(file_name, f"Đã sắp xếp {len(peak_samples)} cao trào theo độ mạnh")
        
        # Tạo hai danh sách riêng biệt cho main và chorus
        main_segments = []  # Cho cao trào
        chorus_segments = []  # Cho điệp khúc

        # 1. Tìm đoạn main (cao trào) trước - luôn lấy 1 đoạn main
        write_log(file_name, "Đang tìm đoạn cao trào (main)...")

        # Sử dụng climax_feature đã tính toán để tìm cao trào
        if len(peak_samples) > 0:
            # Lấy cao trào mạnh nhất
            main_start_sample, main_score = peak_samples[0]
            
            try:
                # Tối ưu vị trí bắt đầu để bắt đầu từ cao trào
                window_start = max(0, main_start_sample - int(5 * sr))
                window_end = min(len(y), main_start_sample + int(2 * sr))
                
                if window_start < window_end:
                    window_start_idx = window_start // hop_length
                    window_end_idx = min(window_end // hop_length, len(climax_feature))
                    
                    if window_start_idx < window_end_idx and window_end_idx <= len(climax_feature):
                        window_feature = climax_feature[window_start_idx:window_end_idx]
                        
                        if len(window_feature) > 0:
                            local_peak_idx = np.argmax(window_feature)
                            local_peak = window_start + local_peak_idx * hop_length
                            adjusted_start = max(0, local_peak - int(0.5 * sr))
                            main_segments.append((adjusted_start, float(main_score), "main"))
                            write_log(file_name, f"Đã chọn đoạn main (cao trào) tại {adjusted_start/sr:.2f}s với điểm số {float(main_score):.4f}")
                        else:
                            main_segments.append((main_start_sample, float(main_score), "main"))
                            write_log(file_name, f"Window feature rỗng, đã chọn đoạn main tại vị trí ban đầu {main_start_sample/sr:.2f}s")
                    else:
                        main_segments.append((main_start_sample, float(main_score), "main"))
                        write_log(file_name, f"Chỉ số window không hợp lệ, đã chọn đoạn main tại vị trí ban đầu {main_start_sample/sr:.2f}s")
                else:
                    main_segments.append((main_start_sample, float(main_score), "main"))
                    write_log(file_name, f"Window không hợp lệ, đã chọn đoạn main tại vị trí ban đầu {main_start_sample/sr:.2f}s")
            except Exception as e:
                write_log(file_name, f"Lỗi khi tối ưu vị trí bắt đầu cho main: {str(e)}, sử dụng vị trí ban đầu")
                main_segments.append((main_start_sample, float(main_score), "main"))
                write_log(file_name, f"Đã chọn đoạn main tại vị trí ban đầu {main_start_sample/sr:.2f}s với điểm số {float(main_score):.4f}")
        else:
            # Nếu không tìm thấy cao trào, sử dụng đoạn có điểm combined_score cao nhất
            write_log(file_name, "Không tìm thấy cao trào, sử dụng đoạn có điểm tổng hợp cao nhất cho main...")
            if len(scores) > 0:
                main_start_sample, main_score = scores[0]  # Lấy đoạn có điểm cao nhất
                main_segments.append((main_start_sample, float(main_score), "main"))
                write_log(file_name, f"Đã chọn đoạn main tại {main_start_sample/sr:.2f}s với điểm số {float(main_score):.4f}")

        # 2. Tìm nhiều đoạn chorus (điệp khúc) - số lượng = num_segments - 1
        num_chorus = max(1, num_segments - 1)  # Đảm bảo ít nhất 1 chorus
        write_log(file_name, f"Đang tìm {num_chorus} đoạn điệp khúc (chorus)...")

        # Tính điểm số riêng cho chorus, tăng trọng số cho chorus_scores
        chorus_specific_scores = []
        for i in range(0, len(y) - frame_length, hop_length):
            # Lấy đoạn chorus_scores tương ứng
            if i < len(chorus_scores):
                segment = chorus_scores[i:i+frame_length] if i+frame_length <= len(chorus_scores) else chorus_scores[i:]
                if len(segment) > 0:
                    # Tính điểm trung bình của đoạn
                    mean_score = np.mean(segment)
                    # Tính thêm độ biến thiên để ưu tiên đoạn có sự thay đổi
                    variation_score = np.std(segment) * 0.2
                    # Tính điểm cuối cùng
                    final_score = mean_score + variation_score
                    chorus_specific_scores.append((i, float(final_score)))

        # Sắp xếp theo điểm số giảm dần
        if chorus_specific_scores:
            chorus_specific_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Tìm nhiều đoạn chorus có điểm cao và đủ xa nhau
            chorus_candidates = []
            for chorus_start_sample, chorus_score in chorus_specific_scores:
                # Kiểm tra khoảng cách với đoạn main
                is_far_from_main = True
                for main_start, _, _ in main_segments:
                    if abs(chorus_start_sample - main_start) < min_distance_samples:
                        is_far_from_main = False
                        break
                
                # Kiểm tra khoảng cách với các đoạn chorus đã chọn
                is_far_from_other_chorus = True
                for other_chorus_start, _, _ in chorus_segments:
                    if abs(chorus_start_sample - other_chorus_start) < min_distance_samples:
                        is_far_from_other_chorus = False
                        break
                
                if is_far_from_main and is_far_from_other_chorus:
                    chorus_segments.append((chorus_start_sample, float(chorus_score), "chorus"))
                    write_log(file_name, f"Đã chọn đoạn chorus (điệp khúc) tại {chorus_start_sample/sr:.2f}s với điểm số {float(chorus_score):.4f}")
                    
                    # Nếu đã đủ số lượng chorus cần lấy thì dừng
                    if len(chorus_segments) >= num_chorus:
                        break

        # Nếu không tìm đủ đoạn chorus, sử dụng đoạn có điểm tổng hợp cao
        if len(chorus_segments) < num_chorus:
            write_log(file_name, f"Chỉ tìm được {len(chorus_segments)} đoạn chorus, cần thêm {num_chorus - len(chorus_segments)} đoạn...")
            for start_sample, score in scores:
                # Kiểm tra khoảng cách với đoạn main
                is_far_from_main = True
                for main_start, _, _ in main_segments:
                    if abs(start_sample - main_start) < min_distance_samples:
                        is_far_from_main = False
                        break
                
                # Kiểm tra khoảng cách với các đoạn chorus đã chọn
                is_far_from_other_chorus = True
                for other_chorus_start, _, _ in chorus_segments:
                    if abs(start_sample - other_chorus_start) < min_distance_samples:
                        is_far_from_other_chorus = False
                        break
                
                if is_far_from_main and is_far_from_other_chorus:
                    chorus_segments.append((start_sample, float(score), "chorus"))
                    write_log(file_name, f"Đã chọn đoạn chorus thay thế tại {start_sample/sr:.2f}s với điểm số {float(score):.4f}")
                    
                    # Nếu đã đủ số lượng chorus cần lấy thì dừng
                    if len(chorus_segments) >= num_chorus:
                        break

        # Kết hợp main và chorus segments
        top_segments = main_segments + chorus_segments

        # Sắp xếp lại theo thứ tự thời gian
        top_segments.sort(key=lambda x: x[0])
        write_log(file_name, f"Đã chọn tổng cộng {len(top_segments)} đoạn highlight (main: {len(main_segments)}, chorus: {len(chorus_segments)})")
        
        highlights = []
        write_log(file_name, "Đang xuất các đoạn highlight...")
        
        # Lấy tên file gốc không có phần mở rộng
        base_filename = os.path.splitext(os.path.basename(file_name))[0]
        
        # Xử lý tên file để tránh lỗi Unicode
        if any(ord(c) > 127 for c in base_filename):
            base_filename_ascii = unicodedata.normalize('NFKD', base_filename)
            base_filename_ascii = re.sub(r'[^\x00-\x7F]+', '_', base_filename_ascii)
            write_log(file_name, f"Đã chuyển đổi tên file có dấu '{base_filename}' thành '{base_filename_ascii}'")
            base_filename = base_filename_ascii
        
        for idx, (start_sample, score, segment_type) in enumerate(top_segments):
            start_time_hl = start_sample / sr
            
            audio_segment = AudioSegment.from_file(audio_file_path)
            start_ms = int(start_time_hl * 1000)
            end_ms = min(len(audio_segment), start_ms + int(segment_duration * 1000))
            highlight_audio = audio_segment[start_ms:end_ms]
            
            # Chuyển đổi sang định dạng yêu cầu: 8000Hz, mono
            highlight_audio = highlight_audio.set_frame_rate(8000).set_channels(1)
            
            # Tạo tên file theo yêu cầu: tên file gốc + số thứ tự
            highlight_output_path = os.path.join(output_dir, f"{base_filename} {idx+1}.mp3")
            
            # Xuất file MP3 với bitrate 64kbps
            highlight_audio.export(
                highlight_output_path, 
                format="mp3",
                bitrate="64k"
            )
            
            # Kiểm tra kích thước file, nếu > 500KB thì giảm bitrate và xuất lại
            file_size = os.path.getsize(highlight_output_path)
            if file_size > 500 * 1024:  # 500KB
                write_log(file_name, f"File {highlight_output_path} có kích thước {file_size/1024:.2f}KB > 500KB, đang giảm bitrate...")
                # Giảm bitrate và xuất lại
                highlight_audio.export(
                    highlight_output_path, 
                    format="mp3",
                    bitrate="48k"  # Giảm bitrate xuống 48kbps
                )
                
                # Nếu vẫn > 500KB, giảm thời lượng
                file_size = os.path.getsize(highlight_output_path)
                if file_size > 500 * 1024:
                    # Giảm thời lượng xuống 40 giây (vẫn trong khoảng 40-60s)
                    highlight_audio = highlight_audio[:40000]  # 40 giây
                    highlight_audio.export(
                        highlight_output_path, 
                        format="mp3",
                        bitrate="48k"
                    )
            
            write_log(file_name, f"Đã xuất highlight {idx+1} tại {start_time_hl:.2f}s vào file {highlight_output_path}")
            
            # Lưu thông tin highlight vào database với thông tin loại đoạn
            save_highlight(job_id, idx+1, float(start_time_hl), highlight_output_path, float(score))
            
            highlights.append({
                "highlight_time": float(start_time_hl),
                "highlight_file": highlight_output_path,
                "score": float(score),
                "duration": segment_duration,
                "type": segment_type  # Thêm thông tin loại đoạn
            })
        
        # Cập nhật trạng thái job thành công
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        update_job_status(job_id, "COMPLETED", end_time, output_dir)
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
async def get_highlight(audio_file: UploadFile, segment_duration: float = 40.0, num_segments: int = 2):
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
@app.post("/highlight_multiple")
async def get_multiple_highlights(audio_files: list[UploadFile], segment_duration: float = 40.0, num_segments: int = 2):
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
@app.get("/download_highlight")
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
