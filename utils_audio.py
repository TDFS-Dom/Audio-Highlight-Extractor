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
from utils_sql import (
    write_log, save_job, update_job_status, save_highlight,
    get_processing_jobs, get_job_highlights, get_job_details
)
import logging
import whisper
import torch

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

def find_highest_pitch_point(start_pos, audio_data, sample_rate, window_size=10, is_main=False):
    """Tìm điểm có cao độ cao nhất trong khoảng thời gian được chỉ định"""
    # File name sẽ được truyền từ hàm gọi
    file_name = "audio_processing"  # Mặc định nếu không có thông tin file_name
    
    if start_pos + window_size * sample_rate > len(audio_data):
        window_size = (len(audio_data) - start_pos) / sample_rate
    
    # Lấy đoạn âm thanh cần phân tích
    segment = audio_data[start_pos:start_pos + int(window_size * sample_rate)]
    
    # Tính toán các đặc trưng âm thanh
    # 1. Pitch (f0) sử dụng librosa
    pitches, magnitudes = librosa.piptrack(y=segment, sr=sample_rate)
    
    # 2. Năng lượng (RMS)
    rms = librosa.feature.rms(y=segment)[0]
    
    # 3. Onset strength - phát hiện sự thay đổi đột ngột trong âm thanh
    onset_env = librosa.onset.onset_strength(y=segment, sr=sample_rate)
    
    # 4. Spectral contrast - đặc trưng cho sự khác biệt giữa peak và valley trong spectrum
    contrast = np.mean(librosa.feature.spectral_contrast(y=segment, sr=sample_rate), axis=0)
    
    # Tìm pitch cao nhất tại mỗi frame và kết hợp với các đặc trưng khác
    frame_scores = []
    for t in range(pitches.shape[1]):
        # Tìm pitch cao nhất trong frame
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        magnitude = magnitudes[index, t]
        
        if pitch > 0:  # Chỉ xét các pitch > 0
            # Lấy các đặc trưng khác tại frame này
            frame_idx = min(t, len(rms) - 1)
            energy = rms[frame_idx] if frame_idx < len(rms) else 0
            onset = onset_env[frame_idx] if frame_idx < len(onset_env) else 0
            spec_contrast = contrast[frame_idx] if frame_idx < len(contrast) else 0
            
            # Tính điểm tổng hợp cho frame
            if is_main:
                # Cho main segment, ưu tiên cao độ cao và năng lượng mạnh
                score = (pitch * 0.6 + magnitude * 0.15 + energy * 0.15 + onset * 0.1)
            else:
                # Cho chorus segment, ưu tiên độ rõ ràng của giọng hát
                score = (pitch * 0.4 + magnitude * 0.2 + energy * 0.25 + onset * 0.15)
            
            frame_scores.append((t, pitch, score))
    
    if not frame_scores:
        return start_pos
    
    # Sắp xếp theo điểm số giảm dần
    frame_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Lấy top 5 frame có điểm cao nhất
    top_frames = frame_scores[:5]
    
    # Tìm frame có điểm cao nhất trong top 5 và nằm trong 1/3 đầu của đoạn phân tích
    early_high_pitch_frames = [f for f in top_frames if f[0] < pitches.shape[1] // 3]
    
    if early_high_pitch_frames and is_main:
        # Nếu là main segment và có frame cao độ cao ở đầu, ưu tiên chọn
        best_frame = early_high_pitch_frames[0]
    else:
        # Nếu không, lấy frame có điểm cao nhất
        best_frame = top_frames[0]
    
    # Chuyển đổi từ frame index sang vị trí mẫu
    highest_pitch_pos = start_pos + int(best_frame[0] * sample_rate / pitches.shape[1])
    
    # THAY ĐỔI: Điều chỉnh vị trí bắt đầu để luôn bắt đầu ở phần có lời nhưng không lùi quá nhiều
    if is_main:
        adjusted_start = max(0, highest_pitch_pos - int(1.0 * sample_rate))  # Giảm từ 1.5s xuống 1.0s
    else:
        adjusted_start = max(0, highest_pitch_pos - int(1.5 * sample_rate))  # Giảm từ 2.5s xuống 1.5s
    
    return adjusted_start

def detect_vocals_with_whisper(audio_data, sample_rate, file_name, segment_start_time=0):
    """Sử dụng OpenAI Whisper để phát hiện chính xác đoạn có lời trong bài hát"""
    write_log(file_name, f"Đang phát hiện lời bài hát bằng Whisper cho đoạn từ {segment_start_time:.2f}s")
    
    try:
        # Tạo file tạm thời để lưu đoạn audio cần phân tích
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        # Chuyển đổi numpy array sang định dạng WAV
        audio_segment = AudioSegment(
            audio_data.tobytes(),
            frame_rate=sample_rate,
            sample_width=audio_data.dtype.itemsize,
            channels=1
        )
        audio_segment.export(temp_filename, format="wav")
        
        # Tải mô hình Whisper (kích thước nhỏ để xử lý nhanh)
        # Có thể chọn: "tiny", "base", "small", "medium", "large"
        model = whisper.load_model("tiny")
        
        # Thiết lập các tùy chọn nhận dạng tiếng nói
        options = {
            "word_timestamps": True,  # Bật timestamp ở mức từ
            "max_initial_timestamp": 1.0,  # Giới hạn thời gian bắt đầu nhận dạng
        }
        
        # Thực hiện nhận dạng tiếng nói
        result = model.transcribe(temp_filename, **options)
        
        # Xóa file tạm
        os.unlink(temp_filename)
        
        # Xử lý kết quả để tìm thời điểm bắt đầu có lời
        segments = result["segments"]
        words = []
        
        # Thu thập tất cả các từ với timestamp
        for segment in segments:
            if "words" in segment:
                words.extend(segment["words"])
        
        # Nếu không tìm thấy từ nào, có thể không có lời
        if not words:
            write_log(file_name, "Không phát hiện lời bài hát trong đoạn audio này")
            return None
        
        # Lấy thời điểm của từ đầu tiên
        first_word_time = words[0]["start"]
        
        # Tính toán thời điểm tuyệt đối trong file gốc
        absolute_start_time = segment_start_time + first_word_time
        
        write_log(file_name, f"Phát hiện lời bài hát tại {absolute_start_time:.2f}s")
        return absolute_start_time
        
    except Exception as e:
        write_log(file_name, f"Lỗi khi phát hiện lời bằng Whisper: {str(e)}")
        return None

def detect_instrumental_intro(start_pos, audio_data, sample_rate, is_main=False):
    """Phát hiện và trả về vị trí bắt đầu có lời, loại bỏ nhạc không lời ở đầu đoạn"""
    file_name = "audio_processing"  # Mặc định nếu không có thông tin file_name
    
    # Thử phát hiện lời bằng Whisper trước
    window_size = min(45, (len(audio_data) - start_pos) / sample_rate)  # Tối đa 45s
    segment = audio_data[start_pos:start_pos + int(window_size * sample_rate)]
    
    # Tính thời điểm bắt đầu phân đoạn tính từ đầu file
    segment_start_time = start_pos / sample_rate
    
    # Sử dụng Whisper để phát hiện lời chính xác
    whisper_vocal_time = detect_vocals_with_whisper(segment, sample_rate, file_name, segment_start_time)
    
    if whisper_vocal_time is not None:
        # Chuyển từ thời gian sang mẫu
        absolute_vocal_pos = int(whisper_vocal_time * sample_rate)
        
        # Điều chỉnh vị trí khác nhau cho main và chorus
        if is_main:
            # Main: bắt đầu gần với lời, lùi 1s 
            adjusted_start = max(start_pos, absolute_vocal_pos - 1 * sample_rate)
            write_log(file_name, f"MAIN (Whisper): Điều chỉnh vị trí từ {start_pos/sample_rate:.2f}s thành {adjusted_start/sample_rate:.2f}s (lời bắt đầu tại {whisper_vocal_time:.2f}s)")
        else:
            # Chorus: cần 3s nhạc dẫn
            adjusted_start = max(start_pos, absolute_vocal_pos - 3 * sample_rate)
            write_log(file_name, f"CHORUS (Whisper): Điều chỉnh vị trí từ {start_pos/sample_rate:.2f}s thành {adjusted_start/sample_rate:.2f}s (nhạc dẫn 3s trước lời tại {whisper_vocal_time:.2f}s)")
        
        return adjusted_start
    
    # Nếu Whisper không phát hiện được, quay lại phương pháp cũ
    write_log(file_name, "Whisper không phát hiện được lời, quay lại phương pháp phân tích âm thanh")
    
    # Tiếp tục với phương pháp phân tích âm thanh hiện tại...
    # ... existing detection code ...

def calculate_audio_features(y, sr, file_name):
    """Tính toán các đặc trưng âm thanh của một file"""
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
    tempo = float(tempo)  # Chuyển đổi từ numpy.ndarray sang Python float
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
    
    return {
        "rms": rms,
        "spectral_centroid": spectral_centroid,
        "contrast": contrast,
        "onset_env": onset_env,
        "tempo": tempo,
        "beat_strength": beat_strength,
        "y_harmonic": y_harmonic,
        "y_percussive": y_percussive,
        "harmonic_strength": harmonic_strength,
        "percussive_strength": percussive_strength,
        "mfcc_dynamic": mfcc_dynamic
    }

def find_peaks_in_audio(climax_feature, sr, hop_length, file_name):
    """Tìm các đỉnh (peaks) trong đặc trưng cao trào"""
    try:
        from scipy.signal import find_peaks
        
        # Đảm bảo distance là số nguyên dương
        min_peak_distance = max(1, int(10 * sr / hop_length))  # Giảm xuống 10 giây
        peaks, _ = find_peaks(climax_feature, 
                              height=np.mean(climax_feature) + 0.3*np.std(climax_feature),  # Giảm ngưỡng
                              distance=min_peak_distance)
        
        write_log(file_name, f"Đã phát hiện {len(peaks)} cao trào tiềm năng")
        return peaks
    except Exception as e:
        write_log(file_name, f"Lỗi khi tìm peaks: {str(e)}, sử dụng phương pháp thay thế")
        # Phương pháp thay thế: lấy các điểm có giá trị cao nhất
        num_peaks = 10  # Số lượng đỉnh cần tìm
        peak_indices = np.argsort(climax_feature)[-num_peaks:]
        peaks = sorted(peak_indices)
        write_log(file_name, f"Đã phát hiện {len(peaks)} cao trào tiềm năng bằng phương pháp thay thế")
        return peaks

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
        
        # Tính toán nhiều đặc trưng âm thanh - sử dụng hàm đã tách
        features = calculate_audio_features(y, sr, file_name)
        
        # Giải nén các đặc trưng
        rms = features["rms"]
        spectral_centroid = features["spectral_centroid"]
        contrast = features["contrast"]
        onset_env = features["onset_env"]
        tempo = features["tempo"]
        beat_strength = features["beat_strength"]
        y_harmonic = features["y_harmonic"]
        y_percussive = features["y_percussive"]
        harmonic_strength = features["harmonic_strength"]
        percussive_strength = features["percussive_strength"]
        
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
        
        # Sử dụng kết hợp chroma và MFCC để nhận diện cấu trúc
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        mfcc_chorus = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Tính ma trận tự tương quan với trọng số cao hơn
        similarity_matrix = librosa.segment.recurrence_matrix(
            chroma, width=int(segment_duration * sr / hop_length), mode='affinity'
        )
        
        # Phát hiện đoạn lặp lại nhiều lần (đặc trưng của chorus)
        repeat_structure = np.sum(similarity_matrix, axis=1)
        
        # Kết hợp với năng lượng để nhận diện chorus tốt hơn
        chorus_scores = (
            repeat_structure * 0.7 +  # Tăng trọng số cho cấu trúc lặp lại
            rms[:len(repeat_structure)] * 0.3  # Kết hợp với năng lượng
        )
        
        # Chuẩn hóa chorus_scores
        chorus_scores = (chorus_scores - np.min(chorus_scores)) / (np.max(chorus_scores) - np.min(chorus_scores))
        
        # THAY ĐỔI QUAN TRỌNG: Xử lý tham số đầu vào từ UI
        # Luôn có 1 đoạn main và số đoạn chorus phụ thuộc vào num_segments
        num_main = 1
        num_chorus = max(1, num_segments - num_main)
        
        write_log(file_name, f"Sẽ tạo {num_main} đoạn main và {num_chorus} đoạn chorus (tổng {num_segments} đoạn)")
        
        # Khởi tạo các danh sách trước khi sử dụng
        main_segments = []
        chorus_segments = []
        
        # Sửa lỗi kết hợp chorus_scores và combined_feature khi kích thước khác nhau
        if len(chorus_scores) < len(combined_feature):
            chorus_scores_resampled = np.interp(
                np.linspace(0, 1, len(combined_feature)),
                np.linspace(0, 1, len(chorus_scores)),
                chorus_scores
            )
            combined_score = combined_feature * 0.7 + chorus_scores_resampled * 0.3
            write_log(file_name, f"Đã nội suy chorus_scores từ {len(chorus_scores)} lên {len(combined_feature)}")
        else:
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
        
        # Sắp xếp theo điểm số giảm dần
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # THAY ĐỔI: Bắt đầu với phát hiện cao trào (cho main) trước
        # Phát hiện cao trào trong bài hát
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
        window_length = min(51, len(climax_feature) - 2)
        if window_length % 2 == 0:
            window_length -= 1
        
        if window_length >= 3:  # Savgol filter yêu cầu window_length >= 3
            try:
                climax_feature = signal.savgol_filter(climax_feature, window_length, 3)
            except Exception as e:
                write_log(file_name, f"Lỗi khi áp dụng savgol_filter: {str(e)}, sử dụng dữ liệu gốc")
        
        # Tìm các đỉnh cục bộ (local maxima) - đây là các cao trào tiềm năng
        peaks = find_peaks_in_audio(climax_feature, sr, hop_length, file_name)
        
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
                    peak_value = float(climax_feature[peak])
                    peak_samples.append((sample_pos, peak_value))
        
        # Sắp xếp các cao trào theo độ mạnh giảm dần
        peak_samples.sort(key=lambda x: x[1], reverse=True)
        write_log(file_name, f"Đã sắp xếp {len(peak_samples)} cao trào theo độ mạnh")
        
        # Tìm đoạn cao trào (main) - lấy đoạn có điểm cao nhất
        if len(peak_samples) > 0:
            # THAY ĐỔI: Luôn sử dụng đỉnh cao trào mạnh nhất cho main
            main_start_sample, main_score = peak_samples[0]
            
            # THAY ĐỔI: Đảm bảo main không bị lùi quá nhiều
            # Tìm điểm cao độ cao nhất và điều chỉnh vị trí bắt đầu
            adjusted_start = find_highest_pitch_point(main_start_sample, y, sr, is_main=True)
            
            # Phát hiện giọng hát và điều chỉnh vị trí - truyền tham số is_main=True
            vocal_start = detect_instrumental_intro(adjusted_start, y, sr, is_main=True)
            
            # Thêm vào danh sách main
            main_segments.append((vocal_start, float(main_score), "main"))
            write_log(file_name, f"MAIN: Đã chọn đoạn cao trào mạnh nhất tại {vocal_start/sr:.2f}s với điểm số {float(main_score):.4f}")
        else:
            # Nếu không tìm thấy cao trào, sử dụng đoạn có điểm tổng hợp cao nhất
            main_start_sample, main_score = scores[0]
            adjusted_start = find_highest_pitch_point(main_start_sample, y, sr, is_main=True)
            vocal_start = detect_instrumental_intro(adjusted_start, y, sr, is_main=True)
            main_segments.append((vocal_start, float(main_score), "main"))
            write_log(file_name, f"MAIN (thay thế): Đã chọn đoạn tại {vocal_start/sr:.2f}s với điểm số {float(main_score):.4f}")
        
        # Đảm bảo khoảng cách giữa các đoạn
        min_distance_samples = int(segment_duration * sr * 2.5)  # Tăng khoảng cách từ 1.2 lên 2.5 lần segment_duration
        
        # Tìm các đoạn chorus dựa trên chorus_specific_scores
        chorus_specific_scores = []
        for i in range(0, len(y) - frame_length, hop_length):
            if i < len(chorus_scores):
                segment = chorus_scores[i:i+frame_length] if i+frame_length <= len(chorus_scores) else chorus_scores[i:]
                if len(segment) > 0:
                    mean_score = np.mean(segment)
                    variation_score = np.std(segment) * 0.2
                    final_score = mean_score + variation_score
                    # KHÔNG CHỌN ĐOẠN ĐẦU BÀI làm chorus để tránh intro dài
                    if i > 30 * sr:  # Bỏ qua 30 giây đầu để tránh intro
                        chorus_specific_scores.append((i, float(final_score)))
        
        # Sắp xếp theo điểm số giảm dần
        if chorus_specific_scores:
            chorus_specific_scores.sort(key=lambda x: x[1], reverse=True)
            
            # THAY ĐỔI: Đảm bảo chorus khác với main bằng cách ưu tiên lấy từ nửa sau của bài
            # Lấy vị trí của đoạn main
            main_position = 0
            if main_segments:
                main_position = main_segments[0][0]
            main_half_point = len(y) // 2
            
            # Ưu tiên chorus từ nửa đối diện với main
            prioritized_scores = []
            other_scores = []
            
            for start_sample, score in chorus_specific_scores:
                # Xác định nửa bài hát
                is_in_second_half = start_sample > main_half_point
                is_in_same_half_as_main = (main_position > main_half_point) == is_in_second_half
                
                # Nếu main ở nửa đầu, ưu tiên chorus ở nửa sau và ngược lại
                if not is_in_same_half_as_main:
                    prioritized_scores.append((start_sample, score))
                else:
                    other_scores.append((start_sample, score))
            
            # Kết hợp lại, ưu tiên các đoạn ở nửa đối diện
            chorus_candidates = prioritized_scores + other_scores
            
            # Tìm các đoạn chorus có điểm cao và đủ xa đoạn main
            for chorus_start_sample, chorus_score in chorus_candidates:
                # Kiểm tra khoảng cách với các đoạn đã chọn
                is_far_from_existing = True
                for existing_start, _, _ in (main_segments + chorus_segments):
                    if abs(chorus_start_sample - existing_start) < min_distance_samples:
                        is_far_from_existing = False
                        break
                
                if is_far_from_existing:
                    # THAY ĐỔI: Thêm kiểm tra trùng lặp với main
                    if main_segments and abs(chorus_start_sample - main_segments[0][0]) < min_distance_samples:
                        write_log(file_name, f"Bỏ qua đoạn chorus tại {chorus_start_sample/sr:.2f}s vì quá gần với main")
                        continue
                    
                    # Tìm điểm cao độ cao nhất và điều chỉnh vị trí bắt đầu
                    adjusted_start = find_highest_pitch_point(chorus_start_sample, y, sr, is_main=False)
                    
                    # THAY ĐỔI: Phát hiện giọng hát và điều chỉnh vị trí với is_main=False
                    vocal_start = detect_instrumental_intro(adjusted_start, y, sr, is_main=False)
                    
                    # BỎ QUA ĐOẠN KHÔNG CÓ LỜI
                    if vocal_start == -1:
                        write_log(file_name, f"Bỏ qua đoạn tại {adjusted_start/sr:.2f}s vì không phát hiện giọng hát")
                        continue
                    
                    # THAY ĐỔI: Kiểm tra lại sau khi điều chỉnh vị trí
                    is_still_far = True
                    for existing_start, _, _ in (main_segments + chorus_segments):
                        if abs(vocal_start - existing_start) < min_distance_samples:
                            is_still_far = False
                            break
                    
                    if not is_still_far:
                        write_log(file_name, f"Bỏ qua đoạn tại {vocal_start/sr:.2f}s vì sau khi điều chỉnh quá gần với đoạn khác")
                        continue
                    
                    # Thêm vào danh sách chorus
                    chorus_segments.append((vocal_start, float(chorus_score), "chorus"))
                    write_log(file_name, f"CHORUS: Đã thêm đoạn tại {vocal_start/sr:.2f}s với điểm số {float(chorus_score):.4f}")
                    
                    # Dừng khi đã đủ số lượng chorus
                    if len(chorus_segments) >= num_chorus:
                        break
        
        # Nếu chưa đủ số lượng chorus, thêm từ danh sách scores
        if len(chorus_segments) < num_chorus:
            write_log(file_name, f"Chưa đủ số chorus ({len(chorus_segments)}/{num_chorus}), tìm thêm từ danh sách scores")
            
            # Bỏ qua phần tử đầu tiên vì đã dùng cho main
            for start_sample, score in scores[1:]:
                # Kiểm tra khoảng cách
                is_far_from_existing = True
                for existing_start, _, _ in (main_segments + chorus_segments):
                    if abs(start_sample - existing_start) < min_distance_samples:
                        is_far_from_existing = False
                        break
                
                if is_far_from_existing:
                    adjusted_start = find_highest_pitch_point(start_sample, y, sr, is_main=False)
                    vocal_start = detect_instrumental_intro(adjusted_start, y, sr, is_main=False)
                    chorus_segments.append((vocal_start, float(score), "chorus"))
                    write_log(file_name, f"Đã thêm đoạn chorus bổ sung tại {vocal_start/sr:.2f}s")
                    
                    if len(chorus_segments) >= num_chorus:
                        break
        
        # Kết hợp main và chorus segments
        top_segments = main_segments + chorus_segments
        
        # Sắp xếp theo thời gian để dễ theo dõi
        top_segments.sort(key=lambda x: x[0])
        
        write_log(file_name, f"Đã chọn tổng cộng {len(top_segments)} đoạn highlight: {len(main_segments)} main, {len(chorus_segments)} chorus")
        
        highlights = []
        write_log(file_name, "Đang xuất các đoạn highlight...")
        
        # Lấy tên file gốc không có phần mở rộng
        base_filename = os.path.splitext(os.path.basename(file_name))[0]
        
        # Xử lý tên file để bỏ dấu nhưng giữ nguyên ký tự
        if any(ord(c) > 127 for c in base_filename):
            base_filename_ascii = unicodedata.normalize('NFKD', base_filename)
            base_filename_ascii = ''.join([c for c in base_filename_ascii if not unicodedata.combining(c)])
            write_log(file_name, f"Đã chuyển đổi tên file có dấu '{base_filename}' thành '{base_filename_ascii}'")
            base_filename = base_filename_ascii
        
        for idx, (start_sample, score, segment_type) in enumerate(top_segments):
            start_time_hl = start_sample / sr
            
            audio_segment = AudioSegment.from_file(audio_file_path)
            start_ms = int(start_time_hl * 1000)
            end_ms = min(len(audio_segment), start_ms + int(segment_duration * 1000))
            
            # THAY ĐỔI: Đảm bảo segment_duration đúng cho cả main và chorus
            highlight_audio = audio_segment[start_ms:end_ms]
            write_log(file_name, f"Xuất {segment_type} từ {start_time_hl:.2f}s, độ dài {(end_ms-start_ms)/1000:.2f}s")
            
            # THAY ĐỔI: Kiểm tra độ dài trước khi xử lý
            if len(highlight_audio) < 8000:  # Nếu đoạn ngắn hơn 8 giây
                write_log(file_name, f"Cảnh báo: Đoạn {segment_type} quá ngắn ({len(highlight_audio)/1000:.2f}s), điều chỉnh...")
                # Mở rộng đoạn nếu có thể
                extended_end = min(len(audio_segment), start_ms + int(segment_duration * 1000))
                highlight_audio = audio_segment[start_ms:extended_end]
            
            # Chuyển đổi sang định dạng yêu cầu: 8000Hz, mono
            highlight_audio = highlight_audio.set_frame_rate(8000).set_channels(1)
            
            # THAY ĐỔI: Ghi log độ dài sau khi chuyển đổi
            write_log(file_name, f"Sau khi chuyển đổi: độ dài = {len(highlight_audio)/1000:.2f}s")
            
            # Tạo tên file theo yêu cầu: tên file gốc + số thứ tự
            highlight_output_path = os.path.join(output_dir, f"{base_filename} {segment_type} {idx+1}.mp3")
            
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
            
            write_log(file_name, f"Đã xuất highlight {idx+1} ({segment_type}) tại {start_time_hl:.2f}s vào file {highlight_output_path}, độ dài cuối: {len(highlight_audio)/1000:.2f}s")
            
            # Lưu thông tin highlight vào database
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