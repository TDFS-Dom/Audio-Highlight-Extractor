import streamlit as st
import requests
import pandas as pd
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import io
import librosa
import json

st.set_page_config(page_title="Audio Highlight Extractor", layout="wide")

# Tạo sidebar để chọn chức năng
st.sidebar.title("Menu")
page = st.sidebar.radio("Chọn chức năng:", ["Xử lý Audio", "Lịch sử xử lý"])

if page == "Xử lý Audio":
    st.title("Audio Highlight Extractor")

    uploaded_files = st.file_uploader("Upload your audio files", type=["mp3", "wav", "flac", "m4a", "aac", "ogg", "wma", "aiff"],, accept_multiple_files=True)
    
    # Display waveform visualization for uploaded files
    if uploaded_files:
        # Hiển thị số lượng file đã upload
        st.info(f"Đã tải lên {len(uploaded_files)} file audio")
        
        # Create tabs for each uploaded file to show waveforms
        preview_tabs = st.tabs([f"Preview: {file.name}" for file in uploaded_files])
        
        for i, uploaded_file in enumerate(uploaded_files):
            with preview_tabs[i]:
                try:
                    # Read audio data for visualization
                    audio_bytes = uploaded_file.getvalue()
                    audio_data = io.BytesIO(audio_bytes)
                    
                    # Load audio with librosa
                    y, sr = librosa.load(audio_data, sr=None)
                    
                    # Create waveform visualization
                    fig, ax = plt.subplots(figsize=(12, 3))
                    ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='#3366FF')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Amplitude')
                    ax.set_title(f'Waveform - {uploaded_file.name}')
                    ax.grid(True, alpha=0.3)
                    
                    # Add time markers
                    total_duration = len(y)/sr
                    time_markers = np.linspace(0, total_duration, 10)
                    ax.set_xticks(time_markers)
                    ax.set_xticklabels([f"{t:.1f}" for t in time_markers])
                    
                    st.pyplot(fig)
                    
                    # Display audio player for the uploaded file
                    st.audio(audio_bytes, format=f"audio/{uploaded_file.type.split('/')[-1]}")
                    
                    # Show audio information
                    st.info(f"Duration: {total_duration:.2f} seconds | Sample rate: {sr} Hz")
                except Exception as e:
                    st.warning(f"Không thể hiển thị waveform cho file {uploaded_file.name}: {str(e)}")
    
    # Adjusted segment_duration slider to match the 40-60 second requirement
    segment_duration = st.slider("Segment Duration (seconds)", min_value=40.0, max_value=60.0, value=45.0, step=5.0)

    # Add this where you're collecting user input parameters
    num_segments = st.number_input(
        "Số đoạn highlight (Number of highlight segments)",
        min_value=1,
        max_value=5,  # Setting a reasonable maximum
        value=2,  # Default value
        step=1,
        help="Chọn số lượng đoạn highlight bạn muốn trích xuất từ mỗi file âm thanh"
    )

    if uploaded_files:
        # Thêm nút bấm để bắt đầu xử lý
        process_button = st.button("Bắt đầu xử lý")
        
        if process_button:
            # Thêm option để chọn phương thức xử lý
            use_batch = st.checkbox("Xử lý đồng thời tất cả file (nhanh hơn)", value=True)
            
            if use_batch and len(uploaded_files) > 1:
                # Xử lý tất cả file cùng lúc bằng endpoint highlight_multiple
                with st.spinner(f"Đang xử lý {len(uploaded_files)} file audio..."):
                    files = []
                    for uploaded_file in uploaded_files:
                        files.append(
                            ("audio_files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))
                        )
                    
                    response = requests.post(
                        f"http://127.0.0.1:8000/highlight_multiple?segment_duration={segment_duration}&num_segments={num_segments}", 
                        files=files
                    )
                    
                    if response.status_code == 200:
                        results = response.json()["results"]
                        
                        # Tạo tab cho từng file
                        tabs = st.tabs([result["filename"] for result in results])
                        
                        for i, result in enumerate(results):
                            with tabs[i]:
                                file_result = result["result"]
                                
                                # Kiểm tra nếu có lỗi
                                if "error" in file_result:
                                    st.error(f"Lỗi: {file_result['error']}")
                                    if "job_id" in file_result:
                                        st.info(f"ID công việc: {file_result['job_id']} - Bạn có thể kiểm tra chi tiết lỗi trong phần Lịch sử xử lý")
                                    continue
                                
                                st.success(f"Found {file_result['num_segments']} highlight segments")
                                
                                # Hiển thị job ID
                                if "job_id" in file_result:
                                    st.info(f"ID công việc: {file_result['job_id']}")
                                
                                for j, highlight in enumerate(file_result['highlights']):
                                    # Add segment type to the subheader if available
                                    segment_type = highlight.get('type', '')
                                    type_label = f" ({segment_type.upper()})" if segment_type else ""
                                    st.subheader(f"Highlight {j+1}{type_label}")
                                    st.info(f"Time: {highlight['highlight_time']} seconds")
                                    
                                    # Đọc file highlight từ backend để hiển thị
                                    highlight_file = highlight['highlight_file']
                                    audio_response = requests.get(f"http://127.0.0.1:8000/download_highlight?file_path={highlight_file}")
                                    
                                    # Hiển thị audio tương tự như xử lý đơn lẻ
                                    if audio_response.status_code == 200:
                                        # Tạo visualizer cho audio highlight
                                        audio_bytes = audio_response.content
                                        try:
                                            # Đọc audio data để tạo visualizer
                                            audio_data = io.BytesIO(audio_bytes)
                                            y, sr = librosa.load(audio_data, sr=None)
                                            
                                            # Tạo waveform visualization
                                            fig, ax = plt.subplots(figsize=(10, 2))
                                            ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='#1DB954')
                                            ax.set_xlabel('Time (s)')
                                            ax.set_ylabel('Amplitude')
                                            ax.set_title(f'Waveform - Highlight at {highlight["highlight_time"]}s')
                                            ax.grid(True, alpha=0.3)
                                            st.pyplot(fig)
                                        except Exception as e:
                                            st.warning(f"Không thể tạo visualizer: {str(e)}")
                                        
                                        # Sử dụng định dạng MP3 cho audio player
                                        st.audio(audio_bytes, format="audio/mp3")
                                    else:
                                        st.error(f"Could not retrieve highlight audio file {j+1}.")
                    else:
                        st.error(f"Error processing audio batch: {response.text}")
            
            else:
                # Tạo tab cho từng file
                tabs = st.tabs([file.name for file in uploaded_files])
                
                for i, uploaded_file in enumerate(uploaded_files):
                    with tabs[i]:
                        with st.spinner(f"Đang xử lý file {uploaded_file.name}..."):
                            files = {
                                "audio_file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                            }

                            try:
                                response = requests.post(
                                    f"http://127.0.0.1:8000/highlight?segment_duration={segment_duration}&num_segments={num_segments}", 
                                    files=files,
                                    timeout=300  # 5 minute timeout
                                )
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    
                                    # Kiểm tra nếu có lỗi
                                    if "error" in result:
                                        st.error(f"Lỗi: {result['error']}")
                                        if "job_id" in result:
                                            st.info(f"ID công việc: {result['job_id']} - Bạn có thể kiểm tra chi tiết lỗi trong phần Lịch sử xử lý")
                                        continue
                                    
                                    st.success(f"Found {result['num_segments']} highlight segments")
                                    
                                    # Hiển thị job ID
                                    if "job_id" in result:
                                        st.info(f"ID công việc: {result['job_id']}")
                                    
                                    for j, highlight in enumerate(result['highlights']):
                                        # Add segment type to the subheader if available
                                        segment_type = highlight.get('type', '')
                                        type_label = f" ({segment_type.upper()})" if segment_type else ""
                                        st.subheader(f"Highlight {j+1}{type_label}")
                                        st.info(f"Time: {highlight['highlight_time']} seconds")
                                        
                                        # Đọc file highlight từ backend để hiển thị
                                        highlight_file = highlight['highlight_file']
                                        audio_response = requests.get(f"http://127.0.0.1:8000/download_highlight?file_path={highlight_file}")
                                        if audio_response.status_code == 200:
                                            # Tạo visualizer cho audio highlight
                                            audio_bytes = audio_response.content
                                            try:
                                                # Đọc audio data để tạo visualizer
                                                audio_data = io.BytesIO(audio_bytes)
                                                y, sr = librosa.load(audio_data, sr=None)
                                                
                                                # Tạo waveform visualization
                                                fig, ax = plt.subplots(figsize=(10, 2))
                                                ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='#1DB954')
                                                ax.set_xlabel('Time (s)')
                                                ax.set_ylabel('Amplitude')
                                                ax.set_title(f'Waveform - Highlight at {highlight["highlight_time"]}s')
                                                ax.grid(True, alpha=0.3)
                                                st.pyplot(fig)
                                            except Exception as e:
                                                st.warning(f"Không thể tạo visualizer: {str(e)}")
                                            
                                            # Sử dụng định dạng MP3 cho audio player
                                            st.audio(audio_bytes, format="audio/mp3")
                                        else:
                                            st.error(f"Could not retrieve highlight audio file {j+1}.")
                                else:
                                    try:
                                        # Try to parse error message as JSON
                                        error_detail = response.json()
                                        st.error(f"Error processing audio: {error_detail.get('detail', response.text)}")
                                    except json.JSONDecodeError:
                                        # If not JSON, display the raw response
                                        st.error(f"Error processing audio: {response.text}")
                                    
                                    # Add more debug info
                                    st.info("Debug information: Check the server logs for more details")
                                    st.code(f"Status code: {response.status_code}\nRequest URL: {response.url}")
                            except requests.exceptions.RequestException as e:
                                st.error(f"Lỗi kết nối đến server: {str(e)}")
                                st.info("Vui lòng kiểm tra xem server API đã khởi động chưa hoặc có lỗi trong quá trình xử lý")

elif page == "Lịch sử xử lý":
    st.title("Lịch sử xử lý Audio")
    
    # Tạo nút refresh
    if st.button("Làm mới dữ liệu"):
        st.experimental_rerun()
    
    # Lấy danh sách các job đã xử lý
    try:
        response = requests.get("http://127.0.0.1:8000/jobs")
        if response.status_code == 200:
            jobs = response.json()["jobs"]
            
            if not jobs:
                st.info("Chưa có file audio nào được xử lý")
            else:
                # Tạo DataFrame để hiển thị dạng bảng
                jobs_df = pd.DataFrame(jobs)
                jobs_df = jobs_df.rename(columns={
                    "job_id": "ID",
                    "file_name": "Tên file",
                    "status": "Trạng thái",
                    "start_time": "Thời gian bắt đầu",
                    "end_time": "Thời gian kết thúc",
                    "num_segments": "Số đoạn highlight",
                    "segment_duration": "Thời lượng đoạn (giây)"
                })
                
                # Thêm cột màu cho trạng thái
                def highlight_status(val):
                    if val == "COMPLETED":
                        return 'background-color: #d4edda; color: #155724'
                    elif val == "PROCESSING":
                        return 'background-color: #fff3cd; color: #856404'
                    elif val == "FAILED":
                        return 'background-color: #f8d7da; color: #721c24'
                    return ''
                
                # Hiển thị bảng với màu sắc
                st.dataframe(jobs_df.style.applymap(highlight_status, subset=['Trạng thái']))
                
                # Chọn job để xem chi tiết
                selected_job_id = st.selectbox("Chọn ID công việc để xem chi tiết:", 
                                              options=[job["job_id"] for job in jobs],
                                              format_func=lambda x: f"ID: {x} - {next((job['file_name'] for job in jobs if job['job_id'] == x), '')}")
                
                if selected_job_id:
                    job_response = requests.get(f"http://127.0.0.1:8000/job/{selected_job_id}")
                    if job_response.status_code == 200:
                        job_detail = job_response.json()
                        
                        # Hiển thị thông tin chi tiết
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Thông tin công việc")
                            st.write(f"**Tên file:** {job_detail['file_name']}")
                            st.write(f"**Trạng thái:** {job_detail['status']}")
                            st.write(f"**Thời gian bắt đầu:** {job_detail['start_time']}")
                            st.write(f"**Thời gian kết thúc:** {job_detail['end_time'] or 'N/A'}")
                            st.write(f"**Số đoạn highlight:** {job_detail['num_segments']}")
                            st.write(f"**Thời lượng đoạn:** {job_detail['segment_duration']} giây")
                            
                            if job_detail['error_message']:
                                st.error(f"**Lỗi:** {job_detail['error_message']}")
                        
                        with col2:
                            st.subheader("Các đoạn highlight")
                            if job_detail['status'] == "COMPLETED" and job_detail['highlights']:
                                for highlight in job_detail['highlights']:
                                    # Add segment type to the subheader if available
                                    segment_type = highlight.get('type', '')
                                    type_label = f" ({segment_type.upper()})" if segment_type else ""
                                    st.write(f"**Highlight {highlight['highlight_index']}** - Thời điểm: {highlight['highlight_time']} giây")
                                    
                                    # Hiển thị audio
                                    audio_response = requests.get(f"http://127.0.0.1:8000/download_highlight?file_path={highlight['highlight_file']}")
                                    if audio_response.status_code == 200:
                                        # Tạo visualizer cho audio highlight
                                        audio_bytes = audio_response.content
                                        try:
                                            # Đọc audio data để tạo visualizer
                                            audio_data = io.BytesIO(audio_bytes)
                                            y, sr = librosa.load(audio_data, sr=None)
                                            
                                            # Tạo waveform visualization
                                            fig, ax = plt.subplots(figsize=(10, 2))
                                            ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='#1DB954')
                                            ax.set_xlabel('Time (s)')
                                            ax.set_ylabel('Amplitude')
                                            ax.set_title(f'Waveform - Highlight at {highlight["highlight_time"]}s')
                                            ax.grid(True, alpha=0.3)
                                            st.pyplot(fig)
                                        except Exception as e:
                                            st.warning(f"Không thể tạo visualizer: {str(e)}")
                                        
                                        # Sử dụng định dạng MP3 cho audio player
                                        st.audio(audio_response.content, format="audio/mp3")
                                    else:
                                        st.warning(f"Không thể tải file audio highlight {highlight['highlight_index']}")
                            elif job_detail['status'] == "PROCESSING":
                                st.info("Đang xử lý...")
                            elif job_detail['status'] == "FAILED":
                                st.error("Xử lý thất bại")
                            else:
                                st.info("Không có highlight nào")
                        
                        # Hiển thị log
                        st.subheader("Log xử lý")
                        if job_detail['log']:
                            st.code(job_detail['log'])
                        else:
                            st.info("Không có log xử lý")
                    else:
                        st.error("Không thể lấy thông tin chi tiết của công việc")
        else:
            st.error(f"Lỗi khi lấy danh sách công việc: {response.text}")
    except Exception as e:
        st.error(f"Lỗi kết nối đến server: {str(e)}")
