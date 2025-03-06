import streamlit as st
import requests
import pandas as pd
import time
import os

st.set_page_config(page_title="Audio Highlight Extractor", layout="wide")

# Tạo sidebar để chọn chức năng
st.sidebar.title("Menu")
page = st.sidebar.radio("Chọn chức năng:", ["Xử lý Audio", "Lịch sử xử lý"])

if page == "Xử lý Audio":
    st.title("Audio Highlight Extractor")

    uploaded_files = st.file_uploader("Upload your audio files", type=["mp3", "wav"], accept_multiple_files=True)
    num_segments = st.number_input("Number of highlight segments", min_value=1, max_value=10, value=1, step=1)
    segment_duration = st.slider("Segment Duration (seconds)", min_value=1.0, max_value=15.0, value=10.0, step=0.5)

    if uploaded_files:
        # Hiển thị số lượng file đã upload
        st.info(f"Đã tải lên {len(uploaded_files)} file audio")
        
        # Thêm nút bấm để bắt đầu xử lý
        process_button = st.button("Bắt đầu xử lý")
        
        if process_button:
            # Tạo tab cho từng file
            tabs = st.tabs([file.name for file in uploaded_files])
            
            for i, uploaded_file in enumerate(uploaded_files):
                with tabs[i]:
                    with st.spinner(f"Đang xử lý file {uploaded_file.name}..."):
                        files = {
                            "audio_file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                        }

                        response = requests.post(
                            f"http://127.0.0.1:8000/highlight?num_segments={num_segments}&segment_duration={segment_duration}", 
                            files=files
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
                                st.subheader(f"Highlight {j+1}")
                                st.info(f"Time: {highlight['highlight_time']} seconds")
                                
                                # Đọc file highlight từ backend để hiển thị
                                highlight_file = highlight['highlight_file']
                                audio_response = requests.get(f"http://127.0.0.1:8000/download_highlight?file_path={highlight_file}")
                                if audio_response.status_code == 200:
                                    st.audio(audio_response.content, format="audio/mp3")
                                else:
                                    st.error(f"Could not retrieve highlight audio file {j+1}.")
                        else:
                            st.error(f"Error processing audio: {response.text}")

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
                                    st.write(f"**Highlight {highlight['highlight_index']}** - Thời điểm: {highlight['highlight_time']} giây")
                                    
                                    # Hiển thị audio
                                    audio_response = requests.get(f"http://127.0.0.1:8000/download_highlight?file_path={highlight['highlight_file']}")
                                    if audio_response.status_code == 200:
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
