FROM python:3.9-slim

WORKDIR /app

# Cài đặt các gói phụ thuộc hệ thống
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    sqlite3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Sao chép requirements trước để tận dụng cache của Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn ứng dụng
COPY . .

# Tạo thư mục cần thiết
RUN mkdir -p output logs

# Expose cổng cho FastAPI và Streamlit
EXPOSE 8000 8501

# Tạo script khởi động
RUN echo '#!/bin/bash\n\
uvicorn audio_api:app --host 0.0.0.0 --port 8000 & \n\
streamlit run audio_ui.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"] 