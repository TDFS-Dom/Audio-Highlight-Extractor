@echo off
echo ===================================================
echo    Audio Highlight Extractor - Cai dat va Chay
echo ===================================================
echo.

REM Kiem tra Python da duoc cai dat chua
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python chua duoc cai dat. Vui long cai dat Python tu python.org
    echo Truy cap: https://www.python.org/downloads/windows/
    echo Dam bao tich chon "Add Python to PATH" khi cai dat
    pause
    exit /b
)

echo Python da duoc cai dat. Dang tien hanh cai dat cac thu vien can thiet...
echo.

REM Tao thu muc output neu chua ton tai
if not exist output mkdir output

REM Cai dat cac thu vien can thiet
echo Dang cai dat cac thu vien can thiet...
pip install streamlit fastapi uvicorn librosa numpy pydub python-multipart
if %errorlevel% neq 0 (
    echo Loi khi cai dat cac thu vien. Vui long kiem tra ket noi mang hoac thu lai sau.
    pause
    exit /b
)

echo.
echo Cai dat hoan tat!
echo.

REM Khoi dong backend API trong cua so moi
echo Dang khoi dong Backend API...
start cmd /k "title Backend API && uvicorn audio_api:app --reload"

REM Doi 5 giay de backend khoi dong
echo Doi backend khoi dong...
timeout /t 5 /nobreak > nul

REM Khoi dong frontend UI
echo Dang khoi dong Frontend UI...
start cmd /k "title Frontend UI && streamlit run audio_ui.py"

echo.
echo ===================================================
echo    Ung dung da duoc khoi dong thanh cong!
echo.
echo    Backend API: http://127.0.0.1:8000
echo    Frontend UI: http://localhost:8501
echo.
echo    Luu y: Khong dong cua so nay khi dang su dung ung dung
echo ===================================================
echo.

pause 