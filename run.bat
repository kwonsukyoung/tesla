@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================================
echo  🚗 블랙박스 범인 잡기 - 로컬 분석 서버 준비 중...
echo ========================================================

if not exist "venv" (
    echo 📦 Python 가상환경 생성 중...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo 📦 필수 패키지 확인 및 설치 중...
pip install -r requirements.txt

echo 🚀 서버를 시작합니다...
python app.py
pause
