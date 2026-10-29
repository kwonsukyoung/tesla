#!/bin/bash
cd "$(dirname "$0")"

echo "========================================================"
echo " 🚗 블랙박스 범인 잡기 - 로컬 분석 서버 준비 중..."
echo "========================================================"

if [ ! -d "venv" ]; then
  echo "📦 Python 가상환경(venv) 생성 중..."
  python3 -m venv venv
fi

source venv/bin/activate

echo "📦 필수 패키지 확인 및 설치 중..."
pip install --upgrade pip -q
pip install -r requirements.txt

echo "🚀 서버를 시작합니다 (브라우저가 자동으로 열립니다)..."
python3 app.py
