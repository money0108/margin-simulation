@echo off
chcp 65001 >nul
title 遠期契約保證金模擬平台

echo ========================================
echo   遠期契約保證金模擬平台 啟動中...
echo ========================================
echo.

cd /d "%~dp0"

REM 檢查 Python 是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo [錯誤] 找不到 Python，請先安裝 Python
    echo 下載網址: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 檢查 streamlit 是否已安裝
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [提示] 尚未安裝必要套件，正在安裝中...
    echo.
    pip install -r requirements.txt
    echo.
    echo 套件安裝完成！
    echo.
)

echo 正在啟動 Streamlit 伺服器...
echo 請稍候，瀏覽器將自動開啟...
echo.
echo 按 Ctrl+C 可停止伺服器
echo ========================================

python -m streamlit run app.py

pause
