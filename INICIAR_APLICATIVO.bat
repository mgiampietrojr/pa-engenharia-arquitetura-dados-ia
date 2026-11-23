@echo off
chcp 65001 >nul
setlocal
pushd "%~dp0"
set "APP_PYTHON=%~dp0..\..\.venv_sprint3\Scripts\python.exe"
if exist "%APP_PYTHON%" goto executar
set "APP_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%APP_PYTHON%" goto dependencias
where py >nul 2>nul
if errorlevel 1 (python -m venv .venv) else (py -3.11 -m venv .venv)
if errorlevel 1 goto falha
:dependencias
if exist ".venv\produto_instalado.ok" goto executar
"%APP_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto falha
echo instalado>".venv\produto_instalado.ok"
:executar
"%APP_PYTHON%" -m streamlit run app_streamlit.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
popd
exit /b 0
:falha
echo Nao foi possivel preparar o Python 3.11. Consulte LEIA_APLICATIVO.txt.
pause
popd
exit /b 1
