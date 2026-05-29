@echo off
setlocal

cd /d "%~dp0"

echo.
echo ==========================================
echo  Programas presidenciales Colombia 2026
echo ==========================================
echo.

set "PYTHON_EXE="

if exist ".venv\Scripts\python.exe" (
  set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
  where py >nul 2>nul
  if %ERRORLEVEL%==0 (
    set "PYTHON_EXE=py"
  ) else (
    where python >nul 2>nul
    if %ERRORLEVEL%==0 (
      set "PYTHON_EXE=python"
    )
  )
)

if "%PYTHON_EXE%"=="" (
  echo No encontre Python en este equipo.
  echo Instala Python 3.11 o superior y vuelve a ejecutar este archivo.
  echo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creando entorno virtual .venv...
  %PYTHON_EXE% -m venv .venv
  if errorlevel 1 (
    echo.
    echo No pude crear el entorno virtual.
    pause
    exit /b 1
  )
)

set "PYTHON_EXE=.venv\Scripts\python.exe"
set "STREAMLIT_EXE=.venv\Scripts\streamlit.exe"

echo Verificando dependencias...
%PYTHON_EXE% -c "import streamlit, pandas, numpy, pypdf, sentence_transformers, sklearn, plotly, umap, hdbscan, pyarrow" >nul 2>nul
if errorlevel 1 (
  echo Instalando dependencias. Esto puede tardar varios minutos la primera vez...
  %PYTHON_EXE% -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Fallo la instalacion de dependencias.
    echo Revisa tu conexion a internet o ejecuta manualmente:
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
  )
)

if not exist "data\artifacts" (
  mkdir "data\artifacts" >nul 2>nul
)

dir /b "data\artifacts\diagnostics_*.json" >nul 2>nul
if errorlevel 1 (
  echo Generando analisis semantico inicial. Esto puede tardar varios minutos...
  %PYTHON_EXE% scripts\build_artifacts.py
  if errorlevel 1 (
    echo.
    echo Fallo la generacion de artefactos.
    pause
    exit /b 1
  )
)

echo.
echo Abriendo la app en http://localhost:8501
echo Para cerrar la app, cierra esta ventana o presiona Ctrl+C.
echo.

start "" "http://localhost:8501"
%STREAMLIT_EXE% run app.py --server.port 8501

endlocal
