@echo off
cd /d "%~dp0"
C:\Users\gesit\miniconda3\envs\ai\python.exe -m jupyter notebook --no-browser --NotebookApp.token=''
pause
