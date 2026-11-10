@echo off
setlocal
set "TORCHINDUCTOR_CACHE_DIR=%~dp0..\training_env\runtime-cache\torch"
set "CUDA_CACHE_PATH=%~dp0..\training_env\runtime-cache\cuda"
set "MPLCONFIGDIR=%~dp0..\training_env\runtime-cache\matplotlib"
set "PYTHONDONTWRITEBYTECODE=1"
pushd "%~dp0"
call "%~dp0..\training_env\python.cmd" %*
set "TASK_RC=%errorlevel%"
popd
exit /b %TASK_RC%
