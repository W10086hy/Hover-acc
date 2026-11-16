@echo off
setlocal
if not defined HOVER_VM_HOST set "HOVER_VM_HOST=s@192.168.174.128"
if not defined HOVER_VM_ROOT set "HOVER_VM_ROOT=/home/s/hover_bptt_px4/acceleration_dual_20261115"
ssh -t %HOVER_VM_HOST% "cd '%HOVER_VM_ROOT%' && bash px4_offboard/launch_acceleration.sh %*"
exit /b %errorlevel%
