@echo off
REM ============================================
REM 五子棋AI训练服务 - 卸载脚本 (Windows)
REM ============================================

echo ============================================
echo 五子棋AI训练服务卸载脚本
echo ============================================

REM 检查管理员权限
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 请以管理员身份运行此脚本！
    pause
    exit /b 1
)

REM 停止服务
echo 正在停止服务...
net stop GomokuAI_Trainer 2>nul

REM 删除服务
where nssm >nul 2>&1
if %errorlevel% equ 0 (
    set NSSM=nssm
) else if exist ".\nssm.exe" (
    set NSSM=.\nssm.exe
) else (
    echo [错误] nssm 未找到
    echo 请手动删除服务: sc delete GomokuAI_Trainer
    pause
    exit /b 1
)

echo 正在删除服务...
%NSSM% remove GomokuAI_Trainer confirm

if %errorlevel% equ 0 (
    echo ============================================
    echo 服务已成功卸载
    echo ============================================
) else (
    echo [错误] 服务卸载失败
    echo 请尝试: sc delete GomokuAI_Trainer
)

pause
