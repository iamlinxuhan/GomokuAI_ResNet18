@echo off
REM ============================================
REM 五子棋AI训练服务 - 安装脚本 (Windows)
REM 使用 nssm 将训练脚本注册为Windows服务
REM ============================================

echo ============================================
echo 五子棋AI训练服务安装脚本
echo ============================================

REM 检查管理员权限
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 请以管理员身份运行此脚本！
    pause
    exit /b 1
)

REM 检查 nssm
where nssm >nul 2>&1
if %errorlevel% neq 0 (
    echo [信息] nssm 未安装，正在下载...
    
    REM 创建临时目录
    if not exist ".\temp" mkdir temp
    
    REM 下载 nssm
    powershell -Command "Invoke-WebRequest -Uri 'https://nssm.cc/release/nssm-2.24.zip' -OutFile '.\temp\nssm.zip'"
    if %errorlevel% neq 0 (
        echo [错误] 下载 nssm 失败，请手动下载并放置到 PATH 中
        echo 下载地址: https://nssm.cc/download
        pause
        exit /b 1
    )
    
    REM 解压
    powershell -Command "Expand-Archive -Path '.\temp\nssm.zip' -DestinationPath '.\temp\nssm' -Force"
    
    REM 复制 nssm.exe 到当前目录
    copy ".\temp\nssm\nssm-2.24\win64\nssm.exe" .\nssm.exe
    if %errorlevel% neq 0 (
        echo [错误] 复制 nssm 失败
        pause
        exit /b 1
    )
    
    echo [信息] nssm 已下载到当前目录
    set NSSM=.\nssm.exe
) else (
    set NSSM=nssm
)

REM 获取 Python 路径
for /f "tokens=*" %%i in ('where python') do set PYTHON_PATH=%%i
echo Python路径: %PYTHON_PATH%

REM 获取脚本路径
set SCRIPT_DIR=%~dp0
set SERVICE_SCRIPT=%SCRIPT_DIR%train_service.py
set CONFIG_FILE=%SCRIPT_DIR%config.yaml

echo 服务脚本: %SERVICE_SCRIPT%
echo 配置文件: %CONFIG_FILE%

REM 检查文件
if not exist "%SERVICE_SCRIPT%" (
    echo [错误] train_service.py 未找到！
    pause
    exit /b 1
)

REM 停止并删除已有服务
%NSSM% stop GomokuAI_Trainer 2>nul
%NSSM% remove GomokuAI_Trainer confirm 2>nul

REM 安装服务
echo 正在安装服务 GomokuAI_Trainer...
%NSSM% install GomokuAI_Trainer "%PYTHON_PATH%" "%SERVICE_SCRIPT% --config %CONFIG_FILE%"

REM 设置服务描述
%NSSM% set GomokuAI_Trainer Description "五子棋AI神经网络训练服务 - 后台自我对弈训练"
%NSSM% set GomokuAI_Trainer DisplayName "GomokuAI Trainer"

REM 设置工作目录
%NSSM% set GomokuAI_Trainer AppDirectory "%SCRIPT_DIR%"

REM 设置启动类型为自动
%NSSM% set GomokuAI_Trainer Start SERVICE_AUTO_START

REM 设置重启策略
%NSSM% set GomokuAI_Trainer AppExit Default Restart
%NSSM% set GomokuAI_Trainer AppRestartDelay 60000

REM 启动服务
echo 正在启动服务...
%NSSM% start GomokuAI_Trainer

if %errorlevel% equ 0 (
    echo ============================================
    echo 服务安装成功！
    echo 服务名称: GomokuAI_Trainer
    echo 启动类型: 自动（开机自启）
    echo ============================================
    echo 管理命令:
    echo   查看状态: nssm status GomokuAI_Trainer
    echo   停止服务: net stop GomokuAI_Trainer
    echo   启动服务: net start GomokuAI_Trainer
    echo   卸载服务: uninstall_service.bat
    echo   查看日志: type logs\training.log
    echo ============================================
) else (
    echo [错误] 服务启动失败，请检查日志
)

pause
