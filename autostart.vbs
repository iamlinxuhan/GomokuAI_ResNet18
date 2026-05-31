' ============================================
' 五子棋AI静默训练 - 开机自启启动器
' 使用 pythonw.exe 运行，无控制台窗口
' ============================================
Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

' 获取脚本所在目录
scriptDir = objFSO.GetParentFolderName(WScript.ScriptFullName)

' Python 路径（优先使用 venv）
pythonExe = scriptDir & "\venv\Scripts\pythonw.exe"
If Not objFSO.FileExists(pythonExe) Then
    pythonExe = "pythonw"
End If

' 训练脚本路径
trainScript = scriptDir & "\train_service.py"
configFile = scriptDir & "\config.yaml"

' 日志目录
logDir = scriptDir & "\logs"
If Not objFSO.FolderExists(logDir) Then
    objFSO.CreateFolder(logDir)
End If

' 运行训练服务（静默模式，窗口隐藏）
cmd = """" & pythonExe & """ """ & trainScript & """ --config """ & configFile & """"
objShell.Run cmd, 0, False
