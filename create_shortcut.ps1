$WshShell = New-Object -comObject WScript.Shell
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$Shortcut = $WshShell.CreateShortcut("$DesktopPath\Minervini Pro V2.lnk")
$Shortcut.TargetPath = "wscript.exe"
$Shortcut.Arguments = "`"C:\Users\kiike\.gemini\antigravity\projects\minervini-pro\run_minervini_silent.vbs`""
$Shortcut.IconLocation = "C:\Users\kiike\.gemini\antigravity\projects\minervini-pro\app_icon.ico"
$Shortcut.WorkingDirectory = "C:\Users\kiike\.gemini\antigravity\projects\minervini-pro\code"
$Shortcut.Save()
