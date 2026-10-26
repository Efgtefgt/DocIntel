; Inno Setup script for DocIntel. Built by packaging\build_installer.ps1, which passes
; /DAppVersion, /DStageDir (the staged program files) and /DOutputDir.
; Installs per user (no administrator rights) to %LOCALAPPDATA%\Programs\DocIntel.
; User data (%LOCALAPPDATA%\DocIntel, Documents\DocIntel) is never removed by upgrades or uninstall.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef StageDir
  #error StageDir is required (run packaging\build_installer.ps1)
#endif
#ifndef OutputDir
  #define OutputDir "..\dist"
#endif

[Setup]
; Never change AppId: upgrades find the existing installation through it.
AppId={{778f2bab-2b82-42dc-b84a-82bf7bfc2ad0}
AppName=DocIntel
AppVersion={#AppVersion}
AppVerName=DocIntel {#AppVersion}
AppComments=Local, offline image-to-text and document intelligence
DefaultDirName={localappdata}\Programs\DocIntel
DefaultGroupName=DocIntel
DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutputDir}
OutputBaseFilename=DocIntel-Setup-{#AppVersion}
SetupIconFile={#StageDir}\DocIntel.ico
UninstallDisplayIcon={app}\DocIntel.ico
UninstallDisplayName=DocIntel
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
WizardStyle=modern
CloseApplications=yes
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; Remove the previous version's program files first, so no stale module survives an upgrade.
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{app}\tools"

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\DocIntel"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\DocIntel.pyw"""; \
  WorkingDir: "{app}\app"; IconFilename: "{app}\DocIntel.ico"; Comment: "Image to text and document intelligence (runs offline)"
Name: "{autoprograms}\DocIntel - Install PaddleOCR (optional)"; Filename: "{app}\python\python.exe"; \
  Parameters: """{app}\app\install_paddle.py"""; WorkingDir: "{app}\app"; IconFilename: "{app}\DocIntel.ico"; \
  Comment: "Download the optional PaddleOCR engine (about 700 MB, needs internet once)"
Name: "{autodesktop}\DocIntel"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\DocIntel.pyw"""; \
  WorkingDir: "{app}\app"; IconFilename: "{app}\DocIntel.ico"; Comment: "Image to text and document intelligence (runs offline)"; \
  Tasks: desktopicon

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\DocIntel.pyw"""; WorkingDir: "{app}\app"; \
  Description: "Start DocIntel"; Flags: postinstall nowait skipifsilent

[UninstallRun]
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\DocIntel.pyw"" --stop"; WorkingDir: "{app}\app"; \
  Flags: runhidden waituntilterminated; RunOnceId: "StopDocIntel"

[UninstallDelete]
; Python writes __pycache__ folders at run time; remove whatever is left in the program folder.
Type: filesandordirs; Name: "{app}"

[Code]
// Stop a running DocIntel before files are replaced during an upgrade.
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
  Launcher: String;
begin
  Launcher := ExpandConstant('{app}\app\DocIntel.pyw');
  if FileExists(ExpandConstant('{app}\python\pythonw.exe')) and FileExists(Launcher) then
    Exec(ExpandConstant('{app}\python\pythonw.exe'), '"' + Launcher + '" --stop', '', SW_HIDE,
         ewWaitUntilTerminated, ResultCode);
  Result := '';
end;
