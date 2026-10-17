; Inno Setup script for IC-Inspection-Setup.exe. Built by build_app.ps1 after
; PyInstaller has filled dist\IC Inspection\:
;
;   ISCC.exe /DAppVersion=0.1.0 packaging\installer.iss
;
; Installs per user (no admin) into %LOCALAPPDATA%\Programs\IC Inspection by
; default. The app writes results\ beside its exe, so the folder must stay
; writable: the folder page refuses one that is not (e.g. Program Files).

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#define AppName "IC Inspection"
#define AppExe "IC Inspection.exe"
#define Source "..\dist\IC Inspection"

[Setup]
AppId={{6B0E3B7A-4F2D-4C61-9A57-1C2E8F4D9A31}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=no
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#AppExe}
OutputDir=..\dist
OutputBaseFilename=IC-Inspection-Setup
Compression=lzma2/normal
SolidCompression=yes
LZMAUseSeparateProcess=yes
WizardStyle=modern
ShowLanguageDialog=auto
CloseApplications=yes
#ifdef DiskSpan
; Over ~2 GB compressed one exe cannot hold it all: Setup.exe + Setup-N.bin,
; which must be handed out together.
DiskSpanning=yes
DiskSliceSize=max
#endif

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "th"; MessagesFile: "compiler:Languages\Thai.isl"

[CustomMessages]
en.NotWritable=Setup cannot write to this folder:%n%n%1%n%nIC Inspection keeps its results beside the program, so it needs a folder you can write to. Choose another folder (the suggested one works).
th.NotWritable=ไม่สามารถเขียนไฟล์ลงโฟลเดอร์นี้ได้:%n%n%1%n%nIC Inspection บันทึกผลการตรวจไว้ข้างตัวโปรแกรม จึงต้องติดตั้งในโฟลเดอร์ที่เขียนไฟล์ได้ กรุณาเลือกโฟลเดอร์อื่น (โฟลเดอร์ที่ตั้งไว้ให้ใช้ได้)

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#Source}\*"; DestDir: "{app}"; Excludes: "configs\paths.yaml"; \
    Flags: ignoreversion recursesubdirs createallsubdirs
; Where the dataset lives is the user's own setting: keep it across upgrades.
Source: "{#Source}\configs\paths.yaml"; DestDir: "{app}\configs"; Flags: onlyifdoesntexist uninsneveruninstall

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
function CanWriteTo(Dir: String): Boolean;
var
  Probe: String;
  Created: Boolean;
begin
  Created := not DirExists(Dir);
  if Created and not ForceDirectories(Dir) then
  begin
    Result := False;
    exit;
  end;
  Probe := AddBackslash(Dir) + '.write-test';
  Result := SaveStringToFile(Probe, 'ok', False);
  DeleteFile(Probe);
  if Created then
    RemoveDir(Dir);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = wpSelectDir) and not CanWriteTo(WizardDirValue) then
  begin
    MsgBox(FmtMessage(CustomMessage('NotWritable'), [WizardDirValue]), mbError, MB_OK);
    Result := False;
  end;
end;
