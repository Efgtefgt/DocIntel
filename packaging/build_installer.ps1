<#
.SYNOPSIS
  Build the DocIntel Windows installer (dist\DocIntel-Setup-<version>.exe).

.DESCRIPTION
  The installer needs nothing else on the target PC: it contains a private Python 3.10
  (the official "embeddable package"), DocIntel's Python libraries, Tesseract OCR and
  Ghostscript. It installs for the current user only (no administrator rights), adds
  Start-menu (and optional desktop) shortcuts and an uninstaller.

  Needs on the BUILD PC:
    * Python 3.10 64-bit (the same minor version as the bundled one, so wheels match)
    * Tesseract and Ghostscript installed (they are copied from their install folders)
    * Inno Setup 6 (winget install JRSoftware.InnoSetup --scope user)
    * Internet access the first time (downloads the embeddable Python and the wheels)

  Run from the project folder:
      powershell -ExecutionPolicy Bypass -File packaging\build_installer.ps1
      powershell -ExecutionPolicy Bypass -File packaging\build_installer.ps1 -SkipCompile   (stage only)

  PaddleOCR is not bundled (it would add ~1 GB); the installed app uses Tesseract.
#>
[CmdletBinding()]
param(
    [string]$HostPython = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    [string]$TestedPython = "$env:USERPROFILE\.venvs\docintel\Scripts\python.exe",
    [string]$TesseractDir = "$env:ProgramFiles\Tesseract-OCR",
    [string]$GhostscriptDir = "",
    [string]$BuildDir = "$env:LOCALAPPDATA\DocIntelBuild",
    [switch]$SkipCompile
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$PythonVersion = "3.10.11"
$EmbedUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
# SHA-256 of the file above (its MD5 f1c0538b060e03cbb697ab3581cb73bc matches python.org's release data).
$EmbedSha256 = "608619F8619075629C9C69F361352A0DA6ED7E62F83A0E19C63E0EA32EB7629D"

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Check($ok, $text) { if (-not $ok) { throw $text } }

# The build folder lives outside OneDrive: it holds ~15,000 files that must not sync.
$Stage = Join-Path $BuildDir "stage"
$Cache = Join-Path $BuildDir "cache"
$Dist = Join-Path $ProjectRoot "dist"

Step "Checking build tools"
Check (Test-Path $HostPython) "Python 3.10 not found at $HostPython (pass -HostPython)."
$hostVersion = & $HostPython -c "import sys, struct; print('%d.%d.%d-%d' % (*sys.version_info[:3], struct.calcsize('P') * 8))"
Check ($hostVersion -like "3.10.*-64") "Build Python must be 3.10 64-bit (found $hostVersion)."
Check (Test-Path "$TesseractDir\tesseract.exe") "Tesseract not found in $TesseractDir (pass -TesseractDir)."
if (-not $GhostscriptDir) {
    $GhostscriptDir = Get-ChildItem "$env:ProgramFiles\gs" -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
}
Check ($GhostscriptDir -and (Test-Path "$GhostscriptDir\bin\gswin64c.exe")) "Ghostscript not found (pass -GhostscriptDir)."
$Iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $SkipCompile) { Check $Iscc "Inno Setup 6 not found. Install it: winget install JRSoftware.InnoSetup --scope user" }
$Version = & $HostPython -c "import runpy; print(runpy.run_path(r'$ProjectRoot\docintel\__init__.py')['__version__'])"
Write-Host "DocIntel $Version | Python $hostVersion | Tesseract $TesseractDir | Ghostscript $GhostscriptDir"

Step "Preparing a clean build folder ($BuildDir)"
if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory -Force $Stage, $Cache, $Dist | Out-Null

Step "Embeddable Python $PythonVersion"
$zip = Join-Path $Cache "python-$PythonVersion-embed-amd64.zip"
if (-not (Test-Path $zip)) { Invoke-WebRequest $EmbedUrl -OutFile $zip -UseBasicParsing }
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash
Check ($hash -eq $EmbedSha256) "Checksum mismatch for $zip (got $hash). Delete it and retry."
& $HostPython -m zipfile -e $zip (Join-Path $Stage "python")   # Expand-Archive crashes in some PowerShell 5.1 setups
Check ($LASTEXITCODE -eq 0) "Could not unpack $zip."
# Search path of the private Python: standard library, its site-packages and the app folder.
@("python310.zip", ".", "Lib\site-packages", "..\app", "import site") |
    Set-Content (Join-Path $Stage "python\python310._pth") -Encoding ascii

Step "Python libraries (requirements.txt + requirements-optional.txt)"
$sitePackages = Join-Path $Stage "python\Lib\site-packages"
# Pin every indirect dependency to the versions the test suite passed with (the main venv);
# otherwise pip would pick the newest releases, which were never tested with DocIntel.
$constraints = Join-Path $Cache "constraints.txt"
$pinArgs = @()
if (Test-Path $TestedPython) {
    & $TestedPython -m pip freeze --exclude-editable | Where-Object { $_ -match '^[A-Za-z0-9_.\-]+==' } |
        Set-Content $constraints -Encoding ascii
    $pinArgs = @("-c", $constraints)
    Write-Host "Pinned to the versions in $TestedPython"
} else {
    Write-Warning "Tested environment $TestedPython not found; indirect dependencies are not pinned."
}
# (pip may print "dependency conflicts" for packages of the BUILD Python; they do not affect the bundle.)
& $HostPython -m pip install --disable-pip-version-check --no-warn-script-location --only-binary=:all: `
    --target $sitePackages -r "$ProjectRoot\requirements.txt" -r "$ProjectRoot\requirements-optional.txt" @pinArgs
Check ($LASTEXITCODE -eq 0) "pip install failed."
Remove-Item (Join-Path $sitePackages "bin") -Recurse -Force -ErrorAction SilentlyContinue

Step "Application files"
$app = Join-Path $Stage "app"
New-Item -ItemType Directory -Force $app | Out-Null
Copy-Item "$ProjectRoot\app.py", "$ProjectRoot\README.md" $app
Copy-Item "$PSScriptRoot\DocIntel.pyw" $app
foreach ($dir in "docintel", "app_pages", ".streamlit", "docs") { Copy-Item "$ProjectRoot\$dir" $app -Recurse }
New-Item -ItemType Directory -Force "$app\data\samples\pdf" | Out-Null
Copy-Item "$ProjectRoot\data\templates" "$app\data" -Recurse
Copy-Item "$ProjectRoot\data\samples\incident_report.png" "$app\data\samples"   # "Try a sample" buttons
Copy-Item "$ProjectRoot\data\samples\pdf\mixed.pdf" "$app\data\samples\pdf"
Get-ChildItem $app -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

Step "Tesseract (runtime files only) and Ghostscript"
$tess = Join-Path $Stage "tools\tesseract"
New-Item -ItemType Directory -Force $tess | Out-Null
Copy-Item "$TesseractDir\tesseract.exe", "$TesseractDir\*.dll" $tess
Copy-Item "$TesseractDir\tessdata" $tess -Recurse
Copy-Item "$TesseractDir\doc" "$tess\doc" -Recurse          # LICENSE (Apache-2.0), AUTHORS
$gs = Join-Path $Stage "tools\ghostscript"
New-Item -ItemType Directory -Force "$gs\bin", "$gs\doc" | Out-Null
Copy-Item "$GhostscriptDir\bin\gswin64c.exe", "$GhostscriptDir\bin\gsdll64.dll" "$gs\bin"
foreach ($dir in "lib", "Resource", "iccprofiles") { Copy-Item "$GhostscriptDir\$dir" $gs -Recurse }
Copy-Item "$GhostscriptDir\doc\COPYING" "$gs\doc" -ErrorAction SilentlyContinue  # AGPL-3.0
Copy-Item "$PSScriptRoot\THIRD_PARTY_NOTICES.txt" $Stage

Step "Icon"
$icon = Join-Path $Stage "DocIntel.ico"
& $HostPython "$PSScriptRoot\make_icon.py" "$ProjectRoot\docintel\ui\assets\icon.svg" $icon
Check ($LASTEXITCODE -eq 0 -and (Test-Path $icon)) "Icon creation failed (needs PyMuPDF and Pillow in the build Python)."

Step "Pre-compiling Python files (faster first start)"
$py = Join-Path $Stage "python\python.exe"
& $py -m compileall -q -j 0 $sitePackages $app | Out-Null   # a few test files in libraries do not compile; harmless

Step "Smoke test of the staged app"
& $py "$PSScriptRoot\smoke_test.py" $Stage
Check ($LASTEXITCODE -eq 0) "Smoke test failed; the installer was not built."

if ($SkipCompile) { Write-Host "`nStaged in $Stage (installer not compiled)."; exit 0 }

Step "Compiling the installer with Inno Setup"
& $Iscc /Q "/DAppVersion=$Version" "/DStageDir=$Stage" "/DOutputDir=$Dist" "$PSScriptRoot\DocIntel.iss"
Check ($LASTEXITCODE -eq 0) "Inno Setup failed."
$setup = Join-Path $Dist "DocIntel-Setup-$Version.exe"
$size = [math]::Round((Get-Item $setup).Length / 1MB)
Write-Host "`nBuilt $setup ($size MB)" -ForegroundColor Green
